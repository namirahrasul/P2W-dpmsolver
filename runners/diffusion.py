import os
import logging
import time
import glob

import blobfile as bf
import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torch.distributed as dist
import torchvision.utils as tvu
import torch.nn.functional as F

from models.p2_weighing.unet import UNetModel as LWDM_Model
from models.p2_weighing.image_datasets import load_data
from models.p2_weighing import logger
from dpm_solver.sampler import NoiseScheduleVP, model_wrapper, DPM_Solver
from metrics_cal import *
from datasets import get_dataset, data_transform, inverse_data_transform

def load_data_for_worker(base_samples, batch_size, cond_class):
    with bf.BlobFile(base_samples, "rb") as f:
        obj = np.load(f)
        image_arr = obj["arr_0"]
        if cond_class:
            label_arr = obj["arr_1"]
    buffer = []
    label_buffer = []
    while True:
        for i in range(len(image_arr)):
            buffer.append(image_arr[i])
            if cond_class:
                label_buffer.append(label_arr[i])
            if len(buffer) == batch_size:
                batch = torch.from_numpy(np.stack(buffer)).float()
                batch = batch / 127.5 - 1.0
                batch = batch.permute(0, 3, 1, 2)
                res = dict(low_res=batch)
                if cond_class:
                    res["y"] = torch.from_numpy(np.stack(label_buffer))
                yield res
                buffer, label_buffer = [], []

def load_reference(data_dir, batch_size, image_size, class_cond=False):
    """
    Load reference images or masks from a directory using load_data.
    Args:
        data_dir: Directory containing .png images.
        batch_size: Number of images per batch.
        image_size: Target image size (height, width).
    Yields:
        Dictionary with 'ref_img' containing a batch of images.
    """
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        class_cond=False,
        deterministic=True,
        random_flip=False,
    )
    for large_batch, model_kwargs in data:
        model_kwargs["ref_img"] = large_batch
        yield model_kwargs

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def upsample_image(image_tensor, target_size=(256, 256)):
    image_tensor = (image_tensor + 1) / 2.0
    image_tensor = image_tensor.clamp(0, 1)
    upscaled_image_tensor = F.interpolate(image_tensor, size=target_size, mode='bilinear', align_corners=False)
    upscaled_image_tensor = upscaled_image_tensor * 2.0 - 1
    return upscaled_image_tensor

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2,
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd": # 1/T, 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Diffusion(object):
    def __init__(self, args, config, rank=None):
        self.args = args
        self.config = config
        if rank is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            device = rank
            self.rank = rank
        self.device = device

        self.model_var_type = self.config.model.var_type 
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_cumprod = alphas_cumprod
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def create_dpm_solver(self, model, config, x, base_samples=None, model_kwargs=None, classifier=None, classifier_scale=0.0):
        """Create a DPM_Solver instance with NoiseScheduleVP and necessary functions for noise addition."""

        # Instantiate NoiseScheduleVP
        noise_schedule = NoiseScheduleVP(schedule="discrete", alphas_cumprod=self.alphas_cumprod.to(self.device))

        # Handle classifier and conditional sampling (moved verbatim from sample_image)
        if self.config.sampling.cond_class:
            if base_samples and "y" in base_samples:
                classes = base_samples["y"].to(self.device)
            else:
                classes = torch.randint(low=0, high=config.data.num_classes, size=(x.shape[0],)).to(self.device)
        else:
            classes = None

    # Prepare model_kwargs (moved verbatim from sample_image)
        if base_samples is None:
            model_kwargs = {} if model_kwargs is None else model_kwargs
            if classes is not None:
                model_kwargs = {**model_kwargs, "y": classes}
        else:
            model_kwargs = {"y": base_samples["y"], "low_res": base_samples["low_res"]} if model_kwargs is None else {**model_kwargs, "y": base_samples["y"], "low_res": base_samples["low_res"]}

        # Define model_fn as in sample_image
        def model_fn(x, t, **model_kwargs):
            out = model(x, t, **model_kwargs)
            if self.config.model.out_channels == 6:
                out = torch.split(out, 3, dim=1)[0]
            return out

        # Define classifier_fn as in sample_image
        def classifier_fn(x, t, y, **classifier_kwargs):
            if classifier is None:
                return None
            logits = classifier(x, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            return log_probs[range(len(logits)), y.view(-1)]

        # Create model_fn_continuous as in sample_image
        model_fn_continuous = model_wrapper(
            model_fn,
            noise_schedule,
            model_type="noise",
            model_kwargs=model_kwargs,
            guidance_type="uncond" if classifier is None else "classifier",
            condition=model_kwargs["y"] if "y" in model_kwargs else None,
            guidance_scale=0.0,
            classifier_fn=classifier_fn,
            classifier_kwargs={},
        )

        # Instantiate DPM_Solver
        dpm_solver = DPM_Solver(
            model_fn_continuous,
            noise_schedule,
            algorithm_type=self.args.sample_type,
            correcting_x0_fn="dynamic_thresholding" if self.config.sampling.thresholding else None
        )
        return dpm_solver, noise_schedule, classes, model_kwargs
    def sample(self):
        start_time = time.time()
        logger.configure(dir=self.args.image_folder)
        logger.log("creating model...")
        #config = self.config
        if self.config.model.model_type == "p2-weighing":
            model_64 = LWDM_Model(
                image_size=self.config.model.image_size_coarse,
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions_coarse,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult_coarse,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                num_classes=self.config.model.num_classes,
                use_checkpoint=self.config.model.use_checkpoint,
                use_fp16=self.config.model.use_fp16,
                num_heads=self.config.model.num_heads,
                num_head_channels=self.config.model.num_head_channels,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                resblock_updown=self.config.model.resblock_updown,
                use_new_attention_order=self.config.model.use_new_attention_order
            )
            model_256 = LWDM_Model(
                image_size=self.config.model.image_size_fine,
                in_channels=self.config.model.in_channels,
                model_channels=self.config.model.model_channels,
                out_channels=self.config.model.out_channels,
                num_res_blocks=self.config.model.num_res_blocks,
                attention_resolutions=self.config.model.attention_resolutions,
                dropout=self.config.model.dropout,
                channel_mult=self.config.model.channel_mult_fine,
                conv_resample=self.config.model.conv_resample,
                dims=self.config.model.dims,
                num_classes=self.config.model.num_classes,
                use_checkpoint=self.config.model.use_checkpoint,
                use_fp16=self.config.model.use_fp16,
                num_heads=self.config.model.num_heads,
                num_head_channels=self.config.model.num_head_channels,
                num_heads_upsample=self.config.model.num_heads_upsample,
                use_scale_shift_norm=self.config.model.use_scale_shift_norm,
                resblock_updown=self.config.model.resblock_updown,
                use_new_attention_order=self.config.model.use_new_attention_order
            )

            model_64 = model_64.to(self.rank)
            model_256 = model_256.to(self.rank)
            map_location = {"cuda:%d" % 0: "cuda:%d" % self.rank}

            if "ckpt_dir_coarse" in self.config.model.__dict__.keys() and "ckpt_dir_fine" in self.config.model.__dict__.keys():
                ckpt_dir_coarse = os.path.expanduser(self.args.model_path_64)
                ckpt_dir_fine = os.path.expanduser(self.args.model_path_256)
                states_coarse = torch.load(
                    ckpt_dir_coarse,
                    map_location=map_location
                )
                states_fine = torch.load(
                    ckpt_dir_fine,
                    map_location=map_location
                )
                model_64.load_state_dict(states_coarse, strict=True)
                model_256.load_state_dict(states_fine, strict=True)
                if self.config.model.use_fp16:
                    model_64.convert_to_fp16()
                    model_256.convert_to_fp16()
            else:
                raise NotImplementedError("ckpt_dir_coarse or ckpt_dir_fine not defined")

            classifier = None

            model_64.eval()
            model_256.eval()


            if self.config.model.is_upsampling:
                base_samples_total = load_data_for_worker(self.args.base_samples, self.config.sampling.batch_size, self.config.sampling.cond_class)

            elif self.args.base_samples and self.args.mask_path:
                logger.log("loading data...")
                # Load data for both resolutions
                ref_data_64 = load_reference(
                    self.args.base_samples,
                    self.config.sampling.batch_size,
                    self.config.model.image_size_coarse,
                    class_cond=self.config.sampling.cond_class,
                )
                mask_data_64 = load_reference(
                    self.args.mask_path,
                    self.config.sampling.batch_size,
                    self.config.model.image_size_coarse,
                    class_cond=self.config.sampling.cond_class,
                )
                ref_data_256 = load_reference(
                    self.args.base_samples,
                    self.config.sampling.batch_size,
                    self.config.model.image_size_fine,
                    class_cond=self.config.sampling.cond_class,
                )
                mask_data_256 = load_reference(
                    self.args.mask_path,
                    self.config.sampling.batch_size,
                    self.config.model.image_size_fine,
                    class_cond=self.config.sampling.cond_class,
                )
                # Metrics initialization
                metrics_file_path = os.path.join(self.args.image_folder, "metrics_log.txt")
                lpips_value = 0.
                coarse_lpips_value = 0.
                psnr_value = 0.
                coarse_psnr_value = 0.
                ssim_value = 0.
                coarse_ssim_value = 0.
                l1_value = 0.
                coarse_l1_value = 0.

                # Log conditions
                with open(metrics_file_path, "a") as metrics_file:
                    metrics_file.write(f"Condition:\n")
                    metrics_file.write(f"\tmask_path: {self.args.mask_path}\n")
                    metrics_file.write(f"\ttimesteps_coarse: {self.args.timesteps_coarse}\n")
                    metrics_file.write(f"\ttimesteps_fine: {self.args.timesteps}\n")
                    metrics_file.write(f"\tskip_type: {self.args.skip_type}\n")
                    metrics_file.write(f"\tsample_type: {self.args.sample_type}\n")
                    metrics_file.write(f"\n")

                logger.log("creating samples...")
                count = 0
                all_items = os.listdir(self.args.base_samples)
                num_inputs = self.config.sampling.total_N
                # Define coarse and fine schedule params from config
                schedule_jump_params_coarse = {
                    "t_T": self.args.timesteps_coarse,
                    "n_sample": self.config.sampling.n_sample,
                    "jump_length": self.config.sampling.jump_length_coarse,
                    "jump_n_sample": self.config.sampling.jump_n_sample_coarse,
                    "jump_interval": self.config.sampling.jump_interval_coarse
                }
                ddim_step = self.config.sampling.ddim_step

                schedule_jump_params_fine = {
                    "t_T": self.args.timesteps,
                    "n_sample": self.config.sampling.n_sample,
                    "jump_length": self.config.sampling.jump_length_fine,
                    "jump_n_sample": self.config.sampling.jump_n_sample_fine,
                    "jump_interval": self.config.sampling.jump_interval_fine
                }

                while count < num_inputs:
                    model_mask_kwargs_64 = next(mask_data_64)
                    model_kwargs_64 = next(ref_data_64)
                    model_mask_kwargs_64 = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in model_mask_kwargs_64.items()}
                    model_kwargs_64 = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs_64.items()}
                    model_mask_kwargs_256 = next(mask_data_256)
                    model_kwargs_256 = next(ref_data_256)
                    model_mask_kwargs_256 = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in model_mask_kwargs_256.items()}
                    model_kwargs_256 = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in model_kwargs_256.items()}
                    gt = model_kwargs_256["ref_img"]
                    mask = model_mask_kwargs_256["ref_img"]
                    #print("gt shape: {}".format(gt.shape))
                    #print("mask shape: {}".format(mask.shape))

                    if self.args.use_inverse_masks:
                        model_mask_kwargs_64["ref_img"] = model_mask_kwargs_64["ref_img"] * (-1)
                        model_mask_kwargs_256["ref_img"] = model_mask_kwargs_256["ref_img"] * (-1)

                    base_samples=None
                    mask = (mask > 0).float()
                    x_64 = torch.randn(self.config.sampling.batch_size, self.config.data.channels, self.config.model.image_size_coarse, self.config.model.image_size_coarse, device=self.device)
                    dpm_solver_64, noise_schedule_64, classes_64, _ = self.create_dpm_solver(model_64, self.config, x_64, base_samples, model_kwargs_64, classifier=None, classifier_scale=0.0)
                    sample_coarse = self.sample_image(x_64, model_64,timesteps=self.args.timesteps_coarse, model_kwargs=model_kwargs_64, model_mask_kwargs=model_mask_kwargs_64, inpa_inj_sched_prev=self.args.inpa_inj_sched_prev, inpa_inj_sched_prev_cumnoise=self.args.inpa_inj_sched_prev_cumnoise, dpm_solver=dpm_solver_64, schedule_jump_params=schedule_jump_params_coarse, ddim_step=ddim_step)
                    sample_coarse_256 = upsample_image(sample_coarse)

                    #t_T_fine = torch.tensor([self.args.timesteps], device=self.device)
                    # Prepare noise for fine inpainting using DPM_Solver's add_noise
                    t_T_fine = torch.tensor([self.args.timesteps] * self.config.sampling.batch_size, device=self.device)
                    noise = torch.randn_like(sample_coarse_256)
                    dpm_solver_256, noise_schedule_256, classes_256, _ = self.create_dpm_solver(model_256, self.config, sample_coarse_256, base_samples, model_kwargs_256, classifier=None, classifier_scale=0.0)
                    noised_coarse_256 = dpm_solver_256.add_noise(sample_coarse_256, t_T_fine, noise)

                    # Update model_kwargs for fine inpainting
                    model_fine_kwargs = model_kwargs_256
                    model_fine_kwargs["ref_img"] = sample_coarse_256 * (1 - mask) + model_kwargs_256["ref_img"] * mask 
                    sample_fine = self.sample_image(noised_coarse_256, model_256,timesteps=self.args.timesteps, model_kwargs=model_fine_kwargs, model_mask_kwargs=model_mask_kwargs_256, inpa_inj_sched_prev=self.args.inpa_inj_sched_prev, inpa_inj_sched_prev_cumnoise=self.args.inpa_inj_sched_prev_cumnoise, dpm_solver=dpm_solver_256, schedule_jump_params=schedule_jump_params_fine, ddim_step=ddim_step)
                    logger.log("sample_fine completed.") 
                    # Save images and calculate metrics
                    for i in range(self.config.sampling.batch_size):
                        os.makedirs(os.path.join(self.args.image_folder, "gtImg"), exist_ok=True)
                        os.makedirs(os.path.join(self.args.image_folder, "inputImg"), exist_ok=True)
                        os.makedirs(os.path.join(self.args.image_folder, "sampledImg"), exist_ok=True)
                        os.makedirs(os.path.join(self.args.image_folder, "outImg"), exist_ok=True)
                        os.makedirs(os.path.join(self.args.image_folder, "coarseImg"), exist_ok=True)

                        out_gtImg_path = os.path.join(self.args.image_folder, "gtImg", f"{str(count + i).zfill(4)}.png")
                        out_inputImg_path = os.path.join(self.args.image_folder, "inputImg", f"{str(count + i).zfill(4)}.png")
                        out_sampledImg_path = os.path.join(self.args.image_folder, "sampledImg", f"{str(count + i).zfill(4)}.png")
                        out_outImg_path = os.path.join(self.args.image_folder, "outImg", f"{str(count + i).zfill(4)}.png")
                        out_coarseImg_path = os.path.join(self.args.image_folder, "coarseImg", f"{str(count + i).zfill(4)}.png")

                        tmp_ones = torch.ones_like(gt[i]) * (-1)
                        inputImg = gt[i] * mask[i] + (1 - mask[i]) * tmp_ones
                        sampledImg = sample_fine[i].unsqueeze(0)
                        outImg = mask[i] * inputImg + (1 - mask[i]) * sampledImg
                        coarseImg = sample_coarse_256[i].unsqueeze(0)
                        out_coarseImg = mask[i] * inputImg + (1 - mask[i]) * coarseImg
                        gtImg = gt[i].reshape(outImg.shape).to(outImg.device)

                        tvu.save_image(inverse_data_transform(self.config, gtImg), out_gtImg_path, nrow=1)
                        tvu.save_image(inverse_data_transform(self.config, inputImg), out_inputImg_path, nrow=1)
                        tvu.save_image(inverse_data_transform(self.config, sampledImg), out_sampledImg_path, nrow=1)
                        tvu.save_image(inverse_data_transform(self.config, outImg), out_outImg_path, nrow=1)
                        tvu.save_image(inverse_data_transform(self.config, out_coarseImg), out_coarseImg_path, nrow=1)

                    count += self.config.sampling.batch_size
                    with open(metrics_file_path, "a") as metrics_file:
                        metrics_file.write(f"Coarse {count} samples LPIPS: {coarse_lpips_value / count:.4f}\n")
                        metrics_file.write(f"Coarse {count} samples PSNR: {coarse_psnr_value / count:.4f}\n")
                        metrics_file.write(f"Coarse {count} samples SSIM: {coarse_ssim_value / count:.4f}\n")
                        metrics_file.write(f"Coarse {count} samples L1(%): {coarse_l1_value / count * 100:.2f}\n")
                        metrics_file.write(f"{count} samples LPIPS: {lpips_value / count:.4f}\n")
                        metrics_file.write(f"{count} samples PSNR: {psnr_value / count:.4f}\n")
                        metrics_file.write(f"{count} samples SSIM: {ssim_value / count:.4f}\n")
                        metrics_file.write(f"{count} samples L1(%): {l1_value / count * 100:.2f}\n")
                        metrics_file.write(f"\n")

                    logger.log(f"created {count} samples")

                logger.log("sampling complete")
                end_time = time.time()
                total_time = end_time - start_time
                each_time = total_time / count
                logger.log(f"Total time: {total_time}.")
                logger.log(f"Each time: {each_time}.")
                #metrics_file.write(f"{total_time}s for {count} samples\n")
                #metrics_file.write(f"{each_time}s for 1 sample\n")
                #metrics_file.write(f"\n")
            else:
                raise NotImplementedError("Base sample or mask path not defined")

        if self.args.fid:
            if not os.path.exists(os.path.join(self.args.exp, "fid.npy")):
                self.sample_fid(model_64, model_256, classifier=classifier)
                torch.distributed.barrier()
                if self.rank == 0:
                    print("Computed FID...")
                    #fid = calculate_fid_given_paths((self.config.sampling.fid_stats_dir, self.args.image_folder), batch_size=self.config.sampling.fid_batch_size, device=self.device, dims=2048, num_workers=4)
                    #print("FID: {}".format(fid))
                    #np.save(os.path.join(self.args.exp, "fid"), fid)
        elif self.args.sample_only:
            self.sample_n_images(model, classifier=classifier)
            torch.distributed.barrier()
            if self.rank == 0:
                print("Begin to compute samples...")
        #else:
            #raise NotImplementedError("Sample procedure not defined")
    def sample_fid(self, model_64,model_256, classifier=None):
        pass
    def sample_image(self, x, model, timesteps, last=True, model_kwargs=None, model_mask_kwargs=None, inpa_inj_sched_prev=True, inpa_inj_sched_prev_cumnoise=False, dpm_solver_order=3, skip_type="time_uniform", dpm_solver_method="singlestep", dpm_solver_type="dpmsolver", denoise=False, lower_order_final=False, thresholding=False, atol=0.0078, rtol=0.05, dpm_solver=None, schedule_jump_params=None, ddim_step=None):
        assert last
        #config = self.config
        # DPM-Solver sampling
        if self.args.sample_type in ["dpmsolver", "dpmsolver++"]:
            x = dpm_solver.sample(
                x,
                steps=(timesteps - 1 if denoise else timesteps),
                order=dpm_solver_order,
                skip_type=skip_type,
                method=dpm_solver_method,
                solver_type=dpm_solver_type,
                lower_order_final=lower_order_final,
                denoise_to_zero=denoise,
                atol=atol,
                rtol=rtol,
                model_kwargs=model_kwargs,
                model_mask_kwargs=model_mask_kwargs,
                inpa_inj_sched_prev=inpa_inj_sched_prev,
                inpa_inj_sched_prev_cumnoise=inpa_inj_sched_prev_cumnoise,
                schedule_jump_params=schedule_jump_params,
                ddim_step=ddim_step,
            )
        else:
            raise NotImplementedError(f"Sample type {self.args.sample_type} not supported")

        return x
    def test(self):
        pass
