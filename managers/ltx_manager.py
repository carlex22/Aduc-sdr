# managers/ltx_manager.py
# AducSdr: Uma implementação aberta e funcional da arquitetura ADUC-SDR
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
# Rua Eduardo Carlos Pereira, 4125, B1 Ap32, Curitiba, PR, Brazil, CEP 8102025
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
#
# PENDING PATENT NOTICE: Please see NOTICE.md.
#
# Version: 2.2.2
#
# This file manages the LTX-Video specialist pool. It has been refactored to be
# self-contained by automatically cloning its own dependencies and using a local
# utility module for pipeline creation, fully decoupling it from external scripts.

import torch
import gc
import os
import sys
import yaml
import logging
import huggingface_hub
import time
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple, Union

from tools.optimization import optimize_ltx_worker, can_optimize_fp8
from tools.hardware_manager import hardware_manager
from aduc_types import LatentConditioningItem

logger = logging.getLogger(__name__)

# --- Dependency Management ---
DEPS_DIR = Path("./deps")
LTX_VIDEO_REPO_DIR = DEPS_DIR / "LTX-Video"
LTX_VIDEO_REPO_URL = "https://github.com/Lightricks/LTX-Video.git"

# --- Placeholders for lazy-loaded modules ---
create_ltx_video_pipeline = None
calculate_padding = None
LTXVideoPipeline = None
ConditioningItem = None
LTXMultiScalePipeline = None
vae_encode = None
latent_to_pixel_coords = None
randn_tensor = None

class LtxPoolManager:
    """
    Manages a pool of LtxWorkers for optimized multi-GPU usage.
    Handles its own code dependencies by cloning the LTX-Video repository.
    """
    def __init__(self, device_ids, ltx_config_file_name):
        logger.info(f"LTX POOL MANAGER: Creating workers for devices: {device_ids}")
        self._ltx_modules_loaded = False
        self._setup_dependencies()
        self._lazy_load_ltx_modules()

        self.ltx_config_file = LTX_VIDEO_REPO_DIR / "configs" / ltx_config_file_name

        self.workers = [LtxWorker(dev_id, self.ltx_config_file) for dev_id in device_ids]
        self.current_worker_index = 0
        self.lock = threading.Lock()

        self._apply_ltx_pipeline_patches()

        if all(w.device.type == 'cuda' for w in self.workers):
            logger.info("LTX POOL MANAGER: HOT START MODE ENABLED. Pre-warming all GPUs...")
            for worker in self.workers:
                worker.to_gpu()
            logger.info("LTX POOL MANAGER: All GPUs are hot and ready.")
        else:
            logger.info("LTX POOL MANAGER: Operating in CPU or mixed mode. GPU pre-warming skipped.")

    def _setup_dependencies(self):
        """Clones the LTX-Video repo if not found and adds it to the system path."""
        if not LTX_VIDEO_REPO_DIR.exists():
            logger.info(f"LTX-Video repository not found at '{LTX_VIDEO_REPO_DIR}'. Cloning from GitHub...")
            try:
                DEPS_DIR.mkdir(exist_ok=True)
                subprocess.run(
                    ["git", "clone", LTX_VIDEO_REPO_URL, str(LTX_VIDEO_REPO_DIR)],
                    check=True, capture_output=True, text=True
                )
                logger.info("LTX-Video repository cloned successfully.")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to clone LTX-Video repository. Git stderr: {e.stderr}")
                raise RuntimeError("Could not clone the required LTX-Video dependency from GitHub.")
        else:
            logger.info("Found local LTX-Video repository.")

        if str(LTX_VIDEO_REPO_DIR.resolve()) not in sys.path:
            sys.path.insert(0, str(LTX_VIDEO_REPO_DIR.resolve()))
            logger.info(f"Added '{LTX_VIDEO_REPO_DIR.resolve()}' to sys.path.")
    
    def _lazy_load_ltx_modules(self):
        """Dynamically imports LTX-Video modules after ensuring the repo exists."""
        if self._ltx_modules_loaded:
            return

        global create_ltx_video_pipeline, calculate_padding, LTXVideoPipeline, ConditioningItem, LTXMultiScalePipeline
        global vae_encode, latent_to_pixel_coords, randn_tensor
        
        from managers.ltx_pipeline_utils import create_ltx_video_pipeline, calculate_padding
        from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline, ConditioningItem, LTXMultiScalePipeline
        from ltx_video.models.autoencoders.vae_encode import vae_encode, latent_to_pixel_coords
        from diffusers.utils.torch_utils import randn_tensor
        
        self._ltx_modules_loaded = True
        logger.info("LTX-Video modules have been dynamically loaded.")

    def _apply_ltx_pipeline_patches(self):
        """Applies runtime patches to the LTX pipeline for ADUC-SDR compatibility."""
        logger.info("LTX POOL MANAGER: Applying ADUC-SDR patches to LTX pipeline...")
        for worker in self.workers:
            worker.pipeline.prepare_conditioning = _aduc_prepare_conditioning_patch.__get__(worker.pipeline, LTXVideoPipeline)
        logger.info("LTX POOL MANAGER: All pipeline instances have been patched successfully.")

    def _get_next_worker(self):
        with self.lock:
            worker = self.workers[self.current_worker_index]
            self.current_worker_index = (self.current_worker_index + 1) % len(self.workers)
            return worker
    
    def _prepare_pipeline_params(self, worker: 'LtxWorker', **kwargs) -> dict:
        pipeline_params = {
            "height": kwargs['height'], "width": kwargs['width'], "num_frames": kwargs['video_total_frames'],
            "frame_rate": kwargs.get('video_fps', 24),
            "generator": torch.Generator(device=worker.device).manual_seed(int(time.time()) + kwargs.get('current_fragment_index', 0)),
            "is_video": True, "vae_per_channel_normalize": True,
            "prompt": kwargs.get('motion_prompt', ""), "negative_prompt": kwargs.get('negative_prompt', "blurry, distorted, static, bad quality"),
            "guidance_scale": kwargs.get('guidance_scale', 1.0), "stg_scale": kwargs.get('stg_scale', 0.0),
            "rescaling_scale": kwargs.get('rescaling_scale', 0.15), "num_inference_steps": kwargs.get('num_inference_steps', 20),
            "output_type": "latent"
        }
        if 'latents' in kwargs:
            pipeline_params["latents"] = kwargs['latents'].to(worker.device, dtype=worker.pipeline.transformer.dtype)
        if 'strength' in kwargs:
            pipeline_params["strength"] = kwargs['strength']
        if 'conditioning_items_data' in kwargs:
            final_conditioning_items = []
            for item in kwargs['conditioning_items_data']:
                item.latent_tensor = item.latent_tensor.to(worker.device)
                final_conditioning_items.append(item)
            pipeline_params["conditioning_items"] = final_conditioning_items
        if worker.is_distilled:
            logger.info(f"Worker {worker.device} is using a distilled model. Using fixed timesteps.")
            fixed_timesteps = worker.config.get("first_pass", {}).get("timesteps")
            pipeline_params["timesteps"] = fixed_timesteps
            if fixed_timesteps:
                pipeline_params["num_inference_steps"] = len(fixed_timesteps)
        return pipeline_params

    def generate_latent_fragment(self, **kwargs) -> (torch.Tensor, tuple):
        worker_to_use = self._get_next_worker()
        try:
            height, width = kwargs['height'], kwargs['width']
            padded_h, padded_w = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
            padding_vals = calculate_padding(height, width, padded_h, padded_w)
            kwargs['height'], kwargs['width'] = padded_h, padded_w
            pipeline_params = self._prepare_pipeline_params(worker_to_use, **kwargs)
            logger.info(f"Initiating GENERATION on {worker_to_use.device} with shape {padded_w}x{padded_h}")
            if isinstance(worker_to_use.pipeline, LTXMultiScalePipeline):
                result = worker_to_use.pipeline.video_pipeline(**pipeline_params).images
            else:
                result = worker_to_use.generate_video_fragment_internal(**pipeline_params)
            return result, padding_vals
        except Exception as e:
            logger.error(f"LTX POOL MANAGER: Error during generation on {worker_to_use.device}: {e}", exc_info=True)
            raise e
        finally:
            if worker_to_use and worker_to_use.device.type == 'cuda':
                with torch.cuda.device(worker_to_use.device):
                    gc.collect(); torch.cuda.empty_cache()

    def refine_latents(self, latents_to_refine: torch.Tensor, **kwargs) -> (torch.Tensor, tuple):
        # This function can be expanded later if needed.
        pass

class LtxWorker:
    """
    Represents a single instance of the LTX-Video pipeline on a specific device.
    """
    def __init__(self, device_id, ltx_config_file):
        self.cpu_device = torch.device('cpu')
        self.device = torch.device(device_id if torch.cuda.is_available() else 'cpu')
        logger.info(f"LTX Worker ({self.device}): Initializing with config '{ltx_config_file}'...")
        
        with open(ltx_config_file, "r") as file:
            self.config = yaml.safe_load(file)
        
        self.is_distilled = "distilled" in self.config.get("checkpoint_path", "")

        models_dir = LTX_VIDEO_REPO_DIR / "models_downloaded"
        
        logger.info(f"LTX Worker ({self.device}): Preparing to load model...")
        model_filename = self.config["checkpoint_path"]
        model_path = huggingface_hub.hf_hub_download(
            repo_id="Lightricks/LTX-Video", filename=model_filename,
            local_dir=str(models_dir), local_dir_use_symlinks=False
        )
        
        self.pipeline = create_ltx_video_pipeline(
            ckpt_path=model_path, 
            precision=self.config["precision"],
            text_encoder_model_name_or_path=self.config["text_encoder_model_name_or_path"],
            sampler=self.config["sampler"], 
            device='cpu'
        )
        logger.info(f"LTX Worker ({self.device}): Model ready on CPU. Is distilled model? {self.is_distilled}")
    
    def to_gpu(self):
        if self.device.type == 'cpu': return
        logger.info(f"LTX Worker: Moving pipeline to GPU {self.device}...")
        self.pipeline.to(self.device)
        if self.device.type == 'cuda' and can_optimize_fp8():
            logger.info(f"LTX Worker ({self.device}): FP8 supported GPU detected. Optimizing...")
            optimize_ltx_worker(self)
            logger.info(f"LTX Worker ({self.device}): Optimization complete.")
        elif self.device.type == 'cuda':
            logger.info(f"LTX Worker ({self.device}): FP8 optimization not supported or disabled.")

    def to_cpu(self):
        if self.device.type == 'cpu': return
        logger.info(f"LTX Worker: Unloading pipeline from GPU {self.device}...")
        self.pipeline.to('cpu')
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    def generate_video_fragment_internal(self, **kwargs):
        return self.pipeline(**kwargs).images


def _aduc_prepare_conditioning_patch(
    self: LTXVideoPipeline,
    conditioning_items: Optional[List[Union[ConditioningItem, "LatentConditioningItem"]]],
    init_latents: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    vae_per_channel_normalize: bool = False,
    generator=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if not conditioning_items:
        init_latents, init_latent_coords = self.patchifier.patchify(latents=init_latents)
        init_pixel_coords = latent_to_pixel_coords(init_latent_coords, self.vae, causal_fix=self.transformer.config.causal_temporal_positioning)
        return init_latents, init_pixel_coords, None, 0
    init_conditioning_mask = torch.zeros(init_latents[:, 0, :, :, :].shape, dtype=torch.float32, device=init_latents.device)
    extra_conditioning_latents, extra_conditioning_pixel_coords, extra_conditioning_mask = [], [], []
    extra_conditioning_num_latents = 0
    is_latent_mode = hasattr(conditioning_items[0], 'latent_tensor')
    if is_latent_mode:
        for item in conditioning_items:
            media_item_latents = item.latent_tensor.to(dtype=init_latents.dtype, device=init_latents.device)
            media_frame_number, strength = item.media_frame_number, item.conditioning_strength
            if media_frame_number == 0:
                f_l, h_l, w_l = media_item_latents.shape[-3:]
                init_latents[:, :, :f_l, :h_l, :w_l] = torch.lerp(init_latents[:, :, :f_l, :h_l, :w_l], media_item_latents, strength)
                init_conditioning_mask[:, :f_l, :h_l, :w_l] = strength
            else:
                noise = randn_tensor(media_item_latents.shape, generator=generator, device=media_item_latents.device, dtype=media_item_latents.dtype)
                media_item_latents = torch.lerp(noise, media_item_latents, strength)
                patched_latents, latent_coords = self.patchifier.patchify(latents=media_item_latents)
                pixel_coords = latent_to_pixel_coords(latent_coords, self.vae, causal_fix=self.transformer.config.causal_temporal_positioning)
                pixel_coords[:, 0] += media_frame_number
                extra_conditioning_num_latents += patched_latents.shape[1]
                new_mask = torch.full(patched_latents.shape[:2], strength, dtype=torch.float32, device=init_latents.device)
                extra_conditioning_latents.append(patched_latents)
                extra_conditioning_pixel_coords.append(pixel_coords)
                extra_conditioning_mask.append(new_mask)
    else:
        for item in conditioning_items:
            if not isinstance(item, ConditioningItem): continue
            item = self._resize_conditioning_item(item, height, width)
            media_item_latents = vae_encode(item.media_item.to(dtype=self.vae.dtype, device=self.vae.device), self.vae, vae_per_channel_normalize=vae_per_channel_normalize).to(dtype=init_latents.dtype)
            if item.media_frame_number == 0:
                media_item_latents, l_x, l_y = self._get_latent_spatial_position(media_item_latents, item, height, width, strip_latent_border=True)
                f_l, h_l, w_l = media_item_latents.shape[-3:]
                init_latents[:, :, :f_l, l_y:l_y+h_l, l_x:l_x+w_l] = torch.lerp(init_latents[:, :, :f_l, l_y:l_y+h_l, l_x:l_x+w_l], media_item_latents, item.conditioning_strength)
                init_conditioning_mask[:, :f_l, l_y:l_y+h_l, l_x:l_x+w_l] = item.conditioning_strength
            else:
                logger.warning("Pixel-based conditioning for non-zero frames is not fully implemented in this patch.")
    
    init_latents, init_latent_coords = self.patchifier.patchify(latents=init_latents)
    init_pixel_coords = latent_to_pixel_coords(init_latent_coords, self.vae, causal_fix=self.transformer.config.causal_temporal_positioning)
    init_conditioning_mask, _ = self.patchifier.patchify(latents=init_conditioning_mask.unsqueeze(1))
    init_conditioning_mask = init_conditioning_mask.squeeze(-1)
    if extra_conditioning_latents:
        init_latents = torch.cat([*extra_conditioning_latents, init_latents], dim=1)
        init_pixel_coords = torch.cat([*extra_conditioning_pixel_coords, init_pixel_coords], dim=2)
        init_conditioning_mask = torch.cat([*extra_conditioning_mask, init_conditioning_mask], dim=1)
        if self.transformer.use_tpu_flash_attention:
            init_latents = init_latents[:, :-extra_conditioning_num_latents]
            init_pixel_coords = init_pixel_coords[:, :, :-extra_conditioning_num_latents]
            init_conditioning_mask = init_conditioning_mask[:, :-extra_conditioning_num_latents]
    return init_latents, init_pixel_coords, init_conditioning_mask, extra_conditioning_num_latents


# --- Singleton Instantiation ---
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)
ltx_gpus_required = config['specialists']['ltx']['gpus_required']
ltx_device_ids = hardware_manager.allocate_gpus('LTX', ltx_gpus_required)
ltx_config_filename = config['specialists']['ltx']['config_file']
ltx_manager_singleton = LtxPoolManager(device_ids=ltx_device_ids, ltx_config_file_name=ltx_config_filename)
logger.info("Video Specialist (LTX) ready.")