# managers/seedvr_manager.py
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
# Version: 2.3.0
#
# This file implements the SeedVrManager, which uses the SeedVR model for
# video super-resolution. It is self-contained, automatically cloning its own
# dependencies from the official SeedVR repository.

import torch
import os
import gc
import logging
import sys
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from torch.hub import download_url_to_file
import gradio as gr
import mediapy
from einops import rearrange

# Internalized utility for color correction, ensuring stability.
from tools.tensor_utils import wavelet_reconstruction

logger = logging.getLogger(__name__)

# --- Dependency Management ---
DEPS_DIR = Path("./deps")
SEEDVR_REPO_DIR = DEPS_DIR / "SeedVR"
SEEDVR_REPO_URL = "https://github.com/ByteDance-Seed/SeedVR.git"

def setup_seedvr_dependencies():
    """
    Ensures the SeedVR repository is cloned and available in the sys.path.
    This function is run once when the module is first imported.
    """
    if not SEEDVR_REPO_DIR.exists():
        logger.info(f"SeedVR repository not found at '{SEEDVR_REPO_DIR}'. Cloning from GitHub...")
        try:
            DEPS_DIR.mkdir(exist_ok=True)
            # Use --depth 1 for a shallow clone to save space and time
            subprocess.run(
                ["git", "clone", "--depth", "1", SEEDVR_REPO_URL, str(SEEDVR_REPO_DIR)],
                check=True, capture_output=True, text=True
            )
            logger.info("SeedVR repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone SeedVR repository. Git stderr: {e.stderr}")
            raise RuntimeError("Could not clone the required SeedVR dependency from GitHub.")
    else:
        logger.info("Found local SeedVR repository.")
    
    # Add the cloned repo to Python's path to allow direct imports
    if str(SEEDVR_REPO_DIR.resolve()) not in sys.path:
        sys.path.insert(0, str(SEEDVR_REPO_DIR.resolve()))
        logger.info(f"Added '{SEEDVR_REPO_DIR.resolve()}' to sys.path.")

# --- Execute dependency setup immediately upon module import ---
setup_seedvr_dependencies()

# --- Now that the path is set, we can safely import from the cloned repo ---
from projects.video_diffusion_sr.infer import VideoDiffusionInfer
from common.config import load_config
from common.seed import set_seed
from data.image.transforms.divisible_crop import DivisibleCrop
from data.image.transforms.na_resize import NaResize
from data.video.transforms.rearrange import Rearrange
from torchvision.transforms import Compose, Lambda, Normalize
from torchvision.io.video import read_video
from omegaconf import OmegaConf


def _load_file_from_url(url, model_dir='./', file_name=None):
    """Helper function to download files from a URL to a local directory."""
    os.makedirs(model_dir, exist_ok=True)
    filename = file_name or os.path.basename(urlparse(url).path)
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        logger.info(f'Downloading: "{url}" to {cached_file}')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=True)
    return cached_file

class SeedVrManager:
    """
    Manages the SeedVR model for HD Mastering tasks.
    """
    def __init__(self, workspace_dir="deformes_workspace"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.runner = None
        self.workspace_dir = workspace_dir
        self.is_initialized = False
        logger.info("SeedVrManager initialized. Model will be loaded on demand.")

    def _download_models(self):
        """Downloads the necessary checkpoints for SeedVR2."""
        logger.info("Verifying and downloading SeedVR2 models...")
        ckpt_dir = SEEDVR_REPO_DIR / 'ckpts'
        ckpt_dir.mkdir(exist_ok=True)

        pretrain_model_urls = {
            'vae': 'https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/ema_vae.pth',
            'dit_3b': 'https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/seedvr2_ema_3b.pth',
            'dit_7b': 'https://huggingface.co/ByteDance-Seed/SeedVR2-7B/resolve/main/seedvr2_ema_7b.pth',
            'pos_emb': 'https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/pos_emb.pt',
            'neg_emb': 'https://huggingface.co/ByteDance-Seed/SeedVR2-3B/resolve/main/neg_emb.pt'
        }
        
        for key, url in pretrain_model_urls.items():
            _load_file_from_url(url=url, model_dir=str(ckpt_dir))
            
        logger.info("SeedVR2 models downloaded successfully.")

    def _initialize_runner(self, model_version: str):
        """Loads and configures the SeedVR model on demand based on the selected version."""
        if self.runner is not None: return

        self._download_models()

        logger.info(f"Initializing SeedVR2 {model_version} runner...")
        if model_version == '3B':
            config_path = SEEDVR_REPO_DIR / 'configs_3b' / 'main.yaml'
            checkpoint_path = SEEDVR_REPO_DIR / 'ckpts' / 'seedvr2_ema_3b.pth'
        elif model_version == '7B':
            config_path = SEEDVR_REPO_DIR / 'configs_7b' / 'main.yaml'
            checkpoint_path = SEEDVR_REPO_DIR / 'ckpts' / 'seedvr2_ema_7b.pth'
        else:
            raise ValueError(f"Unsupported SeedVR model version: {model_version}")

        config = load_config(str(config_path))
        
        self.runner = VideoDiffusionInfer(config)
        OmegaConf.set_readonly(self.runner.config, False)
        
        self.runner.configure_dit_model(device=self.device, checkpoint=str(checkpoint_path))
        self.runner.configure_vae_model()
        
        if hasattr(self.runner.vae, "set_memory_limit"):
            self.runner.vae.set_memory_limit(**self.runner.config.vae.memory_limit)
        
        self.is_initialized = True
        logger.info(f"Runner for SeedVR2 {model_version} initialized and ready.")

    def _unload_runner(self):
        """Removes the runner from VRAM to free resources."""
        if self.runner is not None:
            del self.runner; self.runner = None
            gc.collect(); torch.cuda.empty_cache()
            self.is_initialized = False
            logger.info("SeedVR2 runner unloaded from VRAM.")

    def process_video(self, input_video_path: str, output_video_path: str, prompt: str,
                      model_version: str = '3B', steps: int = 50, seed: int = 666, 
                      progress: gr.Progress = None) -> str:
        """Applies HD enhancement to a video using the SeedVR logic."""
        try:
            self._initialize_runner(model_version)
            set_seed(seed, same_across_ranks=True)

            self.runner.config.diffusion.timesteps.sampling.steps = steps
            self.runner.configure_diffusion()

            video_tensor = read_video(input_video_path, output_format="TCHW")[0] / 255.0
            res_h, res_w = video_tensor.shape[-2:]
            
            video_transform = Compose([
                NaResize(resolution=(res_h * res_w) ** 0.5, mode="area", downsample_only=False),
                Lambda(lambda x: torch.clamp(x, 0.0, 1.0)),
                DivisibleCrop((16, 16)),
                Normalize(0.5, 0.5),
                Rearrange("t c h w -> c t h w"),
            ])
            
            cond_latents = [video_transform(video_tensor.to(self.device))]
            input_videos = cond_latents

            self.runner.dit.to("cpu")
            self.runner.vae.to(self.device)
            cond_latents = self.runner.vae_encode(cond_latents)
            self.runner.vae.to("cpu"); gc.collect(); torch.cuda.empty_cache()
            self.runner.dit.to(self.device)

            pos_emb_path = SEEDVR_REPO_DIR / 'ckpts' / 'pos_emb.pt'
            neg_emb_path = SEEDVR_REPO_DIR / 'ckpts' / 'neg_emb.pt'
            text_pos_embeds = torch.load(pos_emb_path).to(self.device)
            text_neg_embeds = torch.load(neg_emb_path).to(self.device)
            text_embeds_dict = {"texts_pos": [text_pos_embeds], "texts_neg": [text_neg_embeds]}

            noises = [torch.randn_like(latent) for latent in cond_latents]
            conditions = [self.runner.get_condition(noise, latent_blur=latent, task="sr") for noise, latent in zip(noises, cond_latents)]

            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16, enabled=True):
                video_tensors = self.runner.inference(noises=noises, conditions=conditions, dit_offload=True, **text_embeds_dict)
            
            self.runner.dit.to("cpu"); gc.collect(); torch.cuda.empty_cache()

            self.runner.vae.to(self.device)
            samples = self.runner.vae_decode(video_tensors)
            
            final_sample = samples[0]
            input_video_sample = input_videos[0]

            if final_sample.shape[1] < input_video_sample.shape[1]:
                input_video_sample = input_video_sample[:, :final_sample.shape[1]]

            final_sample = wavelet_reconstruction(
                rearrange(final_sample, "c t h w -> t c h w"), 
                rearrange(input_video_sample, "c t h w -> t c h w")
            )
            
            final_sample = rearrange(final_sample, "t c h w -> t h w c")
            final_sample = final_sample.clip(-1, 1).mul_(0.5).add_(0.5).mul_(255).round()
            final_sample_np = final_sample.to(torch.uint8).cpu().numpy()

            mediapy.write_video(output_video_path, final_sample_np, fps=24)
            logger.info(f"HD Mastered video saved to: {output_video_path}")
            return output_video_path
        finally:
            self._unload_runner()

# --- Singleton Instance ---
seedvr_manager_singleton = SeedVrManager()