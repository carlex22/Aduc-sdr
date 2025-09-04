# managers/vae_manager.py
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
# 
# This file defines the VaeManager specialist. Its purpose is to abstract all
# direct interactions with the Variational Autoencoder (VAE) model. It handles
# the model's state (CPU/GPU memory), provides clean interfaces for encoding and
# decoding, and ensures that the heavy VAE model only occupies VRAM when actively
# performing a task, freeing up resources for other specialists.
#
# Version 1.0.1


import torch
import logging
import gc
from typing import Generator

# Import the source of the VAE model and the low-level functions
from managers.ltx_manager import ltx_manager_singleton
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode

logger = logging.getLogger(__name__)

class VaeManager:
    """
    A specialist for managing the LTX VAE model. It provides high-level methods
    for encoding pixels to latents and decoding latents to pixels, while managing
    the model's presence on the GPU to conserve VRAM.
    """
    def __init__(self, vae_model):
        self.vae = vae_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cpu_device = torch.device('cpu')
        
        # Initialize the VAE on the CPU to keep VRAM free at startup
        self.vae.to(self.cpu_device)
        logger.info(f"VaeManager initialized. VAE model is on CPU.")

    def to_gpu(self):
        """Moves the VAE model to the active GPU."""
        if self.device == 'cpu': return
        logger.info("VaeManager: Moving VAE to GPU...")
        self.vae.to(self.device)

    def to_cpu(self):
        """Moves the VAE model to the CPU and clears VRAM cache."""
        if self.device == 'cpu': return
        logger.info("VaeManager: Unloading VAE from GPU...")
        self.vae.to(self.cpu_device)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.no_grad()
    def encode(self, pixel_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encodes a pixel-space tensor to the latent space.
        Manages moving the VAE to and from the GPU.
        """
        try:
            self.to_gpu()
            pixel_tensor = pixel_tensor.to(self.device, dtype=self.vae.dtype)
            latents = vae_encode(pixel_tensor, self.vae, vae_per_channel_normalize=True)
            return latents.to(self.cpu_device) # Return to CPU to free VRAM
        finally:
            self.to_cpu()

    @torch.no_grad()
    def decode(self, latent_tensor: torch.Tensor, decode_timestep: float = 0.05) -> torch.Tensor:
        """
        Decodes a latent-space tensor to pixels.
        Manages moving the VAE to and from the GPU.
        """
        try:
            self.to_gpu()
            latent_tensor = latent_tensor.to(self.device, dtype=self.vae.dtype)
            timestep_tensor = torch.tensor([decode_timestep] * latent_tensor.shape[0], device=self.device, dtype=latent_tensor.dtype)
            pixels = vae_decode(latent_tensor, self.vae, is_video=True, timestep=timestep_tensor, vae_per_channel_normalize=True)
            return pixels.to(self.cpu_device) # Return to CPU to free VRAM
        finally:
            self.to_cpu()

# --- Singleton Instance ---
# The VaeManager must use the exact same VAE instance as the LTX pipeline to ensure
# latent space compatibility. We source it directly from the already-initialized ltx_manager.
source_vae_model = ltx_manager_singleton.workers[0].pipeline.vae
vae_manager_singleton = VaeManager(source_vae_model)