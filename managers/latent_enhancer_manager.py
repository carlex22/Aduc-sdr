# latent_enhancer_specialist.py
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
#
# PENDING PATENT NOTICE: Please see NOTICE.md.
#
# Version 1.0.1

import torch
import logging
import time
from diffusers import LTXLatentUpsamplePipeline
from managers.ltx_manager import ltx_manager_singleton

logger = logging.getLogger(__name__)

class LatentEnhancerSpecialist:
    """
    Especialista responsável por melhorar a qualidade de tensores latentes,
    incluindo upscaling espacial e refinamento por denoise.
    """
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe_upsample = None
        self.base_vae = None # VAE para o upscaler

    def _lazy_init_upscaler(self):
        """Inicializa a pipeline de upscaling apenas quando for usada."""
        if self.pipe_upsample is not None:
            return
        try:
            from diffusers.models.autoencoders import AutoencoderKLLTXVideo
            self.base_vae = AutoencoderKLLTXVideo.from_pretrained(
                "linoyts/LTX-Video-spatial-upscaler-0.9.8",
                subfolder="vae",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)

            self.pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
                "linoyts/LTX-Video-spatial-upscaler-0.9.8",
                vae=self.base_vae,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            logger.info("[Enhancer] Pipeline de Upscale carregada com sucesso.")
        except Exception as e:
            logger.error(f"[Enhancer] Falha ao carregar pipeline de Upscale: {e}")
            self.pipe_upsample = None

    @torch.no_grad()
    def upscale(self, latents: torch.Tensor) -> torch.Tensor:
        """Aplica o upscaling 2x nos tensores latentes fornecidos."""
        self._lazy_init_upscaler()
        if self.pipe_upsample is None:
            logger.warning("[Enhancer] Pipeline de Upscale indisponível. Retornando latentes originais.")
            return latents
        try:
            logger.info(f"[Enhancer] Recebido shape {latents.shape} para Upscale.")
            result = self.pipe_upsample(latents=latents, output_type="latent")
            output_tensor = result.frames
            logger.info(f"[Enhancer] Upscale concluído. Novo shape: {output_tensor.shape}")
            return output_tensor
        except Exception as e:
            logger.error(f"[Enhancer] Erro durante upscale: {e}", exc_info=True)
            return latents

    @torch.no_grad()
    def refine(self, latents: torch.Tensor, fps: int = 24, **kwargs) -> torch.Tensor:
        """
        Invoca o LTX Pool Manager para refinar um tensor latente existente.
        """
        logger.info(f"[Enhancer] Refinando tensor latente com shape {latents.shape}.")
        
        main_pipeline_vae = ltx_manager_singleton.workers[0].pipeline.vae
        video_scale_factor = getattr(main_pipeline_vae.config, 'temporal_scale_factor', 8)
        
        _, _, num_latent_frames, _, _ = latents.shape
        
        # --- [CORREÇÃO FINAL E CRÍTICA] ---
        # A pipeline de refinamento (vid2vid) espera o número de frames de pixels que CORRESPONDE
        # ao latente existente, SEM a lógica do +1 que ela aplicará internamente.
        pixel_frames = (num_latent_frames - 1) * video_scale_factor

        final_ltx_params = {
            "video_total_frames": pixel_frames,
            "video_fps": fps,
            "current_fragment_index": int(time.time()),
            **kwargs
        }
        
        refined_latents_tensor, _ = ltx_manager_singleton.refine_latents(latents, **final_ltx_params)
        
        if refined_latents_tensor is None:
            logger.warning("[Enhancer] O refinamento falhou. Retornando tensor original não refinado.")
            return latents
        
        logger.info(f"[Enhancer] Retornando tensor latente refinado com shape: {refined_latents_tensor.shape}")
        return refined_latents_tensor

# --- Singleton Global ---
latent_enhancer_specialist_singleton = LatentEnhancerSpecialist()