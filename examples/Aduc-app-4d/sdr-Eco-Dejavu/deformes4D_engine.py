# deformes4D_engine.py (Versão com Concatenação Corrigida)
# Copyright (C) 2025 Carlos Rodrigues dos Santos

import os
import shutil
import time
import imageio
import numpy as np
import torch
import logging
from PIL import Image
from dataclasses import dataclass
import gradio as gr

from ltx_manager_helpers import ltx_manager_singleton
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode

logger = logging.getLogger(__name__)

@dataclass
class LatentConditioningItem:
    """Estrutura de dados para passar tensores latentes diretamente como condicionamento."""
    latent_tensor: torch.Tensor
    media_frame_number: int
    conditioning_strength: float

class Deformes4DEngine:
    """
    Especialista ADUC para manipulação de vídeo no espaço latente.
    Implementa a lógica da "Cadeia Causal com Déjà-Vu" para a finalização de cenas.
    """
    def __init__(self, ltx_manager, workspace_dir="deformes_workspace"):
        self.ltx_manager = ltx_manager
        self.workspace_dir = workspace_dir
        self._vae = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Especialista Deformes4D inicializado.")

    @property
    def vae(self):
        if self._vae is None:
            self._vae = self.ltx_manager.workers[0].pipeline.vae
        self._vae.to(self.device); self._vae.eval()
        return self._vae

    @torch.no_grad()
    def pixels_to_latents(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.to(self.device, dtype=self.vae.dtype)
        return vae_encode(tensor, self.vae, vae_per_channel_normalize=True)

    @torch.no_grad()
    def latents_to_pixels(self, latent_tensor: torch.Tensor, decode_timestep: float = 0.05) -> torch.Tensor:
        latent_tensor = latent_tensor.to(self.device, dtype=self.vae.dtype)
        timestep_tensor = torch.tensor([decode_timestep] * latent_tensor.shape[0], device=self.device, dtype=latent_tensor.dtype)
        return vae_decode(latent_tensor, self.vae, is_video=True, timestep=timestep_tensor, vae_per_channel_normalize=True)

    def save_video_from_tensor(self, video_tensor: torch.Tensor, path: str, fps: int = 24):
        if video_tensor is None or video_tensor.ndim != 5 or video_tensor.shape[2] == 0: return
        video_tensor = video_tensor.squeeze(0).permute(1, 2, 3, 0)
        video_tensor = (video_tensor.clamp(-1, 1) + 1) / 2.0
        video_np = (video_tensor.detach().cpu().float().numpy() * 255).astype(np.uint8)
        with imageio.get_writer(path, fps=fps, codec='libx264', quality=8) as writer:
            for frame in video_np: writer.append_data(frame)
        logger.info(f"Vídeo salvo em: {path}")

    def pil_to_latent(self, pil_image: Image.Image) -> torch.Tensor:
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        tensor = (tensor * 2.0) - 1.0
        return self.pixels_to_latents(tensor)

    def video_to_latent(self, video_path: str) -> (torch.Tensor, tuple):
        with imageio.get_reader(video_path) as reader:
            frames = np.stack([frame for frame in reader])
        height, width = frames.shape[1], frames.shape[2]
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        video_tensor = (video_tensor * 2.0) - 1.0
        return self.pixels_to_latents(video_tensor), (height, width)

    def preprocess_and_quantize_video(self, video_path: str, fps: int = 24) -> str:
        """Carrega, ajusta FPS e quantiza o número de frames para 8n+1."""
        logger.info(f"Pré-processando o vídeo de entrada: {video_path}")
        try:
            reader = imageio.get_reader(video_path)
            original_fps = reader.get_meta_data().get('fps', fps)
            frames = [frame for frame in reader]
            reader.close()
            num_frames_raw = len(frames)
            logger.info(f"Vídeo original tem {num_frames_raw} frames a {original_fps} FPS.")
            
            num_chunks = max(1, round((num_frames_raw - 1) / 8))
            quantized_frames = num_chunks * 8 + 1
            
            if quantized_frames != num_frames_raw:
                logger.warning(f"Número de frames ajustado de {num_frames_raw} para {quantized_frames}.")
                frames = frames[:quantized_frames]

            temp_video_path = os.path.join(self.workspace_dir, "input_video_processed.mp4")
            writer = imageio.get_writer(temp_video_path, fps=fps, codec='libx264', quality=8)
            for frame in frames: writer.append_data(frame)
            writer.close()
            
            logger.info(f"Vídeo pré-processado salvo em: {temp_video_path}")
            return temp_video_path
        except Exception as e:
            logger.error(f"Falha ao pré-processar o vídeo: {e}", exc_info=True)
            raise IOError(f"Não foi possível processar o arquivo de vídeo: {video_path}.")

    def finalize_scene(self, video_path: str, image_path: str, n_corte: int, n_eco: int, a_frames: int, p_caminho: float, ltx_params: dict, progress) -> str:
        try:
            progress(0.1, desc="Pré-processando e convertendo mídias...")
            processed_video_path = self.preprocess_and_quantize_video(video_path)
            pil_image = Image.open(image_path).convert("RGB")
            
            VideoLatent, (vid_h, vid_w) = self.video_to_latent(processed_video_path)
            if pil_image.size != (vid_w, vid_h):
                pil_image = pil_image.resize((vid_w, vid_h), Image.Resampling.LANCZOS)
            
            destination_latent = self.pil_to_latent(pil_image)
            
            progress(0.2, desc="Extraindo guias latentes...")
            vae_t_scale = 8
            n_corte_latent = n_corte // vae_t_scale
            n_eco_latent = n_eco // vae_t_scale
            
            path_anchor_latent = VideoLatent[:, :, -1:].clone()
            
            VideoLatentCortado = VideoLatent[:, :, :-n_corte_latent]
            if VideoLatentCortado.shape[2] < n_eco_latent:
                raise ValueError(f"Vídeo de entrada (após corte) é muito curto para o eco.")

            kinetic_echo_latent = VideoLatentCortado[:, :, -n_eco_latent:]
            logger.info(f"Guias: Eco Cinético={kinetic_echo_latent.shape}, Âncora de Caminho={path_anchor_latent.shape}, Destino={destination_latent.shape}")

            ltx_params.update({
                'width': vid_w, 'height': vid_h,
                'video_total_frames': a_frames, 'current_fragment_index': 1
            })
            
            ltx_params['conditioning_items_data'] = [
                LatentConditioningItem(latent_tensor=kinetic_echo_latent, media_frame_number=0, conditioning_strength=1.0),
                LatentConditioningItem(latent_tensor=path_anchor_latent, media_frame_number=n_corte, conditioning_strength=p_caminho),
                LatentConditioningItem(latent_tensor=destination_latent, media_frame_number=a_frames - 1, conditioning_strength=1.0),
            ]
            
            progress(0.4, desc="Gerando transição (video_beta)...")
            VideoBetaLatente, _ = self.ltx_manager.generate_latent_fragment(**ltx_params, progress=progress)

            progress(0.9, desc="Costurando vídeo final...")
            
            # Remove os frames do eco do vídeo original para evitar duplicação.
            VideoLatentFinal = VideoLatentCortado[:, :, :-n_eco_latent]
            logger.info(f"Aparando {n_eco_latent} frames de eco do vídeo original antes de concatenar.")
            
            final_latents = torch.cat([VideoLatentFinal, VideoBetaLatente.to(self.device)], dim=2)
            
            decode_timestep_val = ltx_params.get('decode_timestep', 0.05)
            final_pixels = self.latents_to_pixels(final_latents, decode_timestep=decode_timestep_val)
            final_video_path = os.path.join(self.workspace_dir, f"final_dejavu_{int(time.time())}.mp4")
            self.save_video_from_tensor(final_pixels, final_video_path, fps=ltx_params.get('video_fps', 24))
            
            return final_video_path
            
        except Exception as e:
            logger.error(f"Falha crítica em 'finalize_scene': {e}", exc_info=True)
            raise gr.Error(f"A finalização falhou: {e}")