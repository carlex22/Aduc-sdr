# deformes4D_engine.py (Lógica Final: Fragmentos Latentes e de Vídeo são independentes)
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos

import os
import time
import imageio
import numpy as np
import torch
import logging
from PIL import Image, ImageOps
from dataclasses import dataclass
import gradio as gr
import subprocess
import random

from audio_specialist import audio_specialist_singleton
from ltx_manager_helpers import ltx_manager_singleton
from flux_kontext_helpers import flux_kontext_singleton
from gemini_helpers import gemini_singleton
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode

logger = logging.getLogger(__name__)

@dataclass
class LatentConditioningItem:
    latent_tensor: torch.Tensor
    media_frame_number: int
    conditioning_strength: float

class Deformes4DEngine:
    def __init__(self, ltx_manager, workspace_dir="deformes_workspace"):
        self.ltx_manager = ltx_manager
        self.workspace_dir = workspace_dir
        self._vae = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Especialista Deformes4D (SDR Executor) inicializado.")

    @property
    def vae(self):
        if self._vae is None:
            self._vae = self.ltx_manager.workers[0].pipeline.vae
        self._vae.to(self.device); self._vae.eval()
        return self._vae

    def save_latent_tensor(self, tensor: torch.Tensor, path: str):
        torch.save(tensor.cpu(), path)
        logger.info(f"Tensor latente salvo em: {path}")

    def load_latent_tensor(self, path: str) -> torch.Tensor:
        tensor = torch.load(path, map_location=self.device)
        logger.info(f"Tensor latente carregado de: {path} para o dispositivo {self.device}")
        return tensor

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
        
    @torch.no_grad()
    def get_last_frame_from_latent(self, latent_path: str) -> Image.Image:
        latent_tensor = self.load_latent_tensor(latent_path)
        last_frame_latent = latent_tensor[:, :, -1:, :, :]
        pixel_tensor = self.latents_to_pixels(last_frame_latent)
        pixel_array = pixel_tensor.squeeze(0).squeeze(1).permute(1, 2, 0) 
        pixel_array = ((pixel_array.clamp(-1, 1) + 1) / 2.0 * 255).byte().cpu().numpy()
        return Image.fromarray(pixel_array)

    def process_image_for_story(self, image_path: str, size: int, filename: str = None) -> str:
        img = Image.open(image_path).convert("RGB")
        img_square = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        if filename: 
            processed_path = os.path.join(self.workspace_dir, filename)
        else: 
            processed_path = os.path.join(self.workspace_dir, f"ref_processed_{int(time.time()*1000)}.png")
        img_square.save(processed_path)
        return processed_path

    def generate_keyframe(self, prompt: str, reference_images: list[Image.Image], output_filename: str, width: int, height: int, callback: callable = None) -> str:
        new_img = flux_kontext_singleton.generate_image(
            reference_images=reference_images, prompt=prompt, width=width, height=height, 
            seed=int(time.time()), callback=callback
        )
        final_path = os.path.join(self.workspace_dir, output_filename)
        new_img.save(final_path)
        return final_path
        
    def _generate_fragment_and_add_audio(self, latent_tensor, n_eco_frames, scene_prompt, base_name):
        n_eco_latent = n_eco_frames // 8
        
        if n_eco_latent > 0 and latent_tensor.shape[2] > n_eco_latent:
            logger.info(f"Aparando {n_eco_latent*8} frames de eco do início do fragmento de vídeo.")
            latent_tensor_trimmed = latent_tensor[:, :, n_eco_latent:, :, :]
        else:
            latent_tensor_trimmed = latent_tensor

        silent_video_path = os.path.join(self.workspace_dir, f"{base_name}_silent.mp4")
        pixel_tensor = self.latents_to_pixels(latent_tensor_trimmed)
        self.save_video_from_tensor(pixel_tensor, silent_video_path, fps=24)
        
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", silent_video_path],
            capture_output=True, text=True, check=True)
        frag_duration = float(result.stdout.strip())
        
        video_with_audio_path = audio_specialist_singleton.generate_audio_for_video(
            video_path=silent_video_path, prompt=scene_prompt,
            negative_prompt="music, speech", duration_seconds=frag_duration)
        
        os.remove(silent_video_path)
        return video_with_audio_path

    def create_initial_fragment(self, start_image_path: str, end_image_path: str, duration_seconds: float, ltx_params: dict, target_resolution: tuple, scene_prompt: str) -> tuple[str, str]:
        pil_start = Image.open(start_image_path).convert("RGB")
        pil_end = Image.open(end_image_path).convert("RGB")
        if pil_start.size != target_resolution: pil_start = ImageOps.fit(pil_start, target_resolution, Image.Resampling.LANCZOS)
        if pil_end.size != target_resolution: pil_end = ImageOps.fit(pil_end, target_resolution, Image.Resampling.LANCZOS)
        start_latent, end_latent = self.pil_to_latent(pil_start), self.pil_to_latent(pil_end)
        a_frames = int(duration_seconds * 24); a_frames = (a_frames // 8) * 8 + 1
        final_ltx_params = {**ltx_params, 'width': target_resolution[0], 'height': target_resolution[1], 'video_total_frames': a_frames, 'video_fps': 24, 'current_fragment_index': int(time.time()),
            'conditioning_items_data': [LatentConditioningItem(start_latent, 0, 1.0), LatentConditioningItem(end_latent, a_frames - 1, ltx_params.get('p_dest', 1.0))]
        }
        new_latent_fragment, _ = self.ltx_manager.generate_latent_fragment(**final_ltx_params)
        
        base_name = f"fragment_0_{int(time.time())}"
        new_latent_path = os.path.join(self.workspace_dir, f"{base_name}.pt")
        self.save_latent_tensor(new_latent_fragment, new_latent_path)

        video_with_audio = self._generate_fragment_and_add_audio(new_latent_fragment, 0, scene_prompt, base_name)
        
        return new_latent_path, video_with_audio

    def create_next_fragment(self, previous_latent_path: str, end_image_path: str, n_corte: int, n_eco: int, a_frames: int, p_caminho: float, ltx_params: dict, target_resolution: tuple, scene_prompt: str) -> tuple[str, str]:
        previous_latent = self.load_latent_tensor(previous_latent_path)
        end_image = Image.open(end_image_path).convert("RGB")
        if end_image.size != target_resolution: end_image = ImageOps.fit(end_image, target_resolution, Image.Resampling.LANCZOS)
        
        destination_latent = self.pil_to_latent(end_image)
        n_corte_latent, n_eco_latent = n_corte // 8, n_eco // 8
        
        path_anchor_latent = previous_latent[:, :, -1:].clone()
        if previous_latent.shape[2] <= n_corte_latent: raise ValueError(f"Latente anterior ({previous_latent.shape[2]}f) é muito curto para o corte de {n_corte_latent}f.")
        LatentCortado = previous_latent[:, :, :-n_corte_latent]
        
        if LatentCortado.shape[2] < n_eco_latent: raise ValueError(f"Latente cortado ({LatentCortado.shape[2]}f) é muito curto para Eco de {n_eco_latent}f.")
        kinetic_echo_latent = LatentCortado[:, :, -n_eco_latent:]
        
        final_ltx_params = {**ltx_params, 'width': target_resolution[0], 'height': target_resolution[1], 'video_total_frames': a_frames, 'video_fps': 24, 'current_fragment_index': int(time.time()),
            'conditioning_items_data': [
                LatentConditioningItem(kinetic_echo_latent, 0, 1.0),
                LatentConditioningItem(path_anchor_latent, n_corte, p_caminho),
                LatentConditioningItem(destination_latent, a_frames - 1, ltx_params.get('p_dest', 1.0)),
            ]
        }
        
        new_latent_fragment, _ = self.ltx_manager.generate_latent_fragment(**final_ltx_params)
        
        base_name = f"fragment_{int(time.time())}"
        new_latent_path = os.path.join(self.workspace_dir, f"{base_name}.pt")
        self.save_latent_tensor(new_latent_fragment, new_latent_path)

        video_with_audio = self._generate_fragment_and_add_audio(new_latent_fragment, n_eco, scene_prompt, base_name)

        return new_latent_path, video_with_audio

    def create_ffmpeg_bridge(self, start_image_path: str, end_image_path: str, duration: float, fps: int, target_resolution: tuple) -> str:
        output_path = os.path.join(self.workspace_dir, f"bridge_{int(time.time())}.mp4")
        width, height = target_resolution
        fade_effects = ["fade", "wipeleft", "wiperight", "wipeup", "wipedown", "dissolve", "fadeblack", "fadewhite"]
        selected_effect = random.choice(fade_effects)
        transition_duration = duration
        cmd = (
            f"ffmpeg -y -v error -loop 1 -t {transition_duration} -i \"{start_image_path}\" -loop 1 -t {transition_duration} -i \"{end_image_path}\" "
            f"-filter_complex \"[0:v]scale={width}:{height},setsar=1[v0];[1:v]scale={width}:{height},setsar=1[v1];"
            f"[v0][v1]xfade=transition={selected_effect}:duration={transition_duration}:offset=0[out]\" "
            f"-map \"[out]\" -c:v libx264 -r {fps} -pix_fmt yuv420p \"{output_path}\""
        )
        logger.info(f"Criando ponte FFmpeg com efeito aleatório: '{selected_effect}'")
        subprocess.run(cmd, shell=True, check=True, text=True)
        return output_path

    def video_to_latent_path(self, video_path: str, latent_filename: str) -> str:
        with imageio.get_reader(video_path) as reader:
            frames = np.array([frame for frame in reader])
        video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0
        video_tensor = (video_tensor * 2.0) - 1.0
        latent_tensor = self.pixels_to_latents(video_tensor)
        latent_path = os.path.join(self.workspace_dir, latent_filename)
        self.save_latent_tensor(latent_tensor, latent_path)
        return latent_path
        
    def concatenate_videos_ffmpeg(self, video_paths: list[str], output_path: str) -> str:
        if not video_paths:
            raise gr.Error("Nenhum fragmento de vídeo para montar.")
        
        list_file_path = os.path.join(self.workspace_dir, "concat_list.txt")
        with open(list_file_path, "w") as f:
            for path in video_paths:
                f.write(f"file '{os.path.abspath(path).replace(os.sep, '/')}'\n")
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-c', 'copy', output_path]
        
        logger.info(f"Executando concatenação FFmpeg: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no FFmpeg: {e.stderr}")
            raise gr.Error(f"Falha na montagem final do vídeo: {e.stderr}")
        finally:
            if os.path.exists(list_file_path):
                os.remove(list_file_path)
        return output_path
    
    def concatenate_latents_only(self, latent_paths: list, n_eco: int) -> str:
        if not latent_paths: return None
        
        n_eco_latent = n_eco // 8
        all_latents = [self.load_latent_tensor(p) for p in latent_paths]
        
        trimmed_latents = []
        for i, lat in enumerate(all_latents):
            # O primeiro fragmento não tem eco no início para cortar
            if i == 0:
                trimmed_latents.append(lat)
            else:
                if lat.shape[2] > n_eco_latent:
                    trimmed_latents.append(lat[:, :, n_eco_latent:, :, :])
                else:
                    # Se o fragmento for mais curto que o eco, não adicionamos nada.
                    logger.warning(f"Fragmento latente {i} é mais curto que o eco e será pulado na concatenação.")

        if not trimmed_latents:
            logger.warning("Nenhum latente restou após o corte do eco. Não foi possível criar o latente final.")
            return None
            
        final_latent_tensor = torch.cat(trimmed_latents, dim=2)
        
        final_latent_path = os.path.join(self.workspace_dir, f"final_movie_latents_{int(time.time())}.pt")
        self.save_latent_tensor(final_latent_tensor, final_latent_path)
        logger.info(f"Cadeia final de latentes aparados e concatenados salva em: {final_latent_path}")
        return final_latent_path
        