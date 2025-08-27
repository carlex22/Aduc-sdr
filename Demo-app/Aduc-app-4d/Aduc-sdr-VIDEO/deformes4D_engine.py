# deformes4D_engine.py
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
#
# MODIFICATIONS FOR ADUC-SDR:
# Copyright (C) 2025 Carlos Rodrigues dos Santos. All rights reserved.
#
# This file is part of the ADUC-SDR project. It contains the core logic for
# video fragment generation, latent manipulation, and dynamic editing, 
# governed by the ADUC orchestrator.
# This component is licensed under the GNU Affero General Public License v3.0.

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
import gc

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

    def _preprocess_image_for_latent_conversion(self, image: Image.Image, target_resolution: tuple) -> Image.Image:
        if image.size != target_resolution:
            logger.info(f"  - AÇÃO: Redimensionando imagem de {image.size} para {target_resolution} antes da conversão para latente.")
            return ImageOps.fit(image, target_resolution, Image.Resampling.LANCZOS)
        return image

    def pil_to_latent(self, pil_image: Image.Image) -> torch.Tensor:
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        tensor = (tensor * 2.0) - 1.0
        return self.pixels_to_latents(tensor)
        
    def _generate_video_and_audio_from_latents(self, latent_tensor, audio_prompt, base_name):
        silent_video_path = os.path.join(self.workspace_dir, f"{base_name}_silent.mp4")
        pixel_tensor = self.latents_to_pixels(latent_tensor)
        self.save_video_from_tensor(pixel_tensor, silent_video_path, fps=24)
        del pixel_tensor; gc.collect()
        
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", silent_video_path],
                capture_output=True, text=True, check=True)
            frag_duration = float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
             logger.warning(f"ffprobe falhou em {os.path.basename(silent_video_path)}. Calculando duração manualmente.")
             num_pixel_frames = latent_tensor.shape[2] * 8
             frag_duration = num_pixel_frames / 24.0

        video_with_audio_path = audio_specialist_singleton.generate_audio_for_video(
            video_path=silent_video_path, prompt=audio_prompt,
            duration_seconds=frag_duration)
        
        if os.path.exists(silent_video_path):
             os.remove(silent_video_path)
        return video_with_audio_path

    def _generate_latent_tensor_internal(self, conditioning_items, ltx_params, target_resolution, total_frames_to_generate):
        final_ltx_params = {
            **ltx_params, 
            'width': target_resolution[0], 'height': target_resolution[1], 
            'video_total_frames': total_frames_to_generate, 'video_fps': 24, 
            'current_fragment_index': int(time.time()),
            'conditioning_items_data': conditioning_items
        }
        new_full_latents, _ = self.ltx_manager.generate_latent_fragment(**final_ltx_params)
        return new_full_latents

    def concatenate_videos_ffmpeg(self, video_paths: list[str], output_path: str) -> str:
        if not video_paths:
            raise gr.Error("Nenhum fragmento de vídeo para montar.")
        list_file_path = os.path.join(self.workspace_dir, "concat_list.txt")
        with open(list_file_path, 'w', encoding='utf-8') as f:
            for path in video_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")
        cmd_list = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-c', 'copy', output_path]
        logger.info("Executando concatenação FFmpeg...")
        try:
            subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no FFmpeg: {e.stderr}")
            raise gr.Error(f"Falha na montagem final do vídeo. Detalhes: {e.stderr}")
        return output_path
    
    def generate_full_movie(self, 
                            keyframes: list, 
                            global_prompt: str, 
                            storyboard: list, 
                            seconds_per_fragment: float, 
                            overlap_percent: int, 
                            echo_frames: int,
                            handler_strength: float, 
                            destination_convergence_strength: float,
                            base_ltx_params: dict,
                            video_resolution: int, 
                            use_continuity_director: bool, 
                            progress: gr.Progress = gr.Progress()):
        
        keyframe_paths = [item[0] if isinstance(item, tuple) else item for item in keyframes]
        video_clips_paths, story_history, audio_history = [], "", "This is the beginning of the film."
        target_resolution_tuple = (video_resolution, video_resolution) 
        n_trim_latents = 24 #self._quantize_to_multiple(int(seconds_per_fragment * 24 * (overlap_percent / 100.0)), 8)
        echo_frames = 8
        
        previous_latents_path = None
        num_transitions_to_generate = len(keyframe_paths) - 1
        
        for i in range(num_transitions_to_generate):
            progress((i + 1) / num_transitions_to_generate, desc=f"Produzindo Transição {i+1}/{num_transitions_to_generate}")

            start_keyframe_path = keyframe_paths[i]
            destination_keyframe_path = keyframe_paths[i+1]
            present_scene_desc = storyboard[i]

            is_first_fragment = previous_latents_path is None
            if is_first_fragment:
                transition_type = "start"
                motion_prompt = gemini_singleton.get_initial_motion_prompt(
                    global_prompt, start_keyframe_path, destination_keyframe_path, present_scene_desc
                )
            else:
                past_keyframe_path = keyframe_paths[i-1]
                past_scene_desc = storyboard[i-1]
                future_scene_desc = storyboard[i+1] if (i+1) < len(storyboard) else "A cena final."
                decision = gemini_singleton.get_cinematic_decision(
                    global_prompt=global_prompt, story_history=story_history,
                    past_keyframe_path=past_keyframe_path, present_keyframe_path=start_keyframe_path,
                    future_keyframe_path=destination_keyframe_path, past_scene_desc=past_scene_desc,
                    present_scene_desc=present_scene_desc, future_scene_desc=future_scene_desc
                )
                transition_type, motion_prompt = decision["transition_type"], decision["motion_prompt"]
            
            story_history += f"\n- Ato {i+1} ({transition_type}): {motion_prompt}"

            if use_continuity_director: # Assume-se que este checkbox controla os diretores de vídeo e som
                if is_first_fragment:
                    audio_prompt = gemini_singleton.get_sound_director_prompt(
                        audio_history=audio_history,
                        past_keyframe_path=start_keyframe_path, present_keyframe_path=start_keyframe_path,
                        future_keyframe_path=destination_keyframe_path, present_scene_desc=present_scene_desc,
                        motion_prompt=motion_prompt, future_scene_desc=storyboard[i+1] if (i+1) < len(storyboard) else "The final scene."
                    )
                else:
                    audio_prompt = gemini_singleton.get_sound_director_prompt(
                        audio_history=audio_history, past_keyframe_path=keyframe_paths[i-1],
                        present_keyframe_path=start_keyframe_path, future_keyframe_path=destination_keyframe_path,
                        present_scene_desc=present_scene_desc, motion_prompt=motion_prompt,
                        future_scene_desc=storyboard[i+1] if (i+1) < len(storyboard) else "The final scene."
                    )
            else:
                audio_prompt = present_scene_desc # Fallback para o prompt da cena se o diretor de som estiver desligado
            
            audio_history = audio_prompt

            conditioning_items = []
            current_ltx_params = {**base_ltx_params, "handler_strength": handler_strength, "motion_prompt": motion_prompt}
            total_frames_to_generate = self._quantize_to_multiple(int(seconds_per_fragment * 24), 8) + 1

            if is_first_fragment:
                img_start = self._preprocess_image_for_latent_conversion(Image.open(start_keyframe_path).convert("RGB"), target_resolution_tuple)
                start_latent = self.pil_to_latent(img_start)
                conditioning_items.append(LatentConditioningItem(start_latent, 0, 1.0))
                if transition_type != "cut":
                    img_dest = self._preprocess_image_for_latent_conversion(Image.open(destination_keyframe_path).convert("RGB"), target_resolution_tuple)
                    destination_latent = self.pil_to_latent(img_dest)
                    conditioning_items.append(LatentConditioningItem(destination_latent, total_frames_to_generate - 1, destination_convergence_strength))
            else:
                previous_latents = self.load_latent_tensor(previous_latents_path)
                handler_latent = previous_latents[:, :, -1:, :, :]
                trimmed_for_echo = previous_latents[:, :, :-n_trim_latents, :, :] if n_trim_latents > 0 and previous_latents.shape[2] > n_trim_latents else previous_latents
                echo_latents = trimmed_for_echo[:, :, -echo_frames:, :, :]
                handler_frame_position = n_trim_latents + echo_frames
                
                conditioning_items = []

                for i, echo_latent in enumerate(echo_frames):
                      if i == 0:
                           weight = 1.0
                      else:
                           weight = random.uniform(0.2, 0.7)
                           
                           
                           
                conditioning_items.append(LatentConditioningItem(echo_latent, 0, weight))
                #conditioning_items.append(LatentConditioningItem(echo_latents, 0, 1.0))
                conditioning_items.append(LatentConditioningItem(handler_latent, handler_frame_position, handler_strength))
                del previous_latents, handler_latent, trimmed_for_echo, echo_latents; gc.collect()
                if transition_type == "continuous":
                    img_dest = self._preprocess_image_for_latent_conversion(Image.open(destination_keyframe_path).convert("RGB"), target_resolution_tuple)
                    destination_latent = self.pil_to_latent(img_dest)
                    conditioning_items.append(LatentConditioningItem(destination_latent, total_frames_to_generate - 1, destination_convergence_strength))
            
            new_full_latents = self._generate_latent_tensor_internal(conditioning_items, current_ltx_params, target_resolution_tuple, total_frames_to_generate)
            
            base_name = f"fragment_{i}_{int(time.time())}"
            new_full_latents_path = os.path.join(self.workspace_dir, f"{base_name}_full.pt")
            self.save_latent_tensor(new_full_latents, new_full_latents_path)
            
            previous_latents_path = new_full_latents_path

            latents_for_video = new_full_latents
            
            if not is_first_fragment:
                if echo_frames > 0 and latents_for_video.shape[2] > echo_frames: latents_for_video = latents_for_video[:, :, echo_frames:, :, :]
                if n_trim_latents > 0 and latents_for_video.shape[2] > n_trim_latents: latents_for_video = latents_for_video[:, :, :-n_trim_latents, :, :]
            else:
                if n_trim_latents > 0 and latents_for_video.shape[2] > n_trim_latents: latents_for_video = latents_for_video[:, :, :-n_trim_latents, :, :]

            video_with_audio_path = self._generate_video_and_audio_from_latents(latents_for_video, audio_prompt, base_name)
            video_clips_paths.append(video_with_audio_path)
            
            
            if transition_type == "cut":
                previous_latents_path = None
                
                
            yield {"fragment_path": video_with_audio_path}

        final_movie_path = os.path.join(self.workspace_dir, f"final_movie_{int(time.time())}.mp4")
        self.concatenate_videos_ffmpeg(video_clips_paths, final_movie_path)
        
        logger.info(f"Filme completo salvo em: {final_movie_path}")
        yield {"final_path": final_movie_path}

    def _quantize_to_multiple(self, n, m):
        if m == 0: return n
        quantized = int(round(n / m) * m)
        return m if n > 0 and quantized == 0 else quantized