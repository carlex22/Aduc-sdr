# engineers/deformes7D.py
#
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
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License...
# PENDING PATENT NOTICE: Please see NOTICE.md.
#
# Version 3.2.1

import os
import time
import imageio
import numpy as np
import torch
import logging
from PIL import Image, ImageOps
import gradio as gr
import subprocess
import gc
import yaml
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Generator

from aduc_types import LatentConditioningItem
from managers.ltx_manager import ltx_manager_singleton
from managers.latent_enhancer_manager import latent_enhancer_specialist_singleton
from managers.vae_manager import vae_manager_singleton
from engineers.deformes2D_thinker import deformes2d_thinker_singleton
from engineers.deformes3D_thinker import deformes3d_thinker_singleton
from managers.seedvr_manager import seedvr_manager_singleton
from managers.mmaudio_manager import mmaudio_manager_singleton
from tools.video_encode_tool import video_encode_tool_singleton

logger = logging.getLogger(__name__)

class Deformes7DEngine:
    # ... (todo o corpo da classe permanece exatamente o mesmo da nossa última versão) ...
    """
    Unified 3D/4D engine for continuous, interleaved generation of keyframes and video fragments.
    """
    def __init__(self, workspace_dir="deformes_workspace"):
        self.workspace_dir = workspace_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Deformes7D Unified Engine initialized.")
        os.makedirs(self.workspace_dir, exist_ok=True)

    # --- HELPER METHODS ---
    def save_video_from_tensor(self, video_tensor: torch.Tensor, path: str, fps: int = 24):
        """Saves a pixel-space tensor as an MP4 video file."""
        if video_tensor is None or video_tensor.ndim != 5 or video_tensor.shape[2] == 0: return
        video_tensor = video_tensor.squeeze(0).permute(1, 2, 3, 0)
        video_tensor = (video_tensor.clamp(-1, 1) + 1) / 2.0
        video_np = (video_tensor.detach().cpu().float().numpy() * 255).astype(np.uint8)
        with imageio.get_writer(path, fps=fps, codec='libx264', quality=8, output_params=['-pix_fmt', 'yuv420p']) as writer:
            for frame in video_np: writer.append_data(frame)

    def read_video_to_tensor(self, video_path: str) -> torch.Tensor:
        """Reads a video file and converts it into a pixel-space tensor."""
        with imageio.get_reader(video_path, 'ffmpeg') as reader:
            frames = [frame for frame in reader]
        frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2) 
        tensor = tensor.unsqueeze(0)
        tensor = (tensor * 2.0) - 1.0
        return tensor.to(self.device)

    def _preprocess_image(self, image: Image.Image, target_resolution: tuple) -> Image.Image:
        if image.size != target_resolution:
            return ImageOps.fit(image, target_resolution, Image.Resampling.LANCZOS)
        return image

    def _pil_to_pixel_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        return (tensor * 2.0) - 1.0
        
    def _save_image_from_tensor(self, pixel_tensor: torch.Tensor, path: str):
        tensor_chw = pixel_tensor.squeeze(0).squeeze(1)
        tensor_hwc = tensor_chw.permute(1, 2, 0)
        tensor_hwc = (tensor_hwc.clamp(-1, 1) + 1) / 2.0
        image_np = (tensor_hwc.cpu().float().numpy() * 255).astype(np.uint8)
        Image.fromarray(image_np).save(path)

    def _quantize_to_multiple(self, n, m):
        if m == 0: return n
        quantized = int(round(n / m) * m)
        return m if n > 0 and quantized == 0 else quantized

    # --- CORE GENERATION LOGIC ---
    def _generate_next_causal_keyframe(self, base_keyframe_path: str, all_ref_paths: list, 
                                       prompt: str, resolution_tuple: tuple) -> Tuple[str, torch.Tensor]:
        # (código interno deste método permanece o mesmo)
        ltx_context_paths = [base_keyframe_path] + [p for p in all_ref_paths if p != base_keyframe_path][:3]
        ltx_conditioning_items = []
        weight = 1.0
        for path in ltx_context_paths:
            img_pil = Image.open(path).convert("RGB")
            img_processed = self._preprocess_image(img_pil, resolution_tuple)
            pixel_tensor = self._pil_to_pixel_tensor(img_processed)
            latent_tensor = vae_manager_singleton.encode(pixel_tensor)
            ltx_conditioning_items.append(LatentConditioningItem(latent_tensor, 0, weight))
            if weight == 1.0: weight = -0.2
            else: weight -= 0.2
        ltx_base_params = {"guidance_scale": 3.0, "stg_scale": 0.1, "num_inference_steps": 25}
        generated_latents, _ = ltx_manager_singleton.generate_latent_fragment(
            height=resolution_tuple[0], width=resolution_tuple[1],
            conditioning_items_data=ltx_conditioning_items, motion_prompt=prompt,
            video_total_frames=48, video_fps=24, **ltx_base_params
        )
        final_latent = generated_latents[:, :, -1:, :, :]
        upscaled_latent = latent_enhancer_specialist_singleton.upscale(final_latent)
        pixel_tensor_out = vae_manager_singleton.decode(upscaled_latent)
        timestamp = int(time.time() * 1000)
        output_path = os.path.join(self.workspace_dir, f"keyframe_{timestamp}.png")
        self._save_image_from_tensor(pixel_tensor_out, output_path)
        return output_path, final_latent

    def generate_full_movie_interleaved(self, initial_ref_paths: list, storyboard: list, global_prompt: str,
                                        video_resolution: int, seconds_per_fragment: float, trim_percent: int, 
                                        handler_strength: float, dest_strength: float, ltx_params: dict,
                                        progress=gr.Progress()):
        # (código interno deste método permanece o mesmo)
        logger.info("--- DEFORMES 7D: INITIATING INTERLEAVED RENDERING PIPELINE ---")
        run_timestamp = int(time.time())
        temp_video_clips_dir = os.path.join(self.workspace_dir, f"temp_clips_{run_timestamp}")
        os.makedirs(temp_video_clips_dir, exist_ok=True)
        FPS = 24
        FRAMES_PER_LATENT_CHUNK = 8
        resolution_tuple = (video_resolution, video_resolution)
        generated_keyframe_paths, generated_keyframe_latents, generated_video_fragment_paths = [], [], []
        progress(0, desc="Bootstrap: Processing K0...")
        k0_path = initial_ref_paths[0]
        k0_pil = Image.open(k0_path).convert("RGB")
        k0_processed_pil = self._preprocess_image(k0_pil, resolution_tuple)
        k0_pixel_tensor = self._pil_to_pixel_tensor(k0_processed_pil)
        k0_latent = vae_manager_singleton.encode(k0_pixel_tensor)
        generated_keyframe_paths.append(k0_path)
        generated_keyframe_latents.append(k0_latent)
        progress(0.01, desc="Bootstrap: Generating K1...")
        prompt_k1 = deformes2d_thinker_singleton.get_anticipatory_keyframe_prompt(
            global_prompt, "Initial scene.", storyboard[0], storyboard[1], k0_path, initial_ref_paths
        )
        k1_path, k1_latent = self._generate_next_causal_keyframe(k0_path, initial_ref_paths, prompt_k1, resolution_tuple)
        generated_keyframe_paths.append(k1_path)
        generated_keyframe_latents.append(k1_latent)
        story_history = ""
        eco_latent_for_next_loop, dejavu_latent_for_next_loop = None, None
        num_transitions = len(storyboard) - 1
        base_4d_ltx_params = {"rescaling_scale": 0.15, "image_cond_noise_scale": 0.00, **ltx_params}

        for i in range(1, num_transitions):
            act_progress = i / num_transitions
            progress(act_progress, desc=f"Processing Act {i+1}/{num_transitions} (Keyframe Gen)...")
            logger.info(f"--> Step 3D: Generating Keyframe K{i+1}")
            kx_path = generated_keyframe_paths[i]
            prompt_ky = deformes2d_thinker_singleton.get_anticipatory_keyframe_prompt(
                global_prompt, story_history, storyboard[i], storyboard[i+1], kx_path, initial_ref_paths
            )
            ky_path, ky_latent = self._generate_next_causal_keyframe(kx_path, initial_ref_paths, prompt_ky, resolution_tuple)
            generated_keyframe_paths.append(ky_path)
            generated_keyframe_latents.append(ky_latent)
            progress(act_progress, desc=f"Processing Act {i+1}/{num_transitions} (Video Gen)...")
            logger.info(f"--> Step 4D: Generating Video Fragment V{i-1}")
            kb_path, kx_path, ky_path = generated_keyframe_paths[i-1], generated_keyframe_paths[i], generated_keyframe_paths[i+1]
            motion_prompt = deformes3d_thinker_singleton.get_enhanced_motion_prompt(
                global_prompt, story_history, kb_path, kx_path, ky_path,
                storyboard[i-1], storyboard[i], storyboard[i+1]
            )
            transition_type = "continuous"
            story_history += f"\n- Act {i}: {motion_prompt}"
            total_frames_brutos = self._quantize_to_multiple(int(seconds_per_fragment * FPS), FRAMES_PER_LATENT_CHUNK)
            frames_a_podar = self._quantize_to_multiple(int(total_frames_brutos * (trim_percent / 100)), FRAMES_PER_LATENT_CHUNK)
            latents_a_podar = frames_a_podar // FRAMES_PER_LATENT_CHUNK
            DEJAVU_FRAME_TARGET = frames_a_podar - 1 if frames_a_podar > 0 else 0
            DESTINATION_FRAME_TARGET = total_frames_brutos - 1
            conditioning_items = []
            if eco_latent_for_next_loop is None:
                conditioning_items.append(LatentConditioningItem(generated_keyframe_latents[i], 0, 1.0))
            else:
                conditioning_items.append(LatentConditioningItem(eco_latent_for_next_loop, 0, 1.0))
                conditioning_items.append(LatentConditioningItem(dejavu_latent_for_next_loop, DEJAVU_FRAME_TARGET, handler_strength))
            if transition_type != "cut":
                conditioning_items.append(LatentConditioningItem(ky_latent, DESTINATION_FRAME_TARGET, dest_strength))
            fragment_latents_brutos, _ = ltx_manager_singleton.generate_latent_fragment(
                height=video_resolution, width=video_resolution,
                conditioning_items_data=conditioning_items, motion_prompt=motion_prompt,
                video_total_frames=total_frames_brutos, video_fps=FPS, **base_4d_ltx_params
            )
            last_trim = fragment_latents_brutos[:, :, -(latents_a_podar+1):, :, :].clone()
            eco_latent_for_next_loop = last_trim[:, :, :2, :, :].clone()
            dejavu_latent_for_next_loop = last_trim[:, :, -1:, :, :].clone()
            final_fragment_latents = fragment_latents_brutos[:, :, :-(latents_a_podar-1), :, :].clone()
            final_fragment_latents = final_fragment_latents[:, :, 1:, :, :]
            pixel_tensor = vae_manager_singleton.decode(final_fragment_latents)
            fragment_path = os.path.join(temp_video_clips_dir, f"fragment_{i-1}.mp4")
            self.save_video_from_tensor(pixel_tensor, fragment_path, fps=FPS)
            generated_video_fragment_paths.append(fragment_path)
            logger.info(f"Video Fragment V{i-1} saved to {fragment_path}")

        logger.info("--- Final Assembly of Video Fragments ---")
        final_video_path = os.path.join(self.workspace_dir, f"movie_7D_{run_timestamp}.mp4")
        video_encode_tool_singleton.concatenate_videos(generated_video_fragment_paths, final_video_path, self.workspace_dir)
        shutil.rmtree(temp_video_clips_dir)
        logger.info(f"Full movie generated at: {final_video_path}")
        return {"final_path": final_video_path, "all_keyframes": generated_keyframe_paths, "latent_paths": "NOT_IMPLEMENTED_YET"}

    # --- POST-PRODUCTION METHODS ---
    def task_run_latent_upscaling(self, latent_paths: list, chunk_size: int, progress: gr.Progress) -> Generator[Dict[str, any], None, None]:
        # (código interno deste método permanece o mesmo)
        if not latent_paths:
            raise gr.Error("Cannot perform upscaling: no latent paths were provided from the main generation.")
        logger.info("--- POST-PRODUCTION: Latent Upscaling ---")
        run_timestamp = int(time.time())
        temp_upscaled_clips_dir = os.path.join(self.workspace_dir, f"temp_upscaled_clips_{run_timestamp}")
        os.makedirs(temp_upscaled_clips_dir, exist_ok=True)
        final_upscaled_clip_paths = []
        num_chunks = -(-len(latent_paths) // chunk_size)
        for i in range(num_chunks):
            chunk_start_index = i * chunk_size
            chunk_end_index = chunk_start_index + chunk_size
            chunk_paths = latent_paths[chunk_start_index:chunk_end_index]
            progress(i / num_chunks, desc=f"Upscaling & Decoding Batch {i+1}/{num_chunks}")
            tensors_in_chunk = [torch.load(p, map_location=self.device) for p in chunk_paths]
            tensors_para_concatenar = [frag[:, :, :-1, :, :] if j < len(tensors_in_chunk) - 1 else frag for j, frag in enumerate(tensors_in_chunk)]
            sub_group_latent = torch.cat(tensors_para_concatenar, dim=2)
            del tensors_in_chunk, tensors_para_concatenar; gc.collect(); torch.cuda.empty_cache()
            upscaled_latent_chunk = latent_enhancer_specialist_singleton.upscale(sub_group_latent)
            del sub_group_latent; gc.collect(); torch.cuda.empty_cache()
            pixel_tensor = vae_manager_singleton.decode(upscaled_latent_chunk)
            del upscaled_latent_chunk; gc.collect(); torch.cuda.empty_cache()
            base_name = f"upscaled_clip_{i:04d}_{run_timestamp}"
            current_clip_path = os.path.join(temp_upscaled_clips_dir, f"{base_name}.mp4")
            self.save_video_from_tensor(pixel_tensor, current_clip_path, fps=24)
            final_upscaled_clip_paths.append(current_clip_path)
            del pixel_tensor; gc.collect(); torch.cuda.empty_cache()
        progress(0.98, desc="Assembling upscaled clips...")
        final_video_path = os.path.join(self.workspace_dir, f"upscaled_movie_{run_timestamp}.mp4")
        video_encode_tool_singleton.concatenate_videos(video_paths=final_upscaled_clip_paths, output_path=final_video_path, workspace_dir=self.workspace_dir)
        shutil.rmtree(temp_upscaled_clips_dir)
        logger.info(f"Latent upscaling complete! Final video at: {final_video_path}")
        yield {"final_path": final_video_path}

    def master_video_hd(self, source_video_path: str, model_version: str, steps: int, prompt: str, progress: gr.Progress):
        # (código interno deste método permanece o mesmo)
        logger.info(f"--- POST-PRODUCTION: HD Mastering with SeedVR {model_version} ---")
        run_timestamp = int(time.time())
        output_path = os.path.join(self.workspace_dir, f"{Path(source_video_path).stem}_hd.mp4")
        try:
            final_path = seedvr_manager_singleton.process_video(
                input_video_path=source_video_path, output_video_path=output_path,
                prompt=prompt, model_version=model_version, steps=steps, progress=progress
            )
            yield {"final_path": final_path}
        except Exception as e:
            logger.error(f"HD Mastering failed: {e}", exc_info=True)
            raise gr.Error(f"HD Mastering failed. Details: {e}")
    
    def generate_audio(self, source_video_path: str, audio_prompt: str, progress: gr.Progress):
        # (código interno deste método permanece o mesmo)
        logger.info(f"--- POST-PRODUCTION: Audio Generation ---")
        run_timestamp = int(time.time())
        output_path = os.path.join(self.workspace_dir, f"{Path(source_video_path).stem}_audio.mp4")
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", source_video_path],
                capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            progress(0.5, desc="Generating audio track...")
            final_path = mmaudio_manager_singleton.generate_audio_for_video(
                video_path=source_video_path, prompt=audio_prompt,
                duration_seconds=duration, output_path_override=output_path
            )
            yield {"final_path": final_path}
        except Exception as e:
            logger.error(f"Audio generation failed: {e}", exc_info=True)
            raise gr.Error(f"Audio generation failed. Details: {e}")

# --- Singleton Instantiation ---
try:
    config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    deformes7d_engine_singleton = Deformes7DEngine(workspace_dir=WORKSPACE_DIR)
# <--- INÍCIO DA CORREÇÃO --->
except Exception as e:
    # Loga o erro como CRÍTICO, pois a aplicação não pode funcionar sem este motor.
    logger.critical(f"CRITICAL: Failed to initialize the Deformes7DEngine singleton from {config_path}: {e}", exc_info=True)
    # Relança a exceção para parar a aplicação imediatamente.
    # Isso evita o erro 'NoneType' mais tarde e fornece um ponto claro de falha.
    raise
# <--- FIM DA CORREÇÃO --->