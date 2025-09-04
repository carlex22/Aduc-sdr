# engineers/deformes4D.py
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
# Version 2.0.1

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
import gc
import shutil
from pathlib import Path
from typing import List, Tuple, Generator, Dict, Any

from aduc_types import LatentConditioningItem
from managers.ltx_manager import ltx_manager_singleton
from managers.latent_enhancer_manager import latent_enhancer_specialist_singleton
from managers.vae_manager import vae_manager_singleton
from engineers.deformes2D_thinker import deformes2d_thinker_singleton
from managers.seedvr_manager import seedvr_manager_singleton
from managers.mmaudio_manager import mmaudio_manager_singleton
from tools.video_encode_tool import video_encode_tool_singleton

logger = logging.getLogger(__name__)

class Deformes4DEngine:
    """
    Implements the Camera (Ψ) and Distiller (Δ) of the ADUC-SDR architecture.
    Orchestrates the generation, latent post-production, and final rendering of video fragments.
    """
    def __init__(self, workspace_dir="deformes_workspace"):
        self.workspace_dir = workspace_dir
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Deformes4D Specialist (ADUC-SDR Executor) initialized.")
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
        # (F, H, W, C) -> (C, F, H, W)
        tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2) 
        tensor = tensor.unsqueeze(0) # (B, C, F, H, W)
        tensor = (tensor * 2.0) - 1.0 # Normalize to [-1, 1]
        return tensor.to(self.device)

    def _preprocess_image_for_latent_conversion(self, image: Image.Image, target_resolution: tuple) -> Image.Image:
        """Resizes and fits an image to the target resolution for VAE encoding."""
        if image.size != target_resolution:
            return ImageOps.fit(image, target_resolution, Image.Resampling.LANCZOS)
        return image

    def pil_to_latent(self, pil_image: Image.Image) -> torch.Tensor:
        """Converts a PIL Image to a latent tensor by calling the VaeManager."""
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        tensor = (tensor * 2.0) - 1.0
        return vae_manager_singleton.encode(tensor)

    # --- CORE ADUC-SDR LOGIC ---

    def generate_original_movie(self, keyframes: list, global_prompt: str, storyboard: list,
                                seconds_per_fragment: float, trim_percent: int,
                                handler_strength: float, destination_convergence_strength: float,
                                video_resolution: int, use_continuity_director: bool,
                                guidance_scale: float, stg_scale: float, num_inference_steps: int,
                                progress: gr.Progress = gr.Progress()):
        FPS = 24
        FRAMES_PER_LATENT_CHUNK = 8
        LATENT_PROCESSING_CHUNK_SIZE = 4

        run_timestamp = int(time.time())
        temp_latent_dir = os.path.join(self.workspace_dir, f"temp_latents_{run_timestamp}")
        temp_video_clips_dir = os.path.join(self.workspace_dir, f"temp_clips_{run_timestamp}")
        os.makedirs(temp_latent_dir, exist_ok=True)
        os.makedirs(temp_video_clips_dir, exist_ok=True)

        total_frames_brutos = self._quantize_to_multiple(int(seconds_per_fragment * FPS), FRAMES_PER_LATENT_CHUNK)
        frames_a_podar = self._quantize_to_multiple(int(total_frames_brutos * (trim_percent / 100)), FRAMES_PER_LATENT_CHUNK)
        latents_a_podar = frames_a_podar // FRAMES_PER_LATENT_CHUNK
        total_latent_frames = total_frames_brutos // FRAMES_PER_LATENT_CHUNK

        DEJAVU_FRAME_TARGET = frames_a_podar - 1 if frames_a_podar > 0 else 0
        DESTINATION_FRAME_TARGET = total_frames_brutos - 1

        base_ltx_params = {"guidance_scale": guidance_scale, "stg_scale": stg_scale, "num_inference_steps": num_inference_steps, "rescaling_scale": 0.15, "image_cond_noise_scale": 0.00}
        keyframe_paths = [item[0] if isinstance(item, tuple) else item for item in keyframes]
        story_history = ""
        target_resolution_tuple = (video_resolution, video_resolution)
        eco_latent_for_next_loop, dejavu_latent_for_next_loop = None, None
        latent_fragment_paths = []

        if len(keyframe_paths) < 2: raise gr.Error(f"Generation requires at least 2 keyframes. You provided {len(keyframe_paths)}.")
        num_transitions_to_generate = len(keyframe_paths) - 1

        logger.info("--- STARTING STAGE 1: Latent Fragment Generation ---")
        for i in range(num_transitions_to_generate):
            fragment_index = i + 1
            progress(i / num_transitions_to_generate, desc=f"Generating Latent {fragment_index}/{num_transitions_to_generate}")
            past_keyframe_path = keyframe_paths[i - 1] if i > 0 else keyframe_paths[i]
            start_keyframe_path = keyframe_paths[i]
            destination_keyframe_path = keyframe_paths[i + 1]
            future_story_prompt = storyboard[i + 1] if (i + 1) < len(storyboard) else "The final scene."
            logger.info(f"Calling deformes2D_thinker to generate cinematic decision for fragment {fragment_index}...")
            decision = deformes2d_thinker_singleton.get_cinematic_decision(global_prompt, story_history, past_keyframe_path, start_keyframe_path, destination_keyframe_path, storyboard[i - 1] if i > 0 else "The beginning.", storyboard[i], future_story_prompt)
            transition_type, motion_prompt = decision["transition_type"], decision["motion_prompt"]
            story_history += f"\n- Act {fragment_index}: {motion_prompt}"
            
            conditioning_items = []
            if eco_latent_for_next_loop is None:
               img_start = self._preprocess_image_for_latent_conversion(Image.open(start_keyframe_path).convert("RGB"), target_resolution_tuple)
               conditioning_items.append(LatentConditioningItem(self.pil_to_latent(img_start), 0, 1.0))
            else:
               conditioning_items.append(LatentConditioningItem(eco_latent_for_next_loop, 0, 1.0))
               conditioning_items.append(LatentConditioningItem(dejavu_latent_for_next_loop, DEJAVU_FRAME_TARGET, handler_strength))
            
            if transition_type == "cut":
                logger.info(f"Cinematic Director chose a 'cut'. Creating FFmpeg transition bridge...")
                bridge_duration_seconds = FRAMES_PER_LATENT_CHUNK / FPS
                bridge_video_path = video_encode_tool_singleton.create_transition_bridge(
                    start_image_path=start_keyframe_path, end_image_path=destination_keyframe_path,
                    duration=bridge_duration_seconds, fps=FPS, target_resolution=target_resolution_tuple,
                    workspace_dir=self.workspace_dir
                )
                bridge_pixel_tensor = self.read_video_to_tensor(bridge_video_path)
                bridge_latent_tensor = vae_manager_singleton.encode(bridge_pixel_tensor)
                final_fade_latent = bridge_latent_tensor[:, :, -1:, :, :]
                conditioning_items.append(LatentConditioningItem(final_fade_latent, total_latent_frames - 1, 0.95))
                img_dest = self._preprocess_image_for_latent_conversion(Image.open(destination_keyframe_path).convert("RGB"), target_resolution_tuple)
                conditioning_items.append(LatentConditioningItem(self.pil_to_latent(img_dest), DESTINATION_FRAME_TARGET, destination_convergence_strength * 0.5))
                del bridge_pixel_tensor, bridge_latent_tensor, final_fade_latent
                if os.path.exists(bridge_video_path): os.remove(bridge_video_path)
            else:
               img_dest = self._preprocess_image_for_latent_conversion(Image.open(destination_keyframe_path).convert("RGB"), target_resolution_tuple)
               conditioning_items.append(LatentConditioningItem(self.pil_to_latent(img_dest), DESTINATION_FRAME_TARGET, destination_convergence_strength))
            
            current_ltx_params = {**base_ltx_params, "motion_prompt": motion_prompt}
            logger.info(f"Calling LTX to generate video latents for fragment {fragment_index} ({total_frames_brutos} frames)...")
            latents_brutos, _ = self._generate_latent_tensor_internal(conditioning_items, current_ltx_params, target_resolution_tuple, total_frames_brutos)
            num_latent_frames = latents_brutos.shape[2]
            logger.info(f"LTX responded with a latent tensor of shape {latents_brutos.shape}, representing ~{num_latent_frames * 8 + 1} video frames at {FPS} FPS.")
            
            last_trim = latents_brutos[:, :, -(latents_a_podar+1):, :, :].clone()
            eco_latent_for_next_loop = last_trim[:, :, :2, :, :].clone()
            dejavu_latent_for_next_loop = last_trim[:, :, -1:, :, :].clone()
            latents_video = latents_brutos[:, :, :-(latents_a_podar-1), :, :].clone()
            latents_video = latents_video[:, :, 1:, :, :]
            del last_trim, latents_brutos; gc.collect(); torch.cuda.empty_cache()
            
            if transition_type == "cut":
                eco_latent_for_next_loop, dejavu_latent_for_next_loop = None, None
            
            cpu_latent = latents_video.cpu()
            latent_path = os.path.join(temp_latent_dir, f"latent_fragment_{i:04d}.pt")
            torch.save(cpu_latent, latent_path)
            latent_fragment_paths.append(latent_path)
            del latents_video, cpu_latent; gc.collect()
        del eco_latent_for_next_loop, dejavu_latent_for_next_loop; gc.collect(); torch.cuda.empty_cache()

        logger.info(f"--- STARTING STAGE 2: Processing {len(latent_fragment_paths)} latents in chunks of {LATENT_PROCESSING_CHUNK_SIZE} ---")
        final_video_clip_paths = []
        num_chunks = -(-len(latent_fragment_paths) // LATENT_PROCESSING_CHUNK_SIZE)
        for i in range(num_chunks):
            chunk_start_index = i * LATENT_PROCESSING_CHUNK_SIZE
            chunk_end_index = chunk_start_index + LATENT_PROCESSING_CHUNK_SIZE
            chunk_paths = latent_fragment_paths[chunk_start_index:chunk_end_index]
            progress(i / num_chunks, desc=f"Processing & Decoding Batch {i+1}/{num_chunks}")
            tensors_in_chunk = [torch.load(p, map_location=self.device) for p in chunk_paths]
            tensors_para_concatenar = [frag[:, :, :-1, :, :] if j < len(tensors_in_chunk) - 1 else frag for j, frag in enumerate(tensors_in_chunk)]
            sub_group_latent = torch.cat(tensors_para_concatenar, dim=2)
            del tensors_in_chunk, tensors_para_concatenar; gc.collect(); torch.cuda.empty_cache()
            logger.info(f"Batch {i+1} concatenated. Latent shape: {sub_group_latent.shape}")
            base_name = f"clip_{i:04d}_{run_timestamp}"
            current_clip_path = os.path.join(temp_video_clips_dir, f"{base_name}.mp4")
            pixel_tensor = vae_manager_singleton.decode(sub_group_latent)
            self.save_video_from_tensor(pixel_tensor, current_clip_path, fps=FPS)
            del pixel_tensor, sub_group_latent; gc.collect(); torch.cuda.empty_cache()
            final_video_clip_paths.append(current_clip_path)

        progress(0.98, desc="Final assembly of clips...")
        final_video_path = os.path.join(self.workspace_dir, f"original_movie_{run_timestamp}.mp4")
        video_encode_tool_singleton.concatenate_videos(video_paths=final_video_clip_paths, output_path=final_video_path, workspace_dir=self.workspace_dir)
        logger.info("Cleaning up temporary clip files...")
        try:
            shutil.rmtree(temp_video_clips_dir)
        except OSError as e:
            logger.warning(f"Could not remove temporary clip directory: {e}")
        logger.info(f"Process complete! Original video saved to: {final_video_path}")
        return {"final_path": final_video_path, "latent_paths": latent_fragment_paths}

    def upscale_latents_and_create_video(self, latent_paths: list, chunk_size: int, progress: gr.Progress):
        if not latent_paths:
            raise gr.Error("Cannot perform upscaling: no latent paths were provided.")
        logger.info("--- STARTING POST-PRODUCTION: Latent Upscaling ---")
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
            logger.info(f"Batch {i+1} loaded. Original latent shape: {sub_group_latent.shape}")
            upscaled_latent_chunk = latent_enhancer_specialist_singleton.upscale(sub_group_latent)
            del sub_group_latent; gc.collect(); torch.cuda.empty_cache()
            logger.info(f"Batch {i+1} upscaled. New latent shape: {upscaled_latent_chunk.shape}")
            pixel_tensor = vae_manager_singleton.decode(upscaled_latent_chunk)
            del upscaled_latent_chunk; gc.collect(); torch.cuda.empty_cache()
            base_name = f"upscaled_clip_{i:04d}_{run_timestamp}"
            current_clip_path = os.path.join(temp_upscaled_clips_dir, f"{base_name}.mp4")
            self.save_video_from_tensor(pixel_tensor, current_clip_path, fps=24)
            final_upscaled_clip_paths.append(current_clip_path)
            del pixel_tensor; gc.collect(); torch.cuda.empty_cache()
            logger.info(f"Saved upscaled clip: {Path(current_clip_path).name}")
        progress(0.98, desc="Assembling upscaled clips...")
        final_video_path = os.path.join(self.workspace_dir, f"upscaled_movie_{run_timestamp}.mp4")
        video_encode_tool_singleton.concatenate_videos(video_paths=final_upscaled_clip_paths, output_path=final_video_path, workspace_dir=self.workspace_dir)
        logger.info("Cleaning up temporary upscaled clip files...")
        try:
            shutil.rmtree(temp_upscaled_clips_dir)
        except OSError as e:
            logger.warning(f"Could not remove temporary upscaled clip directory: {e}")
        logger.info(f"Latent upscaling complete! Final video at: {final_video_path}")
        yield {"final_path": final_video_path}

    def master_video_hd(self, source_video_path: str, model_version: str, steps: int, prompt: str, progress: gr.Progress):
        logger.info(f"--- STARTING POST-PRODUCTION: HD Mastering with SeedVR {model_version} ---")
        progress(0.1, desc=f"Preparing for HD Mastering with SeedVR {model_version}...")
        run_timestamp = int(time.time())
        output_path = os.path.join(self.workspace_dir, f"hd_mastered_movie_{model_version}_{run_timestamp}.mp4")
        try:
            final_path = seedvr_manager_singleton.process_video(
                input_video_path=source_video_path,
                output_video_path=output_path,
                prompt=prompt,
                model_version=model_version,
                steps=steps,
                progress=progress
            )
            logger.info(f"HD Mastering complete! Final video at: {final_path}")
            yield {"final_path": final_path}
        except Exception as e:
            logger.error(f"HD Mastering failed: {e}", exc_info=True)
            raise gr.Error(f"HD Mastering failed. Details: {e}")
    
    def generate_audio_for_final_video(self, source_video_path: str, audio_prompt: str, progress: gr.Progress):
        logger.info(f"--- STARTING POST-PRODUCTION: Audio Generation ---")
        progress(0.1, desc="Preparing for audio generation...")
        run_timestamp = int(time.time())
        source_name = Path(source_video_path).stem
        output_path = os.path.join(self.workspace_dir, f"{source_name}_with_audio_{run_timestamp}.mp4")
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", source_video_path],
                capture_output=True, text=True, check=True)
            duration = float(result.stdout.strip())
            logger.info(f"Source video duration: {duration:.2f} seconds.")
            progress(0.5, desc="Generating audio track...")
            final_path = mmaudio_manager_singleton.generate_audio_for_video(
                video_path=source_video_path,
                prompt=audio_prompt,
                duration_seconds=duration,
                output_path_override=output_path
            )
            logger.info(f"Audio generation complete! Final video with audio at: {final_path}")
            progress(1.0, desc="Audio generation complete!")
            yield {"final_path": final_path}
        except Exception as e:
            logger.error(f"Audio generation failed: {e}", exc_info=True)
            raise gr.Error(f"Audio generation failed. Details: {e}")

    def _generate_latent_tensor_internal(self, conditioning_items, ltx_params, target_resolution, total_frames_to_generate):
        """Internal helper to call the LTX manager."""
        final_ltx_params = {**ltx_params, 'width': target_resolution[0], 'height': target_resolution[1], 'video_total_frames': total_frames_to_generate, 'video_fps': 24, 'current_fragment_index': int(time.time()), 'conditioning_items_data': conditioning_items}
        return ltx_manager_singleton.generate_latent_fragment(**final_ltx_params)

    def _quantize_to_multiple(self, n, m):
        """Helper to round n to the nearest multiple of m."""
        if m == 0: return n
        quantized = int(round(n / m) * m)
        return m if n > 0 and quantized == 0 else quantized