# deformes4D_engine.py
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# ORIGINAL SOURCE: LTX-Video by Lightricks Ltd. & other open-source projects.
# Licensed under the Apache License, Version 2.0
# https://github.com/Lightricks/LTX-Video
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
import json

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
        os.makedirs(self.workspace_dir, exist_ok=True)
        self._vae = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Especialista Deformes4D (SDR Executor) inicializado com a nova lógica.")

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
        if latent_tensor is None or latent_tensor.shape[2] == 0:
            logger.error("Tentativa de decodificar um tensor latente com 0 frames. Abortando.")
            return None
        latent_tensor = latent_tensor.to(self.device, dtype=self.vae.dtype)
        timestep_tensor = torch.tensor([decode_timestep] * latent_tensor.shape[0], device=self.device, dtype=latent_tensor.dtype)
        return vae_decode(latent_tensor, self.vae, is_video=True, timestep=timestep_tensor, vae_per_channel_normalize=True)

    def save_video_from_tensor(self, video_tensor: torch.Tensor, path: str, fps: int = 24):
        if video_tensor is None or video_tensor.ndim != 5 or video_tensor.shape[2] == 0:
            logger.warning(f"Não foi possível salvar o vídeo em {path} pois o tensor de vídeo está vazio ou inválido.")
            return
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

    def _get_video_duration(self, video_path: str) -> float:
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                capture_output=True, text=True, check=True
            )
            return float(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            logger.warning(f"Não foi possível obter a duração de {os.path.basename(video_path)} via ffprobe: {e}. Retornando 0.")
            return 0.0

    def _apply_fade_out_ffmpeg(self, input_path: str, output_path: str, fade_duration: float = 0.5) -> str:
        duration = self._get_video_duration(input_path)
        if duration <= fade_duration:
            logger.warning("Duração do vídeo é menor que a duração do fade. Fade não aplicado.")
            return input_path
        
        fade_start_time = duration - fade_duration
        
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', f'fade=type=out:start_time={fade_start_time}:duration={fade_duration}',
            '-af', f'afade=type=out:start_time={fade_start_time}:duration={fade_duration}',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'medium',
            output_path
        ]
        
        logger.info(f"Aplicando fade-out no vídeo: {os.path.basename(input_path)}")
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Vídeo com fade-out salvo em: {os.path.basename(output_path)}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no FFmpeg ao aplicar fade: {e.stderr}")
            return input_path

    def concatenate_videos_ffmpeg(self, video_paths: list[str], output_path: str) -> str:
        if not video_paths:
            raise gr.Error("Nenhum fragmento de vídeo para montar.")

        list_file_path = os.path.join(self.workspace_dir, "concat_list.txt")
        with open(list_file_path, 'w', encoding='utf-8') as f:
            for path in video_paths:
                absolute_path = os.path.abspath(path)
                safe_path = absolute_path.replace("'", "'\\''")
                f.write(f"file '{safe_path}'\n")

        cmd_list = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-c', 'copy', output_path]
        
        logger.info(f"Executando concatenação FFmpeg via demuxer com caminhos absolutos...")
        try:
            subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no FFmpeg: {e.stderr}")
            raise gr.Error(f"Falha na montagem final do vídeo. Detalhes: {e.stderr}")
            
        return output_path

    def _quantize_to_multiple(self, n, m):
        # AJUSTE: Arredonda para CIMA para o próximo múltiplo de m.
        if m == 0: return n
        if n == 0: return 0
        return ((n - 1) // m + 1) * m
        
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
        
        logger.info("LOG: Iniciando a produção do filme completo com a LÓGICA CORRETA.")
        keyframe_paths = [item[0] if isinstance(item, tuple) else item for item in keyframes]
        
        if len(keyframe_paths) < 3:
            raise gr.Error("São necessários pelo menos 3 keyframes para usar a lógica (passado, presente, futuro).")
        
        video_clips_paths = []
        story_history = ""
        target_resolution = (video_resolution, video_resolution) 
        
        # PONTO CRÍTICO 1: Quantização de todas as contagens para múltiplos de 8 (arredondando para cima).
        logger.info("LOG: Quantizando contagens de frames para múltiplos de 8 (arredondando para cima).")
        
        n_echo_latents = self._quantize_to_multiple(echo_frames, 8)
        
        n_trim_latents = self._quantize_to_multiple(
            int(seconds_per_fragment * 24 * (overlap_percent / 100.0)), 8
        )

        visible_frames = self._quantize_to_multiple(int(seconds_per_fragment * 24), 8)
        total_frames_to_generate = visible_frames + n_echo_latents

        logger.info(f"  - Frames de Eco: {echo_frames} -> {n_echo_latents}")
        logger.info(f"  - Frames de Sobreposição/Corte: {n_trim_latents}")
        logger.info(f"  - Frames Visíveis por Fragmento: {visible_frames}")
        logger.info(f"  - Total de Frames a Gerar por Chamada LTX: {total_frames_to_generate}")

        if total_frames_to_generate <= (n_echo_latents + n_trim_latents):
            raise gr.Error(f"Configuração inválida: O número total de frames a gerar ({total_frames_to_generate}) não é suficiente para cobrir os frames de eco ({n_echo_latents}) e sobreposição ({n_trim_latents}). Aumente a duração do fragmento ou diminua o eco/sobreposição.")

        previous_latents_path = None
        
        num_transitions_to_generate = len(keyframe_paths) - 2
        
        for i in range(1, len(keyframe_paths) - 1):
            progress_val = (i) / num_transitions_to_generate
            progress(progress_val, desc=f"Produzindo Transição {i}/{num_transitions_to_generate}")

            logger.info(f"\n======================================================================")
            logger.info(f"LOG: INICIANDO FRAGMENTO {i} / {num_transitions_to_generate} (Keyframe {i} -> {i+1})")
            logger.info(f"======================================================================")
            
            # ... (lógica do Gemini permanece a mesma) ...
            past_keyframe_path = keyframe_paths[i-1]
            present_keyframe_path = keyframe_paths[i]
            future_keyframe_path = keyframe_paths[i+1]
            past_scene_desc = storyboard[i-1]
            present_scene_desc = storyboard[i]
            future_scene_desc = storyboard[i+1]
            logger.info("LOG: Invocando o Diretor de Cinema (Gemini) para decisão de transição.")
            decision = {"transition_type": "continuous", "motion_prompt": f"Transição de '{present_scene_desc}' para '{future_scene_desc}'."}
            if use_continuity_director:
                decision = gemini_singleton.get_cinematic_decision(
                    global_prompt=global_prompt, story_history=story_history,
                    past_keyframe_path=past_keyframe_path, present_keyframe_path=present_keyframe_path,
                    future_keyframe_path=future_keyframe_path, past_scene_desc=past_scene_desc,
                    present_scene_desc=present_scene_desc, future_scene_desc=future_scene_desc
                )
            transition_type = decision.get("transition_type", "continuous").lower()
            motion_prompt = decision.get("motion_prompt", "")
            logger.info(f"LOG: Decisão do Diretor: '{transition_type}'. Motion Prompt: '{motion_prompt[:100]}...'")

            conditioning_items = []
            if previous_latents_path is None:
                # ... (lógica do modo inicial permanece a mesma) ...
                logger.info("LOG: MODO INICIAL (previous_latents_path is None). Condicionando com keyframes.")
                pil_present = Image.open(present_keyframe_path).convert("RGB")
                pil_present = ImageOps.fit(pil_present, target_resolution, Image.Resampling.LANCZOS)
                latent_present = self.pil_to_latent(pil_present)
                if transition_type == "continuous":
                    pil_future = Image.open(future_keyframe_path).convert("RGB")
                    pil_future = ImageOps.fit(pil_future, target_resolution, Image.Resampling.LANCZOS)
                    latent_future = self.pil_to_latent(pil_future)
                    conditioning_items.append(LatentConditioningItem(latent_present, 0, 1.0))
                    conditioning_items.append(LatentConditioningItem(latent_future, total_frames_to_generate - 1, destination_convergence_strength))
                    logger.info("  - Condicionamento: Keyframe Presente (frame 0) + Keyframe Futuro (frame final).")
                else: 
                    conditioning_items.append(LatentConditioningItem(latent_present, 0, 1.0))
                    logger.info("  - Condicionamento: Apenas Keyframe Presente (frame 0).")
            else:
                # ... (lógica do modo contínuo permanece a mesma) ...
                logger.info("LOG: MODO CONTÍNUO (previous_latents_path existe). Condicionando com Eco e Handler.")
                previous_latents = self.load_latent_tensor(previous_latents_path)
                tl = previous_latents.shape[2]
                echo_start = tl - n_trim_latents - n_echo_latents
                echo_end = tl - n_trim_latents
                if echo_start < 0 or echo_end > tl:
                    raise ValueError(f"Cálculo de eco inválido. Tl={tl}, trim={n_trim_latents}, echo={n_echo_latents}")
                echo_latents = previous_latents[:, :, echo_start:echo_end, :, :]
                logger.info(f"  - Extraindo ECO dos frames [{echo_start}:{echo_end}] do latente anterior.")
                handler_latent = previous_latents[:, :, -1:, :, :]
                logger.info(f"  - Extraindo HANDLER (último frame) do latente anterior.")
                conditioning_items.append(LatentConditioningItem(echo_latents, 0, 1.0))
                conditioning_items.append(LatentConditioningItem(handler_latent, n_echo_latents, handler_strength))
                if transition_type == "continuous":
                    pil_future = Image.open(future_keyframe_path).convert("RGB")
                    pil_future = ImageOps.fit(pil_future, target_resolution, Image.Resampling.LANCZOS)
                    latent_future = self.pil_to_latent(pil_future)
                    conditioning_items.append(LatentConditioningItem(latent_future, total_frames_to_generate - 1, destination_convergence_strength))
                    logger.info("  - Adicionando Keyframe Futuro como destino.")
                del previous_latents, echo_latents, handler_latent
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

            final_ltx_params = {
                **base_ltx_params, 'motion_prompt': motion_prompt,
                'width': target_resolution[0], 'height': target_resolution[1], 
                'video_total_frames': total_frames_to_generate, 'video_fps': 24,
                'conditioning_items_data': conditioning_items,
                'current_fragment_index': i,
            }
            
            logger.info(f"LOG: Gerando novo latente com {total_frames_to_generate} frames...")
            new_full_latents, _ = self.ltx_manager.generate_latent_fragment(**final_ltx_params)
            logger.info(f"  - Geração concluída. Shape original: {list(new_full_latents.shape)}")
            
            # PONTO CRÍTICO 2: Remover o último "frame fantasma" adicionado pelo LTX.
            if new_full_latents is not None and new_full_latents.shape[2] > 0:
                new_full_latents = new_full_latents[:, :, :-1, :, :]
                logger.info(f"  - Removido frame fantasma. Novo Shape: {list(new_full_latents.shape)}")
            
            base_name = f"fragment_{i}_{int(time.time())}"
            new_full_latents_path = os.path.join(self.workspace_dir, f"{base_name}_full.pt")
            self.save_latent_tensor(new_full_latents, new_full_latents_path)
            
            previous_latents_path = new_full_latents_path
            
            logger.info("LOG: Criando clone para renderização de vídeo (cortando eco e sobreposição).")
            end_slice = -n_trim_latents if n_trim_latents > 0 else new_full_latents.shape[2]
            latents_for_video = new_full_latents[:, :, n_echo_latents:end_slice, :, :]
            logger.info(f"  - Shape final para vídeo: {list(latents_for_video.shape)}")
            
            silent_video_path = os.path.join(self.workspace_dir, f"{base_name}_silent.mp4")
            pixel_tensor = self.latents_to_pixels(latents_for_video)
            self.save_video_from_tensor(pixel_tensor, silent_video_path, fps=24)
            
            video_to_process = silent_video_path
            
            if transition_type == "cut":
                logger.info("LOG: Transição 'cut' detectada. Aplicando fade e resetando memória.")
                faded_video_path = os.path.join(self.workspace_dir, f"{base_name}_faded.mp4")
                self._apply_fade_out_ffmpeg(silent_video_path, faded_video_path)
                video_to_process = faded_video_path
                
                previous_latents_path = None
                logger.info("  - MEMÓRIA APAGADA. Próximo fragmento será no modo inicial.")

            if os.path.exists(video_to_process) and os.path.getsize(video_to_process) > 0:
                frag_duration = self._get_video_duration(video_to_process)
                video_with_audio_path = audio_specialist_singleton.generate_audio_for_video(
                    video_path=video_to_process, prompt=present_scene_desc,
                    negative_prompt="music, speech", duration_seconds=frag_duration)
                
                video_clips_paths.append(video_with_audio_path)
                yield {"fragment_path": video_with_audio_path}
            else:
                 logger.warning(f"O vídeo para o fragmento {i} não foi gerado corretamente, pulando este fragmento.")
                 video_with_audio_path = None

            if os.path.exists(silent_video_path): os.remove(silent_video_path)
            if transition_type == "cut" and os.path.exists(video_to_process):
                 if video_with_audio_path and video_to_process != video_with_audio_path:
                    os.remove(video_to_process)

            story_history += f"\n- Ato {i} ({transition_type}): {motion_prompt}"
            logger.info(f"LOG: FRAGMENTO {i} CONCLUÍDO. Vídeo: {os.path.basename(video_with_audio_path) if video_with_audio_path else 'N/A'}")
            
            del new_full_latents, pixel_tensor, latents_for_video
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        logger.info("LOG: Todas as transições geradas. Iniciando montagem final.")
        progress(1.0, desc="Montagem final do filme...")
        final_movie_path = os.path.join(self.workspace_dir, f"final_movie_{int(time.time())}.mp4")
        
        self.concatenate_videos_ffmpeg(video_clips_paths, final_movie_path)

        logger.info(f"LOG: Filme completo salvo em: {final_movie_path}")
        yield {"final_path": final_movie_path}