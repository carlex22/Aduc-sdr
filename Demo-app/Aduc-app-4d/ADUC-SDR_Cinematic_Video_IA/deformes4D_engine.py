#Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
# YouTube (Resultados): https://m.youtube.com/channel/UC3EgoJi_Fv7yuDpvfYNtoIQ
#
# Este programa é software livre: você pode redistribuí-lo e/ou modificá-lo
# sob os termos da Licença Pública Geral Affero da GNU como publicada pela
# Free Software Foundation, seja a versão 3 da Licença, ou
# (a seu critério) qualquer versão posterior.
#
# Este programa é distribuído na esperança de que seja útil,
# mas SEM QUALQUER GARANTIA; sem mesmo a garantia implícita de
# COMERCIALIZAÇÃO ou ADEQUAÇÃO A UM DETERMINADO FIM. Consulte a
# Licença Pública Geral Affero da GNU para mais detalhes.
#
# Você deve ter recebido uma cópia da Licença Pública Geral Affero da GNU
# junto com este programa. Se não, veja <https://www.gnu.org/licenses/>.
#
# AVISO DE PATENTE PENDENTE: O método e sistema ADUC implementado neste 
# software está em processo de patenteamento. Consulte NOTICE.md.

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

from ltx_manager_helpers import ltx_manager_singleton
from gemini_helpers import gemini_singleton 
from ltx_video.models.autoencoders.vae_encode import vae_encode, vae_decode

logger = logging.getLogger(__name__)

@dataclass
class LatentConditioningItem:
    """
    Representa uma âncora de condicionamento no espaço latente para a Câmera (Ψ).
    Cada item define um tensor, o frame exato onde sua influência é máxima,
    e a força dessa influência.
    """
    latent_tensor: torch.Tensor
    media_frame_number: int
    conditioning_strength: float

class Deformes4DEngine:
    """
    Implementa a Câmera (Ψ) e o Destilador (Δ) da arquitetura ADUC-SDR.
    Esta classe é o coração da execução, responsável pela geração de fragmentos de vídeo
    e pela extração e aplicação dos contextos causais (Eco e Déjà-Vu) que garantem
    a coerência de longa duração.
    """
    def __init__(self, ltx_manager, workspace_dir="deformes_workspace"):
        self.ltx_manager = ltx_manager
        self.workspace_dir = workspace_dir
        self._vae = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info("Especialista Deformes4D (Executor ADUC-SDR: Câmera Ψ e Destilador Δ) inicializado.")

    @property
    def vae(self):
        """Acessa o decodificador VAE de forma lazy, garantindo que ele esteja na GPU."""
        if self._vae is None:
            self._vae = self.ltx_manager.workers[0].pipeline.vae
        self._vae.to(self.device); self._vae.eval()
        return self._vae

    # MÉTODOS AUXILIARES DE MANIPULAÇÃO DE DADOS E VÍDEO
    
    def save_latent_tensor(self, tensor: torch.Tensor, path: str):
        """Salva um tensor PyTorch no disco."""
        torch.save(tensor.cpu(), path)

    def load_latent_tensor(self, path: str) -> torch.Tensor:
        """Carrega um tensor PyTorch do disco para o dispositivo correto."""
        return torch.load(path, map_location=self.device)

    @torch.no_grad()
    def pixels_to_latents(self, tensor: torch.Tensor) -> torch.Tensor:
        """Converte um tensor de pixels (vídeo) para o espaço latente usando o VAE."""
        tensor = tensor.to(self.device, dtype=self.vae.dtype)
        return vae_encode(tensor, self.vae, vae_per_channel_normalize=True)

    @torch.no_grad()
    def latents_to_pixels(self, latent_tensor: torch.Tensor, decode_timestep: float = 0.05) -> torch.Tensor:
        """Converte um tensor latente de volta para um tensor de pixels (vídeo) usando o VAE."""
        latent_tensor = latent_tensor.to(self.device, dtype=self.vae.dtype)
        timestep_tensor = torch.tensor([decode_timestep] * latent_tensor.shape[0], device=self.device, dtype=latent_tensor.dtype)
        return vae_decode(latent_tensor, self.vae, is_video=True, timestep=timestep_tensor, vae_per_channel_normalize=True)

    def save_video_from_tensor(self, video_tensor: torch.Tensor, path: str, fps: int = 24):
        """Salva um tensor de pixels como um arquivo de vídeo MP4."""
        if video_tensor is None or video_tensor.ndim != 5 or video_tensor.shape[2] == 0:
            logger.warning(f"Tentativa de salvar um tensor de vídeo inválido em {path}. Abortando.")
            return
        video_tensor = video_tensor.squeeze(0).permute(1, 2, 3, 0)
        video_tensor = (video_tensor.clamp(-1, 1) + 1) / 2.0
        video_np = (video_tensor.detach().cpu().float().numpy() * 255).astype(np.uint8)
        with imageio.get_writer(path, fps=fps, codec='libx264', quality=8) as writer:
            for frame in video_np: writer.append_data(frame)

    def _preprocess_image_for_latent_conversion(self, image: Image.Image, target_resolution: tuple) -> Image.Image:
        """Redimensiona uma imagem para a resolução alvo antes de convertê-la para latente."""
        if image.size != target_resolution:
            return ImageOps.fit(image, target_resolution, Image.Resampling.LANCZOS)
        return image

    def pil_to_latent(self, pil_image: Image.Image) -> torch.Tensor:
        """Converte uma imagem PIL para um tensor latente."""
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
        tensor = (tensor * 2.0) - 1.0
        return self.pixels_to_latents(tensor)
    
    def _get_video_frame_count(self, video_path: str) -> int | None:
        """Usa ffprobe para obter o número exato de frames de um arquivo de vídeo."""
        if not os.path.exists(video_path):
            logger.error(f"Arquivo de vídeo não encontrado para contagem de frames: {video_path}")
            return None
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
               '-show_entries', 'stream=nb_read_frames', '-of', 'default=nokey=1:noprint_wrappers=1', video_path]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
            return int(result.stdout.strip())
        except (subprocess.CalledProcessError, ValueError, FileNotFoundError) as e:
            logger.error(f"Erro ao contar frames com ffprobe para {video_path}: {e}")
            return None

    def _trim_last_frame_ffmpeg(self, input_path: str, output_path: str) -> bool:
        """
        Cria uma cópia de um vídeo, removendo o último frame.
        Esta etapa é CRUCIAL para resolver o problema do "frame n+1", onde o VAE
        gera um frame extra, o que causaria "soluços" na concatenação.
        """
        frame_count = self._get_video_frame_count(input_path)
        if frame_count is None or frame_count < 2:
            logger.warning(f"Não foi possível podar o último frame de {input_path}. O vídeo é muito curto ou ocorreu um erro.")
            if os.path.exists(input_path):
                os.rename(input_path, output_path)
            return True

        vf_filter = f"select='lt(n,{frame_count - 1})',setpts=PTS-STARTPTS"
        cmd_list = ['ffmpeg', '-y', '-i', input_path, '-vf', vf_filter, '-an', output_path]
        
        try:
            subprocess.run(cmd_list, check=True, capture_output=True, text=True, encoding='utf-8')
            logger.info(f"Último frame podado com sucesso. Vídeo final salvo em: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no FFmpeg durante a poda do último frame: {e.stderr}")
            return False

    def _generate_video_from_latents(self, latent_tensor, base_name: str) -> str:
        """
        Gera um vídeo a partir de latentes e aplica a poda do último frame.
        Este processo de duas etapas garante que os fragmentos para concatenação sejam perfeitamente limpos.
        """
        untrimmed_video_path = os.path.join(self.workspace_dir, f"{base_name}_untrimmed.mp4")
        trimmed_video_path = os.path.join(self.workspace_dir, f"{base_name}.mp4")

        logger.info(f"Renderizando vídeo bruto (com frame n+1) para: {untrimmed_video_path}")
        pixel_tensor = self.latents_to_pixels(latent_tensor)
        self.save_video_from_tensor(pixel_tensor, untrimmed_video_path, fps=24)
        del pixel_tensor
        gc.collect()
        torch.cuda.empty_cache()

        logger.info(f"Iniciando a poda do frame final de {untrimmed_video_path}...")
        success = self._trim_last_frame_ffmpeg(untrimmed_video_path, trimmed_video_path)
        
        if os.path.exists(untrimmed_video_path):
            os.remove(untrimmed_video_path)
        
        if not success:
            logger.error("Falha na poda do último frame. O fragmento pode conter um artefato.")
            return untrimmed_video_path 
            
        return trimmed_video_path

    def concatenate_videos_ffmpeg(self, video_paths: list[str], output_path: str) -> str:
        """Concatena uma lista de arquivos de vídeo em um único arquivo usando FFmpeg com o método 'concat'."""
        if not video_paths:
            raise gr.Error("Nenhum fragmento de vídeo para montar.")
        
        list_file_path = os.path.join(self.workspace_dir, "concat_list.txt")
        with open(list_file_path, 'w', encoding='utf-8') as f:
            for path in video_paths:
                f.write(f"file '{os.path.abspath(path)}'\n")
        
        cmd_list = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file_path, '-c', 'copy', output_path]
        logger.info("Executando concatenação FFmpeg para montagem final...")
        
        try:
            subprocess.run(cmd_list, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Erro no FFmpeg durante a concatenação: {e.stderr}")
            raise gr.Error(f"Falha na montagem final do vídeo. Detalhes: {e.stderr}")
        
        return output_path

    # NÚCLEO DA LÓGICA ADUC-SDR
    def generate_full_movie(self, keyframes: list, global_prompt: str, storyboard: list, 
                            seconds_per_fragment: float, trim_percent: int,
                            handler_strength: float, destination_convergence_strength: float, 
                            video_resolution: int, use_continuity_director: bool, 
                            progress: gr.Progress = gr.Progress()):
        """
        Orquestra a geração de um filme completo, fragmento por fragmento, seguindo os princípios da ADUC-SDR.
        
        O processo para cada fragmento é:
        1.  Consulta ao Maestro (Γ) para obter a intenção narrativa (motion_prompt).
        2.  Montagem das âncoras causais {C, D, K} para a Câmera (Ψ).
        3.  Execução da Geração Exploratória para criar o tensor bruto (V_bruto).
        4.  Execução do Ciclo de Poda Causal pelo Destilador (Δ) para extrair o Eco (C) e o Déjà-Vu (D) para o próximo ciclo
            e para definir o tensor de vídeo canônico (V_final).
        5.  Renderização do fragmento final e, opcionalmente, de um clipe de diagnóstico do Eco.
        6.  Repetição até que todos os keyframes sejam processados, seguida da montagem final.
        """
        
        # 1. Definição dos Parâmetros da Geração com base na Tese
        FPS = 24
        FRAMES_PER_LATENT_CHUNK = 8  # Fator de conversão: 1 índice na dimensão de tempo do tensor latente = 8 frames de vídeo.
        ECO_LATENT_CHUNKS = 2        # Número de chunks latentes que compõem o Eco Causal (C) para carregar a inércia.
        
        total_frames_brutos = self._quantize_to_multiple(int(seconds_per_fragment * FPS), FRAMES_PER_LATENT_CHUNK)
        total_latents_brutos = total_frames_brutos // FRAMES_PER_LATENT_CHUNK

        frames_a_podar = self._quantize_to_multiple(int(total_frames_brutos * (trim_percent / 100)), FRAMES_PER_LATENT_CHUNK)
        latents_a_podar = frames_a_podar // FRAMES_PER_LATENT_CHUNK

        if total_latents_brutos <= latents_a_podar:
            raise gr.Error(f"A porcentagem de poda ({trim_percent}%) é muito alta. Reduza-a ou aumente a duração.")

        DEJAVU_FRAME_TARGET = frames_a_podar - 1
        DESTINATION_FRAME_TARGET = total_frames_brutos - 1
        
        logger.info("--- CONFIGURAÇÃO DA GERAÇÃO ADUC-SDR ---")
        logger.info(f"Total de Latents por Geração Exploratória (V_bruto): {total_latents_brutos} ({total_frames_brutos} frames)")
        logger.info(f"Latents a serem descartados (Poda Causal): {latents_a_podar} ({frames_a_podar} frames)")
        logger.info(f"Chunks Latentes do Eco Causal (C): {ECO_LATENT_CHUNKS}")
        logger.info(f"Frame alvo do Déjà-Vu (D): {DEJAVU_FRAME_TARGET}")
        logger.info(f"Frame alvo do Destino (K): {DESTINATION_FRAME_TARGET}")
        logger.info("------------------------------------------")

        # 2. Inicialização do Estado
        base_ltx_params = {"guidance_scale": 2.0, "stg_scale": 0.025, "rescaling_scale": 0.15, "num_inference_steps": 20, "image_cond_noise_scale": 0.00}
        keyframe_paths = [item[0] if isinstance(item, tuple) else item for item in keyframes]
        video_clips_paths, story_history = [], ""
        target_resolution_tuple = (video_resolution, video_resolution) 
        
        eco_latent_for_next_loop = None
        dejavu_latent_for_next_loop = None
        
        if len(keyframe_paths) < 2:
            raise gr.Error(f"A geração requer no mínimo 2 keyframes. Você forneceu {len(keyframe_paths)}.")
        
        num_transitions_to_generate = len(keyframe_paths) - 1
        
        # 3. Loop Principal de Geração de Fragmentos
        for i in range(num_transitions_to_generate):
            fragment_index = i + 1
            logger.info(f"--- INICIANDO FRAGMENTO {fragment_index}/{num_transitions_to_generate} ---")
            progress(fragment_index / num_transitions_to_generate, desc=f"Produzindo Transição {fragment_index}/{num_transitions_to_generate}")
            
            # 3.1. Consulta ao Maestro (Γ) para obter a intenção (Pᵢ)
            past_keyframe_path = keyframe_paths[i - 1] if i > 0 else keyframe_paths[i]
            start_keyframe_path = keyframe_paths[i]
            destination_keyframe_path = keyframe_paths[i + 1]
            future_story_prompt = storyboard[i + 1] if (i + 1) < len(storyboard) else "A cena final."
            
            decision = gemini_singleton.get_cinematic_decision(
                global_prompt, story_history, past_keyframe_path, start_keyframe_path, destination_keyframe_path,
                storyboard[i - 1] if i > 0 else "O início.", storyboard[i], future_story_prompt
            )
            transition_type, motion_prompt = decision["transition_type"], decision["motion_prompt"]
            story_history += f"\n- Ato {fragment_index}: {motion_prompt}"

            # 3.2. Montagem das Âncoras para a Fórmula Canônica Ψ({C, D, K}, P)
            conditioning_items = []
            logger.info("  [Ψ.1] Montando âncoras causais...")
            
            if eco_latent_for_next_loop is None: # Lógica para o primeiro fragmento ou um corte ("cut")
               logger.info("    - Primeiro fragmento ou corte: Usando Keyframe inicial como âncora de partida.")
               img_start = self._preprocess_image_for_latent_conversion(Image.open(start_keyframe_path).convert("RGB"), target_resolution_tuple)
               conditioning_items.append(LatentConditioningItem(self.pil_to_latent(img_start), 0, 1.0))
            else: # Lógica para fragmentos contínuos
               logger.info("    - Âncora 1: Eco Causal (C) - Herança do passado.")
               conditioning_items.append(LatentConditioningItem(eco_latent_for_next_loop, 0, 1.0))
               logger.info("    - Âncora 2: Déjà-Vu (D) - Memória de um futuro idealizado.")
               conditioning_items.append(LatentConditioningItem(dejavu_latent_for_next_loop, DEJAVU_FRAME_TARGET, handler_strength))
            
            logger.info("    - Âncora 3: Destino (K) - Âncora geométrica/narrativa.")
            img_dest = self._preprocess_image_for_latent_conversion(Image.open(destination_keyframe_path).convert("RGB"), target_resolution_tuple)
            conditioning_items.append(LatentConditioningItem(self.pil_to_latent(img_dest), DESTINATION_FRAME_TARGET, destination_convergence_strength))

            # 3.3. Execução da Câmera (Ψ): Geração Exploratória para criar V_bruto
            logger.info(f"  [Ψ.2] Câmera (Ψ) executando a geração exploratória de {total_latents_brutos} chunks latentes...")
            current_ltx_params = {**base_ltx_params, "motion_prompt": motion_prompt}
            latents_brutos = self._generate_latent_tensor_internal(conditioning_items, current_ltx_params, target_resolution_tuple, total_frames_brutos)
            logger.info(f"    - Geração concluída. Tensor latente bruto (V_bruto) criado com shape: {latents_brutos.shape}.")

            # 3.4. Execução do Destilador (Δ): Implementação do Ciclo de Poda Causal com workaround empírico.
            # Esta lógica foi refinada para contornar um bug do motor de difusão que gera os 2 primeiros chunks com
            # artefatos, garantindo um resultado final limpo e mantendo a transferência causal.
            logger.info(f"  [Δ] Destilador (Δ) executando o Ciclo de Poda Causal...")
            
            # --- Início do Bloco de Lógica Crítica ---
            # ETAPA 1: Isolar a "cauda longa" de V_bruto. Esta fatia é 1 chunk maior que a porção
            # a ser podada, para nos permitir "recuperar" chunks para o Eco.
            last_trim = latents_brutos[:, :, -(latents_a_podar+1):, :, :].clone()
            
            # ETAPA 2: Extrair o Eco Causal (C). Ele é extraído dos 2 primeiros chunks da
            # cauda isolada. Isso cria uma sobreposição de memória, garantindo que a inércia
            # do final do clipe anterior seja perfeitamente transferida.
            eco_latent_for_next_loop = last_trim[:, :, :ECO_LATENT_CHUNKS, :, :].clone()   
            
            # ETAPA 3: Extrair o Déjà-Vu (D). É o último chunk absoluto da geração
            # exploratória, representando a memória do destino ideal alcançado.
            dejavu_latent_for_next_loop = last_trim[:, :, -1:, :, :].clone()
            
            # ETAPA 4: Definir o tensor de vídeo canônico (V_final). Primeiro, removemos uma cauda
            # que é 1 chunk MENOR que a poda planejada. Isso intencionalmente mantém os chunks
            # que serão usados pelo Eco DENTRO do vídeo final.
            latents_video = latents_brutos[:, :, :-(latents_a_podar-1), :, :].clone()
            
            # ETAPA 5: Saneamento. Removemos o primeiro chunk do vídeo, que empiricamente
            # contém artefatos de inicialização da difusão.
            latents_video = latents_video[:, :, 1:, :, :]
            # --- Fim do Bloco de Lógica Crítica ---
            

            logger.info(f"  [Δ] Shape do tensor para vídeo final: {latents_video.shape}")
            logger.info(f"    - (Δ.1) Déjà-Vu (D) destilado. Shape: {dejavu_latent_for_next_loop.shape}")
            logger.info(f"    - (Δ.2) Eco Causal (C) extraído. Shape: {eco_latent_for_next_loop.shape}")

            # Se o Maestro decidiu por um "corte", a memória causal é resetada para o próximo ciclo.
            if transition_type == "cut":
                logger.warning("  - DECISÃO DO MAESTRO: Corte ('cut'). Resetando a memória causal (Eco e Déjà-Vu).")
                eco_latent_for_next_loop = None
                dejavu_latent_for_next_loop = None

            # 3.5. Renderização e Armazenamento do Fragmento Final
            base_name = f"fragment_{fragment_index}_{int(time.time())}"
            video_path = self._generate_video_from_latents(latents_video, base_name)
            video_clips_paths.append(video_path)
            logger.info(f"--- FRAGMENTO {fragment_index} FINALIZADO E SALVO EM: {video_path} ---")

            # Bloco de Diagnóstico: Gera um vídeo a partir do tensor do Eco para validação visual.
            if eco_latent_for_next_loop is not None:
                logger.info("--- GERANDO VÍDEO DE DIAGNÓSTICO DO ECO CAUSAL ---")
                eco_base_name = f"fragment_{fragment_index}_eco_diagnostic_{int(time.time())}"
                eco_video_path = self._generate_video_from_latents(eco_latent_for_next_loop, eco_base_name)
                video_clips_paths.append(eco_video_path)
                logger.info(f"Vídeo de diagnóstico do Eco salvo em: {eco_video_path} e adicionado à concatenação.")

            yield {"fragment_path": video_path}
                 
        # 4. Montagem Final do Filme
        final_movie_path = os.path.join(self.workspace_dir, f"final_movie_silent_{int(time.time())}.mp4")
        self.concatenate_videos_ffmpeg(video_clips_paths, final_movie_path)
        
        logger.info(f"Filme completo (com clipes de diagnóstico) salvo em: {final_movie_path}")
        yield {"final_path": final_movie_path}

    def _generate_latent_tensor_internal(self, conditioning_items, ltx_params, target_resolution, total_frames_to_generate):
        """Função de baixo nível que invoca o motor de difusão para a geração do tensor latente."""
        final_ltx_params = {
            **ltx_params, 'width': target_resolution[0], 'height': target_resolution[1],
            'video_total_frames': total_frames_to_generate, 'video_fps': 24,
            'current_fragment_index': int(time.time()), 'conditioning_items_data': conditioning_items
        }
        new_full_latents, _ = self.ltx_manager.generate_latent_fragment(**final_ltx_params)
        gc.collect()
        torch.cuda.empty_cache()
        return new_full_latents

    def _quantize_to_multiple(self, n, m):
        """Garante que um número 'n' seja um múltiplo de 'm', necessário para o fatiamento de tensores."""
        if m == 0: return n
        quantized = int(round(n / m) * m)
        return m if n > 0 and quantized == 0 else quantized