# Euia-AducSdr: Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
# Rua Eduardo Carlos Pereira, 4125, B1 Ap32, Curitiba, PR, Brazil, CEP 8102025
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
# Hugging Face: https://huggingface.co/spaces/Carlexx/Ltx-SuperTime-60Secondos/
# Hugging Face: https://huggingface.co/spaces/Carlexxx/Novinho/
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

# --- app.py (NOVINHO-4.2: O Piloto de Testes - Correção Final e Documentação de Bordo) ---

# --- Ato 1: A Convocação da Orquestra (Importações) ---
import gradio as gr
import torch
import os
import yaml
from PIL import Image, ImageOps
import shutil
import gc
import subprocess
import google.generativeai as genai
import numpy as np
import imageio
from pathlib import Path
import huggingface_hub
import json
import time

from inference import create_ltx_video_pipeline, load_image_to_tensor_with_resize_and_crop, ConditioningItem, calculate_padding
from dreamo_helpers import dreamo_generator_singleton

# --- Ato 2: A Preparação do Palco (Configurações) ---
config_file_path = "configs/ltxv-13b-0.9.8-distilled.yaml"
with open(config_file_path, "r") as file: PIPELINE_CONFIG_YAML = yaml.safe_load(file)

LTX_REPO = "Lightricks/LTX-Video"
models_dir = "downloaded_models_gradio_cpu_init"
Path(models_dir).mkdir(parents=True, exist_ok=True)
WORKSPACE_DIR = "aduc_workspace"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

VIDEO_FPS = 36
VIDEO_DURATION_SECONDS = 4
VIDEO_TOTAL_FRAMES = VIDEO_DURATION_SECONDS * VIDEO_FPS
CONVERGENCE_FRAMES = 8
TARGET_RESOLUTION = 720
MAX_REFS = 4

print("Baixando e criando pipelines LTX na CPU...")
distilled_model_actual_path = huggingface_hub.hf_hub_download(repo_id=LTX_REPO, filename=PIPELINE_CONFIG_YAML["checkpoint_path"], local_dir=models_dir, local_dir_use_symlinks=False)
pipeline_instance = create_ltx_video_pipeline(
    ckpt_path=distilled_model_actual_path,
    precision=PIPELINE_CONFIG_YAML["precision"],
    text_encoder_model_name_or_path=PIPELINE_CONFIG_YAML["text_encoder_model_name_or_path"],
    sampler=PIPELINE_CONFIG_YAML["sampler"],
    device='cpu'
)
print("Modelos LTX prontos (na CPU).")


# --- Ato 3: As Partituras dos Músicos (Funções Corrigidas e Documentadas) ---

####
# Carrega uma mídia (imagem ou vídeo) e a converte em um tensor Pytorch.
# Esta função prepara as imagens de condicionamento (início e fim) para o pipeline LTX.
# Se a entrada for um vídeo, extrai e usa apenas o seu primeiro frame.
####
def load_conditioning_tensor(media_path: str, height: int, width: int) -> torch.Tensor:
    if not media_path: raise ValueError("Caminho da mídia de condicionamento não pode ser nulo.")
    lower_path = media_path.lower()
    if lower_path.endswith(('.png', '.jpg', '.jpeg')):
        return load_image_to_tensor_with_resize_and_crop(media_path, height, width)
    elif lower_path.endswith('.mp4'):
        try:
            with imageio.get_reader(media_path) as reader:
                first_frame = reader.get_data(0)
            image = Image.fromarray(first_frame).convert("RGB")
            return load_image_to_tensor_with_resize_and_crop(image, height, width)
        except Exception as e:
            raise gr.Error(f"Falha ao ler o primeiro frame do vídeo '{media_path}': {e}")
    else:
        raise gr.Error(f"Formato de arquivo de condicionamento não suportado: {media_path}")

####
# Executa o pipeline LTX para gerar um único fragmento de vídeo.
# Atua como a "Câmera" do sistema. Recebe um ponto de partida e um ponto de
# chegada (em `conditioning_items_data`), uma instrução de movimento (`motion_prompt`),
# e renderiza o clipe de vídeo correspondente. Gerencia o uso da GPU.
####
def run_ltx_animation(current_fragment_index, motion_prompt, conditioning_items_data, width, height, seed, cfg, progress=gr.Progress()):
    progress(0, desc=f"[TECPIX 5000] Filmando Cena {current_fragment_index}...");
    output_path = os.path.join(WORKSPACE_DIR, f"fragment_{current_fragment_index}.mp4"); target_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        pipeline_instance.to(target_device)
        conditioning_items = []
        for (path, start_frame, strength) in conditioning_items_data:
            tensor = load_conditioning_tensor(path, height, width)
            conditioning_items.append(ConditioningItem(tensor.to(target_device), start_frame, strength))
        
        n_val = round((float(VIDEO_TOTAL_FRAMES) - 1.0) / 8.0); actual_num_frames = int(n_val * 8 + 1)
        padded_h, padded_w = ((height - 1) // 32 + 1) * 32, ((width - 1) // 32 + 1) * 32
        padding_vals = calculate_padding(height, width, padded_h, padded_w)
        for cond_item in conditioning_items: cond_item.media_item = torch.nn.functional.pad(cond_item.media_item, padding_vals)
        kwargs = {"prompt": motion_prompt, "negative_prompt": "blurry, distorted, bad quality, artifacts", "height": padded_h, "width": padded_w, "num_frames": actual_num_frames, "frame_rate": VIDEO_FPS, "generator": torch.Generator(device=target_device).manual_seed(int(seed) + current_fragment_index), "output_type": "pt", "guidance_scale": float(cfg), "timesteps": PIPELINE_CONFIG_YAML.get("first_pass", {}).get("timesteps"), "conditioning_items": conditioning_items, "decode_timestep": PIPELINE_CONFIG_YAML.get("decode_timestep"), "decode_noise_scale": PIPELINE_CONFIG_YAML.get("decode_noise_scale"), "stochastic_sampling": PIPELINE_CONFIG_YAML.get("stochastic_sampling"), "image_cond_noise_scale": 0.15, "is_video": True, "vae_per_channel_normalize": True, "mixed_precision": (PIPELINE_CONFIG_YAML.get("precision") == "mixed_precision"), "offload_to_cpu": False, "enhance_prompt": False}
        result_tensor = pipeline_instance(**kwargs).images
        pad_l, pad_r, pad_t, pad_b = map(int, padding_vals); slice_h = -pad_b if pad_b > 0 else None; slice_w = -pad_r if pad_r > 0 else None
        cropped_tensor = result_tensor[:, :, :VIDEO_TOTAL_FRAMES, pad_t:slice_h, pad_l:slice_w]; video_np = (cropped_tensor[0].permute(1, 2, 3, 0).cpu().float().numpy() * 255).astype(np.uint8)
        with imageio.get_writer(output_path, fps=VIDEO_FPS, codec='libx264', quality=8) as writer:
            for i, frame in enumerate(video_np): progress(i / len(video_np), desc=f"Renderizando frame {i+1}/{len(video_np)}..."); writer.append_data(frame)
        return output_path
    finally:
        pipeline_instance.to('cpu'); gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

####
# Processa a imagem de referência inicial do usuário, tornando-a quadrada.
# Garante que a imagem de referência principal tenha as dimensões corretas
# (ex: 720x720) antes de ser usada no pipeline, evitando distorções.
####
def process_image_to_square(image_path: str, size: int = TARGET_RESOLUTION) -> str:
    if not image_path or not os.path.exists(image_path): return None
    try:
        img = Image.open(image_path).convert("RGB")
        img_square = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        output_filename = f"initial_ref_{size}x{size}.png"
        output_path = os.path.join(WORKSPACE_DIR, output_filename)
        img_square.save(output_path)
        return output_path
    except Exception as e: raise gr.Error(f"Falha ao processar a imagem de referência: {e}")

####
# Gera o roteiro de cenas estáticas (storyboard) usando Gemini.
# Atua como o "Sonhador" ou "Fotógrafo". Analisa a ideia geral do usuário
# e a imagem de referência para criar uma sequência de descrições de cenas.
####
def get_static_scenes_storyboard(num_fragments: int, prompt: str, initial_image_path: str):
    if not initial_image_path: raise gr.Error("Por favor, forneça uma imagem de referência inicial.")
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    genai.configure(api_key=GEMINI_API_KEY)
    prompt_file = "prompts/photographer_prompt.txt"
    with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
    director_prompt = template.format(user_prompt=prompt, num_fragments=int(num_fragments))
    model = genai.GenerativeModel('gemini-1.5-flash'); img = Image.open(initial_image_path)
    response = model.generate_content([director_prompt, img])
    try:
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "")
        storyboard_data = json.loads(cleaned_response)
        return storyboard_data.get("scene_storyboard", [])
    except Exception as e: raise gr.Error(f"O Sonhador (Gemini) falhou ao criar o roteiro: {e}. Resposta: {response.text}")

####
# Gera todas as imagens-chave (keyframes) para a história usando DreamO.
# Atua como o "Pintor". Itera sobre o roteiro e pinta uma imagem para cada cena.
# Opera de forma sequencial, usando o keyframe anterior como referência para o próximo.
####
def run_keyframe_generation(storyboard, initial_ref_image_path, *reference_args):
    if not storyboard:
        raise gr.Error("Nenhum roteiro para gerar imagens-chave.")
    if not initial_ref_image_path or not os.path.exists(initial_ref_image_path):
        raise gr.Error("A imagem de referência principal é obrigatória para iniciar a pintura.")

    num_total_refs = MAX_REFS + 1
    ref_paths = list(reference_args[:num_total_refs])
    ref_tasks = list(reference_args[num_total_refs:])
    
    with Image.open(initial_ref_image_path) as img:
        width, height = img.size
        width, height = (width // 32) * 32, (height // 32) * 32

    keyframe_paths = []
    log_history = ""

    try:
        dreamo_generator_singleton.to_gpu()

        log_history += f"Pintando Keyframe Inicial (Cena 1/{len(storyboard)})...\n"
        yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=keyframe_paths)}
        
        references_for_first_frame = []
        references_for_first_frame.append({'image_np': np.array(Image.open(initial_ref_image_path).convert("RGB")), 'task': 'ip'})
        log_history += f"  - Usando imagem de referência principal '{os.path.basename(initial_ref_image_path)}' (Tarefa: ip)\n"

        for j in range(1, num_total_refs):
            aux_path, aux_task = ref_paths[j], ref_tasks[j]
            if aux_path and os.path.exists(aux_path):
                references_for_first_frame.append({'image_np': np.array(Image.open(aux_path).convert("RGB")), 'task': aux_task})
                log_history += f"  - Usando ref. auxiliar: {os.path.basename(aux_path)} (Tarefa: {aux_task})\n"
        
        first_prompt = storyboard[0]
        output_path = os.path.join(WORKSPACE_DIR, "keyframe_1.png")
        image = dreamo_generator_singleton.generate_image_with_gpu_management(
            reference_items=references_for_first_frame, prompt=first_prompt, width=width, height=height
        )
        image.save(output_path)
        keyframe_paths.append(output_path)
        current_ref_image_path = output_path
        
        for i, prompt in enumerate(storyboard[1:], start=1):
            log_history += f"\nPintando Cena Sequencial {i+1}/{len(storyboard)}...\n"
            yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=keyframe_paths)}
            
            reference_items_for_dreamo = []
            sequential_ref_task = ref_tasks[0]
            reference_items_for_dreamo.append({'image_np': np.array(Image.open(current_ref_image_path).convert("RGB")), 'task': sequential_ref_task})
            log_history += f"  - Usando ref. sequencial: {os.path.basename(current_ref_image_path)} (Tarefa: {sequential_ref_task})\n"

            for j in range(1, num_total_refs):
                aux_path, aux_task = ref_paths[j], ref_tasks[j]
                if aux_path and os.path.exists(aux_path):
                    reference_items_for_dreamo.append({'image_np': np.array(Image.open(aux_path).convert("RGB")), 'task': aux_task})
                    log_history += f"  - Usando ref. auxiliar: {os.path.basename(aux_path)} (Tarefa: {aux_task})\n"

            output_path = os.path.join(WORKSPACE_DIR, f"keyframe_{i+1}.png")
            image = dreamo_generator_singleton.generate_image_with_gpu_management(
                reference_items=reference_items_for_dreamo, prompt=prompt, width=width, height=height
            )
            image.save(output_path)
            keyframe_paths.append(output_path)
            current_ref_image_path = output_path

    except Exception as e:
        raise gr.Error(f"O Pintor (DreamO) encontrou um erro: {e}")
    finally:
        dreamo_generator_singleton.to_cpu()
        
    log_history += "\nPintura de todos os keyframes concluída.\n"
    yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=keyframe_paths), keyframe_images_state: keyframe_paths}

####
# Gera um único prompt de movimento para uma transição, usando Gemini.
# Atua como a "consciência" do Cineasta. Analisa uma mídia de partida e uma
# imagem de chegada para descrever como a câmera deve se mover.
# Usa uma lógica bifocal e espera ativa para robustez.
####
def get_single_motion_prompt(user_prompt: str, story_history: str, start_media_path: str, end_keyframe_path: str, prompt_filename: str):
    if not GEMINI_API_KEY:
        raise gr.Error("Chave da API Gemini não configurada!")

    uploaded_file = None
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')

        print(f"Cineasta: Fazendo upload do arquivo de contexto '{start_media_path}'...")
        file_to_upload = genai.upload_file(start_media_path)
        
        print(f"Cineasta: Aguardando arquivo '{file_to_upload.name}' ficar ATIVO...")
        timeout_seconds = 180
        start_time = time.time()

        while file_to_upload.state.name == "PROCESSING":
            if time.time() - start_time > timeout_seconds:
                genai.delete_file(name=file_to_upload.name)
                raise TimeoutError(f"Tempo de espera para o processamento do arquivo '{file_to_upload.name}' excedido.")
            
            time.sleep(5)
            file_to_upload = genai.get_file(name=file_to_upload.name)
        
        if file_to_upload.state.name != "ACTIVE":
            raise gr.Error(f"O arquivo de mídia '{file_to_upload.name}' não pôde ser processado. Estado final: {file_to_upload.state.name}")
        
        print(f"Cineasta: Arquivo '{file_to_upload.name}' está ATIVO. Gerando prompt...")
        uploaded_file = file_to_upload
        
        end_media = Image.open(end_keyframe_path)
        
        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompts", prompt_filename)
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            template = f.read()
        
        director_prompt = template.format(user_prompt=user_prompt, story_history=story_history)
        
        model_contents = [director_prompt, uploaded_file, end_media]
        response = model.generate_content(model_contents)
        
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[len("```json"):].strip()
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-len("```")].strip()
        
        try:
            motion_data = json.loads(cleaned_text)
            final_prompt = motion_data.get("motion_prompt", "")
            if not final_prompt:
                raise ValueError("Prompt de movimento vazio no JSON.")
            return final_prompt
        except (json.JSONDecodeError, ValueError):
            return cleaned_text.replace("\"", "").replace("{", "").replace("}", "").replace("motion_prompt:", "").strip()

    except Exception as e:
        response_text = getattr(e, 'text', 'Nenhuma resposta de texto disponível.')
        raise gr.Error(f"O Cineasta (Gemini) falhou ao criar o prompt de movimento: {e}. Resposta: {response_text}")
    
    finally:
        if uploaded_file:
            try:
                genai.delete_file(uploaded_file.name)
            except Exception as delete_e:
                print(f"Aviso: Falha ao deletar o arquivo temporário {uploaded_file.name} da API Gemini. Erro: {delete_e}")

####
# Orquestra a produção de todos os fragmentos de vídeo.
# Implementa a lógica central da arquitetura ADUC-SDR para a geração de vídeo:
# 1. Fragmento 1: Gerado a partir da transição do Keyframe 1 para o Keyframe 2.
# 2. Fragmentos Subsequentes: Gerados a partir do "eco" do vídeo anterior para o
#    próximo keyframe na sequência.
####
def run_video_production(prompt_geral, keyframe_image_paths, scene_storyboard, seed, cfg, progress=gr.Progress()):
    if not keyframe_image_paths or len(keyframe_image_paths) < 2:
        raise gr.Error("Pinte pelo menos 2 keyframes na Etapa 2 para produzir as transições.")

    log_history = "\n--- FASE 3/4: A Câmera e o Cineasta estão filmando em sequência just-in-time...\n"
    yield {production_log_output: log_history, video_gallery_glitch: []}
    
    video_fragments = []
    
    previous_media_path = keyframe_image_paths[0]
    
    story_history = ""
    with Image.open(keyframe_image_paths[0]) as img:
        width, height = img.size

    num_transitions = len(keyframe_image_paths) - 1
    for i in range(num_transitions):
        start_media_path = previous_media_path
        end_keyframe_path = keyframe_image_paths[i+1]
        
        is_first_fragment = (i == 0)
        
        fragment_num = i + 1
        progress(i / num_transitions, desc=f"Planejando e Filmando Fragmento {fragment_num}/{num_transitions}")
        
        log_history += f"\n--- FRAGMENTO {fragment_num} ---\n"
        log_history += "Cineasta (Gemini) está analisando a cena anterior e a próxima...\n"
        yield {production_log_output: log_history}
        
        if is_first_fragment:
            prompt_filename_to_use = "director_motion_prompt.txt"
            story_history = f"A história começa com a transição da cena '{scene_storyboard[0]}' para '{scene_storyboard[1]}'."
        else:
            prompt_filename_to_use = "director_motion_prompt_transition.txt"
            story_history += f"\n- Em seguida, a cena muda de '{scene_storyboard[i]}' para '{scene_storyboard[i+1]}'."

        current_motion_prompt = get_single_motion_prompt(prompt_geral, story_history, start_media_path, end_keyframe_path, prompt_filename_to_use)

        log_history += f"Instrução do Cineasta ({prompt_filename_to_use}): '{current_motion_prompt}'\n"
        log_history += f"Filmando transição de '{os.path.basename(start_media_path)}' para '{os.path.basename(end_keyframe_path)}'...\n"
        yield {production_log_output: log_history}

        end_frame_index = VIDEO_TOTAL_FRAMES - CONVERGENCE_FRAMES
        conditioning_items_data = [(start_media_path, 0, 1.0), (end_keyframe_path, end_frame_index, 1.0)]
        
        fragment_path = run_ltx_animation(fragment_num, current_motion_prompt, conditioning_items_data, width, height, seed, cfg, progress)
        video_fragments.append(fragment_path)
        
        log_history += f"Fragmento {fragment_num} filmado. Extraindo memória física para a próxima cena...\n"
        yield {production_log_output: log_history, video_gallery_glitch: video_fragments}
        
        previous_media_path = extract_final_frames_video(fragment_path, fragment_num, CONVERGENCE_FRAMES)
        
    log_history += "\nFilmagem de todos os fragmentos de transição concluída.\n"
    progress(1.0, desc="Produção Concluída.")
    yield {production_log_output: log_history, video_gallery_glitch: video_fragments, fragment_list_state: video_fragments}

####
# Extrai os últimos N frames de um vídeo para criar o "clipe de convergência" ou "eco".
# Esta é a etapa de "Destilação" da arquitetura ADUC-SDR. O clipe gerado
# serve como o ponto de partida (Contexto Causal) para a próxima animação.
# O comando FFmpeg é robustecido para garantir compatibilidade com a API do Google.
####
def extract_final_frames_video(input_video_path: str, fragment_index: int, num_frames: int):
    output_video_path = os.path.join(WORKSPACE_DIR, f"convergence_clip_{fragment_index}.mp4")
    if not os.path.exists(input_video_path): raise gr.Error(f"Erro Interno: Vídeo de entrada para extração não encontrado: {input_video_path}")
    try:
        command_probe = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 \"{input_video_path}\""
        result_probe = subprocess.run(command_probe, shell=True, check=True, capture_output=True, text=True); total_frames = int(result_probe.stdout.strip())
        start_frame_index = total_frames - num_frames
        if start_frame_index < 0:
            shutil.copyfile(input_video_path, output_video_path); return output_video_path
        
        command_extract = (
            f"ffmpeg -y -v error -i \"{input_video_path}\" "
            f"-vf \"select='gte(n,{start_frame_index})'\" "
            f"-c:v libx264 -profile:v baseline -level 3.0 -pix_fmt yuv420p "
            f"-an \"{output_video_path}\""
        )
        
        subprocess.run(command_extract, shell=True, check=True, capture_output=True, text=True); return output_video_path
    except (subprocess.CalledProcessError, ValueError) as e:
        error_message = f"Editor Mágico (FFmpeg) falhou ao extrair o clipe de convergência: {e}"
        if hasattr(e, 'stderr'): error_message += f"\nDetalhes: {e.stderr}"
        raise gr.Error(error_message)

####
# Realiza a pós-produção, unindo todos os fragmentos de vídeo em uma obra final.
# Atua como o "Editor". Remove a sobreposição de "eco" entre os clipes e os concatena.
####
def concatenate_and_trim_masterpiece(fragment_paths: list, progress=gr.Progress()):
    if not fragment_paths: raise gr.Error("Nenhum fragmento de vídeo para concatenar.")
    progress(0.2, desc="Aparando fragmentos para transições suaves...");
    trimmed_dir = os.path.join(WORKSPACE_DIR, "trimmed"); os.makedirs(trimmed_dir, exist_ok=True)
    paths_for_concat = []
    try:
        for i, path in enumerate(fragment_paths):
            if i == len(fragment_paths) - 1:
                paths_for_concat.append(path)
                continue

            trimmed_path = os.path.join(trimmed_dir, f"fragment_{i}_trimmed.mp4")
            probe_cmd = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=noprint_wrappers=1:nokey=1 \"{path}\""
            result = subprocess.run(probe_cmd, shell=True, check=True, capture_output=True, text=True)
            total_frames = int(result.stdout.strip())
            frames_to_keep = total_frames - CONVERGENCE_FRAMES
            if frames_to_keep <= 0:
                shutil.copyfile(path, trimmed_path)
                paths_for_concat.append(trimmed_path)
                continue
            
            trim_cmd = f"ffmpeg -y -v error -i \"{path}\" -vf \"select='lt(n,{frames_to_keep})'\" -c:v libx264 -preset ultrafast -an \"{trimmed_path}\""
            subprocess.run(trim_cmd, shell=True, check=True, capture_output=True, text=True)
            paths_for_concat.append(trimmed_path)

        progress(0.6, desc="Montando a obra-prima final...")
        list_file_path = os.path.join(WORKSPACE_DIR, "concat_list.txt"); final_output_path = os.path.join(WORKSPACE_DIR, "obra_prima_final.mp4")
        with open(list_file_path, "w") as f:
            for p in paths_for_concat: f.write(f"file '{os.path.abspath(p)}'\n")
        concat_cmd = f"ffmpeg -y -v error -f concat -safe 0 -i \"{list_file_path}\" -c copy \"{final_output_path}\""
        subprocess.run(concat_cmd, shell=True, check=True, capture_output=True, text=True)
        return final_output_path
    except (subprocess.CalledProcessError, ValueError) as e:
        error_message = f"FFmpeg falhou durante a pós-produção (corte e concatenação): {e}"
        if hasattr(e, 'stderr'): error_message += f"\nDetalhes do erro do FFmpeg: {e.stderr}"
        raise gr.Error(error_message)

# --- Ato 5: A Interface com o Mundo (A UI Restaurada e Aprimorada) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# NOVINHO-4.2 (Piloto de Testes - Arquitetura Estabilizada)\n*By Carlex & Gemini & DreamO*")
    
    if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR)
    Path("examples").mkdir(exist_ok=True)
    
    scene_storyboard_state = gr.State([])
    keyframe_images_state = gr.State([])
    fragment_list_state = gr.State([])
    prompt_geral_state = gr.State("")
    processed_ref_path_state = gr.State("")
    visible_references_state = gr.State(0)

    # --- ETAPA 1: O ROTEIRO ---
    gr.Markdown("--- \n ## ETAPA 1: O ROTEIRO (Sonhador)")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Ideia Geral (Prompt)")
            num_fragments_input = gr.Slider(2, 10, 4, step=1, label="Número de Cenas")
            image_input = gr.Image(type="filepath", label=f"Imagem de Referência Principal (será {TARGET_RESOLUTION}x{TARGET_RESOLUTION})")
            director_button = gr.Button("▶️ 1. Gerar Roteiro de Cenas", variant="primary")
        with gr.Column(scale=2):
            storyboard_to_show = gr.JSON(label="Roteiro de Cenas Gerado")
            
    # --- ETAPA 2: OS KEYFRAMES ---
    gr.Markdown("--- \n ## ETAPA 2: OS KEYFRAMES (Pintor)")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Controles do Pintor (DreamO)")
            gr.Markdown("**Tarefas:** `style` (estilo), `ip` (conteúdo), `id` (identidade).")
            ref_image_inputs, ref_task_inputs, aux_ref_rows = [], [], []
            with gr.Group():
                with gr.Row():
                    ref_image_inputs.append(gr.Image(label="Referência Sequencial (Automática)", type="filepath", interactive=False))
                    ref_task_inputs.append(gr.Dropdown(choices=["ip", "id", "style"], value="style", label="Tarefa Seq."))
            for i in range(MAX_REFS):
                with gr.Row(visible=False) as ref_row_aux:
                    ref_image_inputs.append(gr.Image(label=f"Ref. Auxiliar {i+1}", type="filepath"))
                    ref_task_inputs.append(gr.Dropdown(choices=["ip", "id", "style"], value="ip", label=f"Tarefa Aux. {i+1}"))
                aux_ref_rows.append(ref_row_aux)
            with gr.Row():
                add_ref_button = gr.Button("➕ Add Ref.")
                remove_ref_button = gr.Button("➖ Rem. Ref.")
            photographer_button = gr.Button("▶️ 2. Pintar Imagens-Chave", variant="primary")
        with gr.Column(scale=1):
            keyframe_log_output = gr.Textbox(label="Diário de Bordo do Pintor", lines=15, interactive=False)
            keyframe_gallery_output = gr.Gallery(label="Imagens-Chave Pintadas", object_fit="contain", height="auto", type="filepath")

    # --- ETAPA 3: A PRODUÇÃO ---
    gr.Markdown("--- \n ## ETAPA 3: A PRODUÇÃO (Cineasta e Câmera)")
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                seed_number = gr.Number(42, label="Seed")
                cfg_slider = gr.Slider(1.0, 10.0, 2.5, step=0.1, label="CFG")
            animator_button = gr.Button("▶️ 3. Produzir Cenas em Vídeo", variant="primary")
            production_log_output = gr.Textbox(label="Diário de Bordo da Produção", lines=10, interactive=False)
        with gr.Column(scale=1):
            video_gallery_glitch = gr.Gallery(label="Fragmentos Gerados (com sobreposição)", object_fit="contain", height="auto", type="video")
    
    # --- ETAPA 4: PÓS-PRODUÇÃO ---
    gr.Markdown(f"--- \n ## ETAPA 4: PÓS-PRODUÇÃO (Editor)")
    editor_button = gr.Button("▶️ 4. Montar Vídeo Final", variant="primary")
    final_video_output = gr.Video(label="A Obra-Prima Final", width=TARGET_RESOLUTION)

    # --- Rodapé Filosófico ---
    gr.Markdown(
        """
        ---
        ### A Arquitetura ADUC-SDR: O Esquema Matemático
        A geração de vídeo é governada por uma função seccional que define como cada fragmento (`V_i`) é criado. A fórmula muda dependendo se estamos no **"Gênesis"** da história ou na **"Cadeia Causal"** que se segue.

        ---
        #### **FÓRMULA 1: O FRAGMENTO INICIAL (Gênesis, `i=1`)**
        *Define a criação do primeiro clipe a partir de imagens estáticas.*

        **Ato 1: Planejamento Inicial (`P_1`)**
        `P_1 = Γ_initial( K_1, K_2, P_geral )`

        **Ato 2: Execução Inicial (`V_1`)**
        `V_1 = Ψ( { (K_1, F_start), (K_2, F_end) }, P_1 )`

        ---
        #### **FÓRMULA 2: A CADEIA CAUSAL (Momentum, `i > 1`)**
        *Define a criação dos fragmentos subsequentes, garantindo a continuidade através do "eco".*

        **Ato 0: Destilação do Eco (`C_(i-1)`)**
        `C_(i-1) = Δ(V_(i-1))`

        **Ato 1: Planejamento de Continuidade (`P_i`)**
        `P_i = Γ_transition( C_(i-1), K_(i+1), P_geral, H_(i-1) )`

        **Ato 2: Execução Causal (`V_i`)**
        `V_i = Ψ( { (C_(i-1), F_start), (K_(i+1), F_end) }, P_i )`

        ---
        #### **Componentes (O Léxico da Arquitetura):**
        - **`V_i`**: Fragmento de Vídeo
        - **`K_i`**: Keyframe (Imagem Estática)
        - **`C_i`**: "Eco" Causal (Clipe de Vídeo)
        - **`P_i`**: Prompt de Movimento
        - **`P_geral`**: Prompt Geral (Intenção do Diretor)
        - **`H_i`**: Histórico Narrativo
        - **`Γ`**: Cineasta (Gerador de Prompt)
        - **`Ψ`**: Câmera (Gerador de Vídeo)
        - **`Δ`**: Editor (Extrator de "Eco")
        - **`F_start`, `F_end`**: Constantes de Frame (Âncoras Temporais)
        """
    )


    # --- Ato 6: A Regência (Lógica de Conexão dos Botões) ---
    def update_reference_visibility(current_count, action):
        if action == "add": new_count = min(MAX_REFS, current_count + 1)
        else: new_count = max(0, current_count - 1)
        return [new_count] + [gr.update(visible=(i < new_count)) for i in range(MAX_REFS)]

    add_ref_button.click(fn=update_reference_visibility, inputs=[visible_references_state, gr.State("add")], outputs=[visible_references_state] + aux_ref_rows)
    remove_ref_button.click(fn=update_reference_visibility, inputs=[visible_references_state, gr.State("remove")], outputs=[visible_references_state] + aux_ref_rows)

    director_button.click(
        fn=get_static_scenes_storyboard, 
        inputs=[num_fragments_input, prompt_input, image_input], 
        outputs=[scene_storyboard_state]
    ).success(
        fn=lambda s, p: (s, p), 
        inputs=[scene_storyboard_state, prompt_input], 
        outputs=[storyboard_to_show, prompt_geral_state]
    ).success(
        fn=process_image_to_square, 
        inputs=[image_input], 
        outputs=[processed_ref_path_state]
    ).success(
        fn=lambda p: p, 
        inputs=[processed_ref_path_state], 
        outputs=[ref_image_inputs[0]]
    )
    
    photographer_button.click(
        fn=run_keyframe_generation, 
        inputs=[scene_storyboard_state, processed_ref_path_state] + ref_image_inputs + ref_task_inputs, 
        outputs=[keyframe_log_output, keyframe_gallery_output, keyframe_images_state]
    )
    
    animator_button.click(
        fn=run_video_production,
        inputs=[prompt_geral_state, keyframe_images_state, scene_storyboard_state, seed_number, cfg_slider],
        outputs=[production_log_output, video_gallery_glitch, fragment_list_state]
    )
    
    editor_button.click(
        fn=concatenate_and_trim_masterpiece, 
        inputs=[fragment_list_state], 
        outputs=[final_video_output]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", share=True)