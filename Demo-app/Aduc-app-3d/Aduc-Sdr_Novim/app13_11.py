# Euia-AducSdr: Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
#
# Contato:
# Carlos Rodrigues dos Santos
# carlex22@gmail.com
#
# Repositórios e Projetos Relacionados:
# GitHub: https://github.com/carlex22/Aduc-sdr
# YouTube (Resultados): https://m.youtube.com/channel/UC3EgoJi_Fv7yuDpvfYNtoIQ
# Hugging Face: https://huggingface.co/spaces/Carlexx/ADUC-Sdr_Gemini_Drem0_Ltx_Video60seconds/
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

# --- app.py (ADUC-SDR-4.7: Fluxo Automatizado e Síncrono) ---

import gradio as gr
import torch
import os
import re
import yaml
from PIL import Image, ImageOps, ExifTags
import shutil
import subprocess
import google.generativeai as genai
import numpy as np
import imageio
from pathlib import Path
import json
import time
import math
import threading
from queue import Queue

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from flux_kontext_helpers import flux_kontext_singleton
from ltx_manager_helpers import ltx_manager_singleton
from ltx_upscaler_manager_helpers import ltx_upscaler_manager_singleton

WORKSPACE_DIR = "aduc_workspace"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ======================================================================================
# SEÇÃO 1: FUNÇÕES UTILITÁRIAS E DE PROCESSAMENTO DE MÍDIA
# ======================================================================================

def robust_json_parser(raw_text: str) -> dict:
    clean_text = raw_text.strip()
    try:
        start_index = clean_text.find('{'); end_index = clean_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = clean_text[start_index : end_index + 1]
            return json.loads(json_str)
        else: raise ValueError("Nenhum objeto JSON válido encontrado na resposta da IA.")
    except json.JSONDecodeError as e: raise ValueError(f"Falha ao decodificar JSON: {e}")

def process_image_to_square(image_path: str, size: int, output_filename: str = None) -> str:
    if not image_path: return None
    try:
        img = Image.open(image_path).convert("RGB")
        img_square = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        if output_filename: output_path = os.path.join(WORKSPACE_DIR, output_filename)
        else: output_path = os.path.join(WORKSPACE_DIR, f"edited_ref_{time.time()}.png")
        img_square.save(output_path)
        return output_path
    except Exception as e: raise gr.Error(f"Falha ao processar a imagem de referência: {e}")

def trim_video_to_frames(input_path: str, output_path: str, frames_to_keep: int) -> str:
    try:
        command = f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='lt(n,{frames_to_keep})'\" -an \"{output_path}\""
        subprocess.run(command, shell=True, check=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e: raise gr.Error(f"FFmpeg falhou ao cortar vídeo: {getattr(e, 'stderr', str(e))}")

def extract_last_n_frames_as_video(input_path: str, output_path: str, n_frames: int) -> str:
    try:
        cmd_probe = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 \"{input_path}\""
        result = subprocess.run(cmd_probe, shell=True, check=True, text=True, capture_output=True)
        total_frames = int(result.stdout.strip())
        if n_frames >= total_frames:
            shutil.copyfile(input_path, output_path)
            return output_path
        start_frame = total_frames - n_frames
        cmd_ffmpeg = f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='gte(n,{start_frame})'\" -vframes {n_frames} -an \"{output_path}\""
        subprocess.run(cmd_ffmpeg, shell=True, check=True, text=True)
        return output_path
    except (subprocess.CalledProcessError, ValueError) as e: raise gr.Error(f"FFmpeg falhou ao extrair os últimos {n_frames} frames: {getattr(e, 'stderr', str(e))}")

def concatenate_final_video(fragment_paths: list, fragment_duration_frames: int, eco_video_frames: int, progress=gr.Progress()):
    if not fragment_paths:
        raise gr.Error("Nenhum fragmento de vídeo para concatenar.")
    progress(0.1, desc="Preparando fragmentos para a montagem final...");
    try:
        list_file_path = os.path.abspath(os.path.join(WORKSPACE_DIR, f"concat_list_final_{time.time()}.txt"))
        final_output_path = os.path.abspath(os.path.join(WORKSPACE_DIR, "masterpiece_final.mp4"))
        temp_files_for_concat = []
        duration_for_non_cut_fragments = max(1, int(fragment_duration_frames - eco_video_frames))
        sorted_fragment_paths = sorted(fragment_paths)
        for i, p in enumerate(sorted_fragment_paths):
            is_last_fragment = (i == len(sorted_fragment_paths) - 1)
            if "_cut" in os.path.basename(p) or is_last_fragment:
                temp_files_for_concat.append(os.path.abspath(p))
            else:
                temp_path = os.path.join(WORKSPACE_DIR, f"final_temp_concat_{i}.mp4")
                trim_video_to_frames(p, temp_path, duration_for_non_cut_fragments)
                temp_files_for_concat.append(os.path.abspath(temp_path))
        progress(0.8, desc="Concatenando clipe final...");
        with open(list_file_path, "w") as f:
            for p_temp in temp_files_for_concat:
                f.write(f"file '{p_temp}'\n")
        ffmpeg_command = f"ffmpeg -y -v error -f concat -safe 0 -i \"{list_file_path}\" -c copy \"{final_output_path}\""
        subprocess.run(ffmpeg_command, shell=True, check=True, text=True)
        progress(1.0, desc="Montagem final concluída!");
        return final_output_path
    except subprocess.CalledProcessError as e:
        raise gr.Error(f"FFmpeg falhou na concatenação final: {e.stderr if e.stderr else 'Nenhum erro reportado.'}")
    except Exception as e:
        raise gr.Error(f"Um erro ocorreu durante a concatenação final: {e}")

def extract_image_exif(image_path: str) -> str:
    try:
        img = Image.open(image_path); exif_data = img._getexif()
        if not exif_data: return "No EXIF metadata found."
        exif = { ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS }
        relevant_tags = ['DateTimeOriginal', 'Model', 'LensModel', 'FNumber', 'ExposureTime', 'ISOSpeedRatings', 'FocalLength']
        metadata_str = ", ".join(f"{key}: {exif[key]}" for key in relevant_tags if key in exif)
        return metadata_str if metadata_str else "No relevant EXIF metadata found."
    except Exception: return "Could not read EXIF data."

# ======================================================================================
# SEÇÃO 2: ORQUESTRADORES DE IA
# ======================================================================================

def run_storyboard_generation(num_fragments: int, prompt: str, reference_paths: list):
    if not reference_paths: raise gr.Error("Por favor, forneça pelo menos uma imagem de referência.")
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    main_ref_path = reference_paths[0]
    exif_metadata = extract_image_exif(main_ref_path)
    prompt_file = "prompts/unified_storyboard_prompt.txt"
    with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
    director_prompt = template.format(user_prompt=prompt, num_fragments=int(num_fragments), image_metadata=exif_metadata)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
    model_contents = [director_prompt]
    for i, img_path in enumerate(reference_paths):
        model_contents.append(f"Reference Image {i+1}:")
        model_contents.append(Image.open(img_path))
    print(f"Gerando roteiro com {len(reference_paths)} imagens de referência...")
    response = model.generate_content(model_contents)
    try:
        storyboard_data = robust_json_parser(response.text)
        storyboard = storyboard_data.get("scene_storyboard", [])
        if not storyboard or len(storyboard) != int(num_fragments): raise ValueError(f"A IA não gerou o número correto de cenas. Esperado: {num_fragments}, Recebido: {len(storyboard)}")
        return storyboard
    except Exception as e: raise gr.Error(f"O Roteirista (Gemini) falhou ao criar o roteiro: {e}. Resposta recebida: {response.text}")

def run_keyframe_generation(storyboard, fixed_reference_paths, keyframe_resolution, global_prompt, progress=gr.Progress()):
    if not storyboard: raise gr.Error("Nenhum roteiro para gerar keyframes.")
    if not fixed_reference_paths: raise gr.Error("A imagem de referência inicial é obrigatória.")
    
    initial_ref_image_path = fixed_reference_paths[0]
    width, height = keyframe_resolution, keyframe_resolution
    
    keyframe_paths_for_video = [] 
    scene_history = "N/A"
    
    wrapper_prompt_path = os.path.join(os.path.dirname(__file__), "prompts/flux_composition_wrapper_prompt.txt")
    with open(wrapper_prompt_path, "r", encoding="utf-8") as f: kontext_template = f.read()
    director_prompt_path = os.path.join(os.path.dirname(__file__), "prompts/director_composition_prompt.txt")
    with open(director_prompt_path, "r", encoding="utf-8") as f: director_template = f.read()
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')
        for i, scene_description in enumerate(storyboard):
            progress(i / len(storyboard), desc=f"Compondo Keyframe {i+1}/{len(storyboard)}")
            print(f"\n--- COMPONDO KEYFRAME {i+1}/{len(storyboard)} ---")
            
            last_three_paths = ([initial_ref_image_path] + keyframe_paths_for_video)[-3:]
            
            print(f"  - Diretor de Cena está analisando o contexto...")
            director_prompt = director_template.format(global_prompt=global_prompt, scene_history=scene_history, current_scene_desc=scene_description)
            
            model_contents, image_map, current_image_index = [], {}, 1
            for path in last_three_paths:
                if path not in image_map.values():
                    image_map[current_image_index] = path
                    model_contents.extend([f"IMG-{current_image_index}:", Image.open(path)])
                    current_image_index += 1
            for path in fixed_reference_paths:
                if path not in image_map.values():
                    image_map[current_image_index] = path
                    model_contents.extend([f"IMG-{current_image_index}:", Image.open(path)])
                    current_image_index += 1
            model_contents.append(director_prompt)

            response_text = model.generate_content(model_contents).text
            composition_prompt_with_tags = response_text.strip()
            referenced_indices = [int(idx) for idx in re.findall(r'\[IMG-(\d+)\]', composition_prompt_with_tags)]
            current_reference_paths = [image_map[idx] for idx in sorted(list(set(referenced_indices))) if idx in image_map]
            if not current_reference_paths: current_reference_paths = [last_three_paths[-1]]
            
            reference_images_pil = [Image.open(p) for p in current_reference_paths]
            final_kontext_prompt = re.sub(r'\[IMG-\d+\]', '', composition_prompt_with_tags).strip()
            print(f"  - Prompt Final do Diretor: \"{final_kontext_prompt}\"")
            scene_history += f"Scene {i+1}: {final_kontext_prompt}\n"
            
            final_kontext_prompt_wrapped = kontext_template.format(target_prompt=final_kontext_prompt)
            output_path = os.path.join(WORKSPACE_DIR, f"keyframe_{i+1}.png")
            image = flux_kontext_singleton.generate_image(reference_images=reference_images_pil, prompt=final_kontext_prompt_wrapped, width=width, height=height, seed=int(time.time()))
            image.save(output_path)
            keyframe_paths_for_video.append(output_path)
            
    except Exception as e: 
        raise gr.Error(f"O Compositor (FluxKontext) ou o Diretor de Cena (Gemini) falhou: {e}")
        
    print("\nComposição de todos os keyframes concluída.")
    final_keyframes = keyframe_paths_for_video
    return final_keyframes, final_keyframes

def get_initial_motion_prompt(user_prompt: str, start_image_path: str, destination_image_path: str, dest_scene_desc: str):
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    try:
        genai.configure(api_key=GEMINI_API_KEY); model = genai.GenerativeModel('gemini-2.5-flash'); prompt_file = "prompts/initial_motion_prompt.txt"
        with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
        cinematographer_prompt = template.format(user_prompt=user_prompt, destination_scene_description=dest_scene_desc)
        start_img, dest_img = Image.open(start_image_path), Image.open(destination_image_path)
        model_contents = ["START Image:", start_img, "DESTINATION Image:", dest_img, cinematographer_prompt]
        response = model.generate_content(model_contents)
        return response.text.strip()
    except Exception as e: raise gr.Error(f"O Cineasta de IA (Inicial) falhou: {e}. Resposta: {getattr(e, 'text', 'No text available.')}")

def get_transition_decision(user_prompt, story_history, memory_media_path, path_image_path, destination_image_path, midpoint_scene_description, dest_scene_desc):
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    try:
        genai.configure(api_key=GEMINI_API_KEY); model = genai.GenerativeModel('gemini-2.5-flash'); prompt_file = "prompts/transition_decision_prompt.txt"
        with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
        continuity_prompt = template.format(user_prompt=user_prompt, story_history=story_history, midpoint_scene_description=midpoint_scene_description, destination_scene_description=dest_scene_desc)
        mem_img = Image.open(memory_media_path) if isinstance(memory_media_path, str) else memory_media_path
        path_img, dest_img = Image.open(path_image_path), Image.open(destination_image_path)
        model_contents = ["START Image (from Kinetic Echo):", mem_img, "MIDPOINT Image (Path):", path_img, "DESTINATION Image (Destination):", dest_img, continuity_prompt]
        response = model.generate_content(model_contents)
        decision_data = robust_json_parser(response.text)
        if "transition_type" not in decision_data or "motion_prompt" not in decision_data: raise ValueError("A resposta da IA não contém as chaves 'transition_type' ou 'motion_prompt'.")
        return decision_data
    except Exception as e: raise gr.Error(f"O Diretor de Continuidade (IA) falhou: {e}. Resposta: {getattr(e, 'text', str(e))}")

# ======================================================================================
# SEÇÃO 3: LÓGICA DE PRODUÇÃO COM FILAS ASSÍNCRONAS
# ======================================================================================

def generation_worker(
    tasks_list, upscale_queue, progress,
    prompt_geral, scene_storyboard, seed, cfg,
    video_total_frames_ltx, video_fps, num_inference_steps, use_attention_slicing,
    decode_timestep, image_cond_noise_scale, fragment_duration_frames, eco_video_frames,
    mid_cond_strength, dest_cond_strength, low_res_width, low_res_height
):
    kinetic_memory_path = None
    story_history = ""
    total_tasks = len(tasks_list)
    for i, task_info in enumerate(tasks_list):
        fragment_num = i + 1
        progress(i / total_tasks, desc=f"Decidindo/Gerando Low-Res {fragment_num}/{total_tasks}...")
        
        start_path = task_info['start_path']
        destination_path = task_info['destination_path']
        
        if i == 0:
            dest_scene_desc = scene_storyboard[i]
            current_motion_prompt = get_initial_motion_prompt(prompt_geral, start_path, destination_path, dest_scene_desc)
            conditioning_items_data = [(start_path, 0, 1.0), (destination_path, video_total_frames_ltx - 1, dest_cond_strength)]
            transition_type = "continuous"
        else:
            path_path = start_path
            path_scene_desc = scene_storyboard[i-1]
            dest_scene_desc = scene_storyboard[i]
            decision_data = get_transition_decision(prompt_geral, story_history, kinetic_memory_path, path_path, destination_path, midpoint_scene_description=path_scene_desc, dest_scene_desc=dest_scene_desc)
            transition_type = decision_data["transition_type"]
            current_motion_prompt = decision_data["motion_prompt"]
            mid_cond_frame = int(video_total_frames_ltx - fragment_duration_frames + eco_video_frames)
            conditioning_items_data = [(kinetic_memory_path, 0, 1.0), (path_path, mid_cond_frame, mid_cond_strength), (destination_path, video_total_frames_ltx - 1, dest_cond_strength)]

        story_history += f"\n- Ato {fragment_num}: {current_motion_prompt}"
        output_path_low_res = os.path.join(WORKSPACE_DIR, f"fragment_{fragment_num}_lowres_{transition_type}.mp4")
        
        _, _ = ltx_manager_singleton.generate_video_fragment(
            motion_prompt=current_motion_prompt, conditioning_items_data=conditioning_items_data,
            width=low_res_width, height=low_res_height, seed=seed, cfg=cfg,
            video_total_frames=video_total_frames_ltx, video_fps=video_fps,
            num_inference_steps=num_inference_steps, use_attention_slicing=use_attention_slicing,
            decode_timestep=decode_timestep, image_cond_noise_scale=image_cond_noise_scale,
            current_fragment_index=fragment_num, output_path=output_path_low_res, progress=progress
        )
        
        upscale_task = {"input_path": output_path_low_res, "output_path": output_path_low_res.replace("_lowres_", "_highres_"), "video_fps": video_fps}
        upscale_queue.put(upscale_task)

        is_last_fragment = (i == total_tasks - 1)
        if not is_last_fragment and transition_type != "cut":
            trimmed_fragment_path = output_path_low_res.replace(".mp4", "_trimmed.mp4")
            trim_video_to_frames(output_path_low_res, trimmed_fragment_path, int(fragment_duration_frames))
            eco_output_path = os.path.join(WORKSPACE_DIR, f"eco_from_frag_{fragment_num}.mp4")
            kinetic_memory_path = extract_last_n_frames_as_video(trimmed_fragment_path, eco_output_path, int(eco_video_frames))
        else:
            kinetic_memory_path = None

def upscaling_worker(upscale_queue, final_results_list):
    while True:
        task = upscale_queue.get()
        if task is None:
            upscale_queue.task_done()
            break
        try:
            upscaled_path = ltx_upscaler_manager_singleton.upscale_video_fragment(
                video_path_low_res=task['input_path'], output_path=task['output_path'], video_fps=task['video_fps']
            )
            final_results_list.append(upscaled_path)
        except Exception as e:
            print(f"ERRO no worker de upscale: {e}")
        finally:
            upscale_queue.task_done()

def run_video_production(
    video_resolution,
    video_duration_seconds, video_fps, eco_video_frames, use_attention_slicing,
    fragment_duration_frames, mid_cond_strength, dest_cond_strength, num_inference_steps,
    decode_timestep, image_cond_noise_scale,
    prompt_geral, keyframe_images_state, scene_storyboard, cfg,
    progress=gr.Progress()
):
    try:
        high_res_width, high_res_height = video_resolution, video_resolution
        low_res_scale = 2 
        low_res_width = (high_res_width // low_res_scale // 8) * 8
        low_res_height = (high_res_height // low_res_scale // 8) * 8
        
        valid_keyframes = [p for p in keyframe_images_state if p is not None and os.path.exists(p)]
        video_total_frames_user = int(video_duration_seconds * video_fps)
        video_total_frames_ltx = int(round((float(video_total_frames_user) - 1.0) / 8.0) * 8 + 1)

        if not valid_keyframes or len(valid_keyframes) < 2: raise gr.Error("São necessários pelo menos 2 keyframes válidos para produzir uma transição.")

        print(f"\n--- FASE 3/4: Iniciando Pipeline de Produção Assíncrona ---")
        
        seed = int(time.time())
        num_transitions = len(valid_keyframes) - 1
        
        generation_tasks = []
        for i in range(num_transitions):
            task_info = {"start_path": valid_keyframes[i], "destination_path": valid_keyframes[i+1]}
            generation_tasks.append(task_info)
        
        print("\nTodas as tarefas de geração foram planejadas. Iniciando workers...")
        
        upscaling_queue = Queue()
        final_results_high_res = []
        
        worker_args = (
            generation_tasks, upscaling_queue, progress,
            prompt_geral, scene_storyboard, seed, cfg,
            video_total_frames_ltx, video_fps, num_inference_steps, use_attention_slicing,
            decode_timestep, image_cond_noise_scale, fragment_duration_frames, eco_video_frames,
            mid_cond_strength, dest_cond_strength, low_res_width, low_res_height
        )
        
        gen_worker_thread = threading.Thread(target=generation_worker, args=worker_args)
        upscale_worker_thread = threading.Thread(target=upscaling_worker, args=(upscaling_queue, final_results_high_res))
        
        gen_worker_thread.start()
        upscale_worker_thread.start()

        gen_worker_thread.join()
        upscaling_queue.put(None)
        upscale_worker_thread.join()
        
        progress(1.0, desc="Produção e upscaling concluídos.")
        print("\nTodos os fragmentos foram processados.")
        
        return (
            sorted(final_results_high_res), 
            sorted(final_results_high_res),
            None,
            gr.update(visible=False),
            None,
            fragment_duration_frames,
            eco_video_frames
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"A Produção de Vídeo (LTX) falhou: {e}")

# ======================================================================================
# SEÇÃO 4: UI e Lógica de Conexão
# ======================================================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# NOVIM-13.1 (Painel de Controle do Diretor)\n*Arquitetura ADUC-SDR com Pipeline Assíncrono*")

    if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR); Path("prompts").mkdir(exist_ok=True)

    # --- Definição dos Estados da UI ---
    scene_storyboard_state = gr.State([])
    keyframe_images_state = gr.State([])
    fragment_list_state = gr.State([])
    prompt_geral_state = gr.State("")
    processed_ref_paths_state = gr.State([])
    fragment_duration_state = gr.State()
    eco_frames_state = gr.State()

    # --- Layout da UI ---
    gr.Markdown("## CONFIGURAÇÕES GLOBAIS DE RESOLUÇÃO")
    with gr.Row():
        video_resolution_selector = gr.Radio([512, 720, 1024], value=1024, label="Resolução Final do Vídeo (px)")
        keyframe_resolution_selector = gr.Radio([512, 720, 1024], value=512, label="Resolução dos Keyframes (px)")

    gr.Markdown("--- \n ## ETAPA 1: O ROTEIRO (IA Roteirista)")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Ideia Geral (Prompt)")
            num_fragments_input = gr.Slider(2, 50, 4, step=1, label="Nº de Keyframes a Gerar")
            reference_gallery = gr.Gallery(label="Imagens de Referência (A primeira é a principal)", type="filepath", columns=4, rows=1, object_fit="contain", height="auto")
            director_button = gr.Button("▶️ 1. Gerar Roteiro", variant="primary")
        with gr.Column(scale=2): storyboard_to_show = gr.JSON(label="Roteiro de Cenas Gerado (em Inglês)")

    gr.Markdown("--- \n ## ETAPA 2: OS KEYFRAMES (IA Compositor & Diretor de Cena)")
    with gr.Row():
        with gr.Column(scale=2):
            photographer_button = gr.Button("▶️ 2. Compor Keyframes e Iniciar Produção", variant="primary")
            keyframe_gallery_output = gr.Gallery(label="Galeria de Keyframes Gerados", object_fit="contain", height="auto", type="filepath", interactive=False)

    gr.Markdown("--- \n ## ETAPA 3: A PRODUÇÃO (IA Cineasta & Câmera)")
    with gr.Row():
        with gr.Column(scale=1):
            cfg_slider = gr.Slider(0.5, 10.0, 1.0, step=0.1, label="CFG (Guidance Scale)")
            with gr.Accordion("Controles Avançados de Timing e Performance", open=False):
                video_duration_slider = gr.Slider(label="Duração da Geração Bruta (s)", minimum=2.0, maximum=10.0, value=6.0, step=0.5)
                video_fps_radio = gr.Radio(choices=[8, 16, 24, 32], value=24, label="FPS do Vídeo")
                num_inference_steps_slider = gr.Slider(label="Etapas de Inferência (Low-Res)", minimum=10, maximum=50, value=28, step=1)
                slicing_checkbox = gr.Checkbox(label="Usar Attention Slicing (Economiza VRAM)", value=True)
                gr.Markdown("---"); gr.Markdown("#### Controles de Duração (Arquitetura Eco + Déjà Vu)")
                fragment_duration_slider = gr.Slider(label="Duração de Cada Fragmento (% da Geração Bruta)", minimum=1, maximum=100, value=75, step=1)
                eco_frames_slider = gr.Slider(label="Tamanho do Eco Cinético (Frames)", minimum=4, maximum=48, value=8, step=1)
                mid_cond_strength_slider = gr.Slider(label="Força do 'Caminho'", minimum=0.1, maximum=1.0, value=0.5, step=0.05)
                dest_cond_strength_slider = gr.Slider(label="Força do 'Destino'", minimum=0.1, maximum=1.0, value=1.0, step=0.05)
                gr.Markdown("---"); gr.Markdown("#### Controles do VAE (Avançado)")
                decode_timestep_slider = gr.Slider(label="VAE Decode Timestep", minimum=0.0, maximum=0.2, value=0.05, step=0.005)
                image_cond_noise_scale_slider = gr.Slider(label="VAE Image Cond Noise Scale", minimum=0.0, maximum=0.1, value=0.025, step=0.005)
                
            with gr.Accordion("Visualização das Mídias de Condicionamento", open=True):
                with gr.Row():
                    prod_media_start_output = gr.Video(label="Mídia Inicial (Eco/K1)", interactive=False)
                    prod_media_mid_output = gr.Image(label="Mídia do Caminho (K_i-1)", interactive=False, visible=False)
                    prod_media_end_output = gr.Image(label="Mídia de Destino (K_i)", interactive=False)
        with gr.Column(scale=1): video_gallery_output = gr.Gallery(label="Fragmentos Gerados (High-Res)", object_fit="contain", height="auto", type="video")

    gr.Markdown(f"--- \n ## ETAPA 4: PÓS-PRODUÇÃO (Montagem Final)")
    with gr.Row():
        with gr.Column():
            editor_button = gr.Button("▶️ 4. Montar Vídeo Final", variant="primary")
            final_video_output = gr.Video(label="A Obra-Prima Final")

    gr.Markdown(
        """
        ---
        ### A Arquitetura: ADUC-SDR com Pipeline Assíncrono
        **ADUC (Arquitetura de Unificação Compositiva):** O sistema usa uma equipe de IAs especializadas. Um **Roteirista** cria a história. Um **Diretor de Cena** compõe cada keyframe. Um **Compositor** (`FluxKontext`) cria as imagens.
        
        **SDR (Escala Dinâmica e Resiliente):** A produção opera como uma linha de montagem com trabalhadores (pools de GPU) independentes e filas de trabalho. O **Gerador** (`cuda:2`/`cuda:3`) produz um fragmento em baixa resolução e o coloca na fila do **Upscaler**. O Upscaler (`cuda:0`/`cuda:1`) pega o trabalho da fila e o refina para alta resolução, enquanto o Gerador já está produzindo o próximo fragmento. Isso garante que todas as GPUs estejam trabalhando em paralelo para máxima eficiência.
        """
    )
    
    def process_and_run_storyboard(num_fragments, prompt, gallery_files, keyframe_resolution):
        if not gallery_files:
            raise gr.Error("Por favor, suba pelo menos uma imagem de referência na galeria.")
        raw_paths = [item[0] for item in gallery_files]
        processed_paths = []
        for i, path in enumerate(raw_paths):
            filename = f"processed_ref_{i}_{keyframe_resolution}x{keyframe_resolution}.png"
            processed_path = process_image_to_square(path, keyframe_resolution, filename)
            processed_paths.append(processed_path)
        storyboard = run_storyboard_generation(num_fragments, prompt, processed_paths)
        return storyboard, prompt, processed_paths

    def run_keyframes_and_video(
        # Inputs da Etapa 2
        storyboard, fixed_reference_paths, keyframe_resolution, global_prompt,
        # Inputs da Etapa 3
        video_resolution, video_duration_seconds, video_fps, eco_video_frames, 
        use_attention_slicing, fragment_duration_percentage, mid_cond_strength, 
        dest_cond_strength, num_inference_steps, decode_timestep, 
        image_cond_noise_scale, cfg, progress=gr.Progress()
    ):
        # --- ETAPA 2 ---
        keyframe_paths, _ = run_keyframe_generation(
            storyboard, fixed_reference_paths, keyframe_resolution, global_prompt, progress
        )
        
        # Prepara a UI para a Etapa 3
        progress(0, desc="Iniciando produção de vídeo...")
        
        # --- ETAPA 3 ---
        total_frames = video_duration_seconds * video_fps
        fragment_duration_in_frames = int(math.floor((fragment_duration_percentage / 100.0) * total_frames))
        fragment_duration_in_frames = max(1, fragment_duration_in_frames)

        # Chama a função síncrona de produção
        final_gallery, final_state, start_media, mid_media, end_media, frag_dur, eco_f = run_video_production(
            video_resolution, video_duration_seconds, video_fps, eco_video_frames, use_attention_slicing,
            fragment_duration_in_frames, mid_cond_strength, dest_cond_strength, num_inference_steps,
            decode_timestep, image_cond_noise_scale,
            global_prompt, keyframe_paths, storyboard, cfg, progress
        )

        # Retorna todos os valores para todos os outputs
        return keyframe_paths, keyframe_paths, final_gallery, final_state, start_media, mid_media, end_media, frag_dur, eco_f

    director_button.click(
        fn=process_and_run_storyboard,
        inputs=[num_fragments_input, prompt_input, reference_gallery, keyframe_resolution_selector],
        outputs=[scene_storyboard_state, prompt_geral_state, processed_ref_paths_state]
    ).success(fn=lambda s: s, inputs=[scene_storyboard_state], outputs=[storyboard_to_show])

    photographer_button.click(
        fn=run_keyframes_and_video,
        inputs=[
            scene_storyboard_state, processed_ref_paths_state, keyframe_resolution_selector, prompt_geral_state,
            video_resolution_selector, video_duration_slider, video_fps_radio, eco_frames_slider, slicing_checkbox,
            fragment_duration_slider, mid_cond_strength_slider, dest_cond_strength_slider, num_inference_steps_slider,
            decode_timestep_slider, image_cond_noise_scale_slider, cfg_slider
        ],
        outputs=[
            keyframe_gallery_output, 
            keyframe_images_state,
            video_gallery_output, 
            fragment_list_state,
            prod_media_start_output, 
            prod_media_mid_output, 
            prod_media_end_output,
            fragment_duration_state, 
            eco_frames_state
        ]
    )
    
    editor_button.click(
        fn=concatenate_final_video,
        inputs=[fragment_list_state, fragment_duration_state, eco_frames_state],
        outputs=[final_video_output]
    )

if __name__ == "__main__":
    if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR); Path("prompts").mkdir(exist_ok=True)
    
    demo.queue().launch(server_name="0.0.0.0", share=True)