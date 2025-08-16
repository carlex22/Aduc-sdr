# Euia-AducSdr: Uma implementação aberta e funcional da arquitetura ADUC-SDR para geração de vídeo coerente.
# Copyright (C) 4 de Agosto de 2025  Carlos Rodrigues dos Santos
# (Licença e cabeçalho completo)

# --- app.py (NOVIM-12.0: Lógica de "Cena Nova" Refinada) ---

import gradio as gr
import torch
import os
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

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from dreamo_worker_base import dreamo_base_singleton
from ltx_manager_helpers import ltx_manager_singleton

WORKSPACE_DIR = "aduc_workspace"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

def robust_json_parser(raw_text: str) -> dict:
    try:
        start_index = raw_text.find('{'); end_index = raw_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = raw_text[start_index : end_index + 1]; return json.loads(json_str)
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
        subprocess.run(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='lt(n,{frames_to_keep})'\" -an \"{output_path}\"", shell=True, check=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e: raise gr.Error(f"FFmpeg falhou ao cortar vídeo: {e.stderr}")

def extract_last_n_frames_as_video(input_path: str, output_path: str, n_frames: int) -> str:
    try:
        cmd_probe = f"ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 \"{input_path}\""
        result = subprocess.run(cmd_probe, shell=True, check=True, text=True, capture_output=True)
        total_frames = int(result.stdout.strip())
        if n_frames >= total_frames: shutil.copyfile(input_path, output_path); return output_path
        start_frame = total_frames - n_frames
        cmd_ffmpeg = f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='gte(n,{start_frame})'\" -vframes {n_frames} -an \"{output_path}\""
        subprocess.run(cmd_ffmpeg, shell=True, check=True, text=True)
        return output_path
    except (subprocess.CalledProcessError, ValueError) as e: raise gr.Error(f"FFmpeg falhou ao extrair os últimos {n_frames} frames: {getattr(e, 'stderr', str(e))}")

def concatenate_final_video(fragment_paths: list, fragment_duration_frames: int, eco_video_frames: int, progress=gr.Progress()):
    if not fragment_paths: raise gr.Error("Nenhum fragmento de vídeo para concatenar.")
    
    progress(0.1, desc="Preparando e cortando fragmentos para a montagem final...");
    try:
        list_file_path = os.path.abspath(os.path.join(WORKSPACE_DIR, f"concat_list_final_{time.time()}.txt"))
        final_output_path = os.path.abspath(os.path.join(WORKSPACE_DIR, "masterpiece_final.mp4"))
        
        temp_files_for_concat = []
        final_clip_len = int(fragment_duration_frames - eco_video_frames)

        for i, p in enumerate(fragment_paths):
            is_last_fragment = (i == len(fragment_paths) - 1)
            
            if is_last_fragment or "_cut.mp4" in os.path.basename(p):
                temp_files_for_concat.append(os.path.abspath(p))
            else:
                temp_path = os.path.join(WORKSPACE_DIR, f"final_temp_concat_{i}.mp4")
                trim_video_to_frames(p, temp_path, final_clip_len)
                temp_files_for_concat.append(os.path.abspath(temp_path))

        progress(0.8, desc="Concatenando clipe final...")
        with open(list_file_path, "w") as f:
            for p_temp in temp_files_for_concat:
                f.write(f"file '{p_temp}'\n")
        
        ffmpeg_command = f"ffmpeg -y -v error -f concat -safe 0 -i \"{list_file_path}\" -c copy \"{final_output_path}\""
        subprocess.run(ffmpeg_command, shell=True, check=True, text=True)
        progress(1.0, desc="Montagem final concluída!")
        return final_output_path
    except subprocess.CalledProcessError as e:
        error_output = e.stderr if e.stderr else "Nenhuma saída de erro do FFmpeg."
        raise gr.Error(f"FFmpeg falhou na concatenação final: {error_output}")

def extract_image_exif(image_path: str) -> str:
    try:
        img = Image.open(image_path); exif_data = img._getexif()
        if not exif_data: return "No EXIF metadata found."
        exif = { ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS }
        relevant_tags = ['DateTimeOriginal', 'Model', 'LensModel', 'FNumber', 'ExposureTime', 'ISOSpeedRatings', 'FocalLength']
        metadata_str = ", ".join(f"{key}: {exif[key]}" for key in relevant_tags if key in exif)
        return metadata_str if metadata_str else "No relevant EXIF metadata found."
    except Exception: return "Could not read EXIF data."

def run_storyboard_generation(num_fragments: int, prompt: str, initial_image_path: str):
    if not initial_image_path: raise gr.Error("Por favor, forneça uma imagem de referência inicial.")
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    exif_metadata = extract_image_exif(initial_image_path)
    prompt_file = "prompts/unified_storyboard_prompt.txt"
    with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
    director_prompt = template.format(user_prompt=prompt, num_fragments=int(num_fragments), image_metadata=exif_metadata)
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash'); img = Image.open(initial_image_path)
    print("Gerando roteiro com análise de visão integrada...")
    response = model.generate_content([director_prompt, img])
    try:
        storyboard_data = robust_json_parser(response.text)
        storyboard = storyboard_data.get("scene_storyboard", [])
        if not storyboard or len(storyboard) != int(num_fragments): raise ValueError(f"A IA não gerou o número correto de cenas. Esperado: {num_fragments}, Recebido: {len(storyboard)}")
        return storyboard
    except Exception as e: raise gr.Error(f"O Roteirista (Gemini) falhou ao criar o roteiro: {e}. Resposta recebida: {response.text}")

def get_dreamo_prompt_for_transition(previous_image_path: str, target_scene_description: str) -> str:
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    genai.configure(api_key=GEMINI_API_KEY)
    prompt_file = "prompts/img2img_evolution_prompt.txt"
    with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
    director_prompt = template.format(target_scene_description=target_scene_description)
    model = genai.GenerativeModel('gemini-1.5-flash'); img = Image.open(previous_image_path)
    response = model.generate_content([director_prompt, "Previous Image:", img])
    return response.text.strip().replace("\"", "")

def get_initial_motion_prompt(user_prompt: str, start_image_path: str, destination_image_path: str, dest_scene_desc: str):
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    try:
        genai.configure(api_key=GEMINI_API_KEY); model = genai.GenerativeModel('gemini-1.5-flash'); prompt_file = "prompts/initial_motion_prompt.txt"
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
        genai.configure(api_key=GEMINI_API_KEY); model = genai.GenerativeModel('gemini-1.5-flash'); prompt_file = "prompts/transition_decision_prompt.txt"
        with open(os.path.join(os.path.dirname(__file__), prompt_file), "r", encoding="utf-8") as f: template = f.read()
        continuity_prompt = template.format(user_prompt=user_prompt, story_history=story_history, midpoint_scene_description=midpoint_scene_description, destination_scene_description=dest_scene_desc)
        with imageio.get_reader(memory_media_path) as reader: mem_img = Image.fromarray(reader.get_data(0))
        path_img, dest_img = Image.open(path_image_path), Image.open(destination_image_path)
        model_contents = ["START Image (from Kinetic Echo):", mem_img, "MIDPOINT Image (Path):", path_img, "DESTINATION Image (Destination):", dest_img, continuity_prompt]
        response = model.generate_content(model_contents)
        decision_data = robust_json_parser(response.text)
        if "transition_type" not in decision_data or "motion_prompt" not in decision_data: raise ValueError("A resposta da IA não contém as chaves 'transition_type' ou 'motion_prompt'.")
        return decision_data
    except Exception as e: raise gr.Error(f"O Diretor de Continuidade (IA) falhou: {e}. Resposta: {getattr(e, 'text', str(e))}")

def run_keyframe_generation(storyboard, ref_images_tasks, keyframe_resolution, progress=gr.Progress()):
    if not storyboard: raise gr.Error("Nenhum roteiro para gerar keyframes.")
    initial_ref_image_path = ref_images_tasks[0]['image']
    if not initial_ref_image_path or not os.path.exists(initial_ref_image_path): raise gr.Error("A imagem de referência inicial é obrigatória.")
    log_history = ""; generated_images_for_gallery = []
    try:
        dreamo_base_singleton.to_gpu()
        width, height = keyframe_resolution, keyframe_resolution
        keyframe_paths_for_video = []
        current_ref_image_path_for_dreamo = initial_ref_image_path
        for i, scene_description in enumerate(storyboard):
            progress(i / len(storyboard), desc=f"Pintando Keyframe {i+1}/{len(storyboard)} ({width}x{height})")
            log_history += f"\n--- PINTANDO KEYFRAME {i+1}/{len(storyboard)} ---\n"
            dreamo_prompt = get_dreamo_prompt_for_transition(current_ref_image_path_for_dreamo, scene_description)
            style_prefix = "generate a same style image, "
            has_style_ref = any(item.get('task') == 'style' for item in ref_images_tasks if item.get('image'))
            if has_style_ref: dreamo_prompt = style_prefix + dreamo_prompt
            reference_items = []
            fixed_references_basenames = [os.path.basename(item['image']) for item in ref_images_tasks if item.get('image')]
            for item in ref_images_tasks:
                if item.get('image'): reference_items.append({'image_np': np.array(Image.open(item['image']).convert("RGB")), 'task': item['task']})
            dynamic_references_paths = keyframe_paths_for_video[-3:] if keyframe_paths_for_video else []
            if i > 0 and keyframe_paths_for_video:
                 if not any(keyframe_paths_for_video[-1] in p for p in dynamic_references_paths): dynamic_references_paths.insert(0, keyframe_paths_for_video[-1])
            elif i == 0: dynamic_references_paths.insert(0, current_ref_image_path_for_dreamo)
            for ref_path in dynamic_references_paths:
                if os.path.basename(ref_path) not in fixed_references_basenames: reference_items.append({'image_np': np.array(Image.open(ref_path).convert("RGB")), 'task': 'ip'})
            log_history += f"  - Roteiro: '{scene_description}'\n  - Usando {len(reference_items)} referências visuais.\n"
            log_history += f"  - Prompt do D.A.: \"{dreamo_prompt}\"\n"
            yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=generated_images_for_gallery)}
            output_path = os.path.join(WORKSPACE_DIR, f"keyframe_{i+1}.png")
            image = dreamo_base_singleton.generate_image(reference_items=reference_items, prompt=dreamo_prompt, width=width, height=height)
            image.save(output_path)
            keyframe_paths_for_video.append(output_path)
            current_ref_image_path_for_dreamo = output_path
            generated_images_for_gallery.append(output_path)
    except Exception as e: 
        raise gr.Error(f"O Pintor (DreamO) falhou: {e}")
    finally:
        dreamo_base_singleton.to_cpu()
    log_history += "\nPintura de todos os keyframes concluída.\n"
    final_keyframes = keyframe_paths_for_video
    yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: final_keyframes, keyframe_images_state: final_keyframes}

def run_video_production(
    video_resolution,
    video_duration_seconds, video_fps, eco_video_frames, use_attention_slicing,
    fragment_duration_frames, mid_cond_strength, dest_cond_strength, num_inference_steps,
    decode_timestep, image_cond_noise_scale,
    prompt_geral, keyframe_images_state, scene_storyboard, cfg,
    progress=gr.Progress()
):
    try:
        valid_keyframes = [p for p in keyframe_images_state if p is not None and os.path.exists(p)]
        width, height = video_resolution, video_resolution
        video_total_frames_user = int(video_duration_seconds * video_fps)
        video_total_frames_ltx = int(round((float(video_total_frames_user) - 1.0) / 8.0) * 8 + 1)
        if not valid_keyframes or len(valid_keyframes) < 2: raise gr.Error("São necessários pelo menos 2 keyframes válidos para produzir uma transição.")
        if int(fragment_duration_frames) > video_total_frames_user: raise gr.Error(f"Duração do fragmento ({fragment_duration_frames}) não pode ser maior que a Duração Bruta ({video_total_frames_user}).")
        log_history = f"\n--- FASE 3/4: Iniciando Produção ({width}x{height})...\n"
        yield {
            production_log_output: log_history, video_gallery_output: [],
            prod_media_start_output: None, prod_media_mid_output: gr.update(visible=False), prod_media_end_output: None
        }
        seed = int(time.time()); video_fragments, story_history = [], ""; kinetic_memory_path = None
        num_transitions = len(valid_keyframes) - 1
        
        for i in range(num_transitions):
            fragment_num = i + 1
            progress(i / num_transitions, desc=f"Gerando Fragmento {fragment_num}...")
            log_history += f"\n--- FRAGMENTO {fragment_num}/{num_transitions} ---\n"
            destination_frame = int(video_total_frames_ltx - 1)
            
            if i == 0 or kinetic_memory_path is None:
                start_path, destination_path = valid_keyframes[i], valid_keyframes[i+1]
                dest_scene_desc = scene_storyboard[i]
                log_history += f"  - Início (Cena Nova): {os.path.basename(start_path)}\n  - Destino: {os.path.basename(destination_path)}\n"
                current_motion_prompt = get_initial_motion_prompt(prompt_geral, start_path, destination_path, dest_scene_desc)
                conditioning_items_data = [(start_path, 0, 1.0), (destination_path, destination_frame, dest_cond_strength)]
                transition_type = "continuous"
                yield { production_log_output: log_history, prod_media_start_output: start_path, prod_media_mid_output: gr.update(visible=False), prod_media_end_output: destination_path }
            else:
                memory_path, path_path, destination_path = kinetic_memory_path, valid_keyframes[i], valid_keyframes[i+1]
                path_scene_desc, dest_scene_desc = scene_storyboard[i-1], scene_storyboard[i]
                log_history += f"  - Diretor de Continuidade analisando...\n  - Memória: {os.path.basename(memory_path)}\n  - Caminho: {os.path.basename(path_path)}\n  - Destino: {os.path.basename(destination_path)}\n"
                yield { production_log_output: log_history, prod_media_start_output: gr.update(value=memory_path, visible=True), prod_media_mid_output: gr.update(value=path_path, visible=True), prod_media_end_output: destination_path }
                decision_data = get_transition_decision(prompt_geral, story_history, memory_path, path_path, destination_path, midpoint_scene_description=path_scene_desc, dest_scene_desc=dest_scene_desc)
                transition_type = decision_data["transition_type"]
                current_motion_prompt = decision_data["motion_prompt"]
                log_history += f"  - Decisão: {transition_type.upper()}\n"
                mid_cond_frame_calculated = int(video_total_frames_ltx - fragment_duration_frames + eco_video_frames)
                conditioning_items_data = [(memory_path, 0, 1.0), (path_path, mid_cond_frame_calculated, mid_cond_strength), (destination_path, destination_frame, dest_cond_strength)]

            story_history += f"\n- Ato {fragment_num + 1}: {current_motion_prompt}"
            log_history += f"  - Instrução do Cineasta: '{current_motion_prompt}'\n"; yield {production_log_output: log_history}
            
            output_filename = f"fragment_{fragment_num}_{transition_type}.mp4"
            full_fragment_path, _ = ltx_manager_singleton.generate_video_fragment(
                motion_prompt=current_motion_prompt, conditioning_items_data=conditioning_items_data,
                width=width, height=height, seed=seed, cfg=cfg, progress=progress,
                video_total_frames=video_total_frames_ltx, video_fps=video_fps,
                use_attention_slicing=use_attention_slicing, num_inference_steps=num_inference_steps,
                decode_timestep=decode_timestep, image_cond_noise_scale=image_cond_noise_scale,
                current_fragment_index=fragment_num, output_path=os.path.join(WORKSPACE_DIR, output_filename)
            )
            log_history += f"  - LOG: Gerei {output_filename}.\n"
            
            is_last_fragment = (i == num_transitions - 1)
            
            if is_last_fragment:
                log_history += "  - Último fragmento. Mantendo duração total.\n"
                video_fragments.append(full_fragment_path)
                kinetic_memory_path = None
            elif transition_type == "cut":
                log_history += "  - CORTE DE CENA: Fragmento mantido, memória reiniciada.\n"
                video_fragments.append(full_fragment_path)
                kinetic_memory_path = None
            else:
                trimmed_fragment_path = os.path.join(WORKSPACE_DIR, f"fragment_{fragment_num}_trimmed.mp4")
                trim_video_to_frames(full_fragment_path, trimmed_fragment_path, int(fragment_duration_frames))
                eco_output_path = os.path.join(WORKSPACE_DIR, f"eco_from_frag_{fragment_num}.mp4")
                kinetic_memory_path = extract_last_n_frames_as_video(trimmed_fragment_path, eco_output_path, int(eco_video_frames))
                video_fragments.append(full_fragment_path)
                log_history += f"  - CONTINUIDADE: Eco criado: {os.path.basename(kinetic_memory_path)}\n"

            yield {production_log_output: log_history, video_gallery_output: video_fragments}

        progress(1.0, desc="Produção dos fragmentos concluída.")
        log_history += "\nProdução de todos os fragmentos concluída. Pronto para montar o vídeo final.\n"
        yield {
            production_log_output: log_history, 
            video_gallery_output: video_fragments, 
            fragment_list_state: video_fragments
        }
    except Exception as e: raise gr.Error(f"A Produção de Vídeo (LTX) falhou: {e}")


def on_select_keyframe(keyframes_list, evt: gr.SelectData):
    selected_path = keyframes_list[evt.index]
    status_message = f"✅ Keyframe {evt.index + 1} selecionado ({os.path.basename(selected_path)}). Suba uma nova imagem abaixo para substituí-lo."
    return evt.index, status_message

def replace_selected_keyframe(keyframes_list, selected_index, new_image_file, keyframe_resolution):
    if selected_index is None: raise gr.Error("❌ Erro: Por favor, clique em um keyframe na galeria acima antes de subir uma nova imagem.")
    if not new_image_file: raise gr.Error("❌ Erro: Nenhum arquivo de imagem foi enviado.")
    new_image_path = new_image_file.name
    processed_new_path = process_image_to_square(
        new_image_path, 
        size=keyframe_resolution,
        output_filename=f"edited_keyframe_{selected_index}_{time.time()}.png"
    )
    updated_list = list(keyframes_list)
    updated_list[selected_index] = processed_new_path
    gr.Info(f"Keyframe {selected_index + 1} foi substituído com sucesso!")
    return updated_list, updated_list, "Pronto para a próxima edição."

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# NOVIM-12.0 (Painel de Controle do Diretor)\n*Arquitetura com Lógica de 'Cena Nova' Refinada*")

    if os.path.exists(WORKSPACE_DIR): shutil.rmtree(WORKSPACE_DIR)
    os.makedirs(WORKSPACE_DIR); Path("prompts").mkdir(exist_ok=True)

    scene_storyboard_state = gr.State([])
    keyframe_images_state = gr.State([])
    fragment_list_state = gr.State([])
    prompt_geral_state = gr.State("")
    processed_ref_path_state = gr.State("")
    fragment_duration_state = gr.State()
    eco_frames_state = gr.State()

    gr.Markdown("## CONFIGURAÇÕES GLOBAIS DE RESOLUÇÃO")
    with gr.Row():
        video_resolution_selector = gr.Radio([512, 720, 1024], value=512, label="Resolução de Geração do Vídeo (px)")
        keyframe_resolution_selector = gr.Radio([512, 720, 1024], value=512, label="Resolução dos Keyframes (px)")

    gr.Markdown("--- \n ## ETAPA 1: O ROTEIRO (IA Roteirista)")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Ideia Geral (Prompt)")
            num_fragments_input = gr.Slider(2, 50, 4, step=1, label="Nº de Keyframes a Gerar")
            image_input = gr.Image(type="filepath", label="Referência Inicial (será redimensionada)")
            director_button = gr.Button("▶️ 1. Gerar Roteiro", variant="primary")
        with gr.Column(scale=2): storyboard_to_show = gr.JSON(label="Roteiro de Cenas Gerado (em Inglês)")

    gr.Markdown("--- \n ## ETAPA 2: OS KEYFRAMES (IA Pintor & Diretor de Arte)")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(f"As imagens-chave serão geradas na resolução selecionada acima.")
            with gr.Row():
                with gr.Column():
                    ref1_image = gr.Image(label="Referência Inicial (Definida na Etapa 1)", type="filepath", interactive=False)
                    ref1_task = gr.Dropdown(choices=["ip", "id", "style"], value="ip", label="Tarefa da Ref. Inicial")
                with gr.Column():
                    ref2_image = gr.Image(label="Referência Secundária (Opcional)", type="filepath")
                    ref2_task = gr.Dropdown(choices=["ip", "id", "style"], value="style", label="Tarefa da Ref. Secundária")
            photographer_button = gr.Button("▶️ 2. Pintar Imagens-Chave em Cadeia", variant="primary")
        with gr.Column(scale=1): keyframe_log_output = gr.Textbox(label="Diário de Bordo do Pintor", lines=15, interactive=False)

    gr.Markdown("--- \n ## ETAPA 2.5: CURADORIA DOS KEYFRAMES (Editor de Cenas)")
    with gr.Column():
        keyframe_gallery_output = gr.Gallery(label="Galeria de Keyframes Gerados (Clique para substituir)", object_fit="contain", height="auto", type="filepath", interactive=True)
        with gr.Row():
            edit_status_textbox = gr.Textbox(label="Status da Edição", interactive=False, placeholder="Clique em uma imagem acima para começar...")
            replace_keyframe_button = gr.UploadButton("⬆️ Subir Imagem para Substituir o Keyframe Selecionado", file_types=["image"], variant="secondary")
    selected_keyframe_index_state = gr.State(None)

    gr.Markdown("--- \n ## ETAPA 3: A PRODUÇÃO (IA Cineasta & Câmera)")
    with gr.Row():
        with gr.Column(scale=1):
            cfg_slider = gr.Slider(0.5, 10.0, 1.0, step=0.1, label="CFG (Guidance Scale)")
            with gr.Accordion("Controles Avançados de Timing e Performance", open=False):
                video_duration_slider = gr.Slider(label="Duração da Geração Bruta (s)", minimum=2.0, maximum=10.0, value=6.0, step=0.5)
                video_fps_radio = gr.Radio(choices=[8, 16, 24, 32], value=24, label="FPS do Vídeo")
                num_inference_steps_slider = gr.Slider(label="Etapas de Inferência", minimum=4, maximum=20, value=10, step=1)
                slicing_checkbox = gr.Checkbox(label="Usar Attention Slicing (Economiza VRAM)", value=True)
                gr.Markdown("---"); gr.Markdown("#### Controles de Duração (Arquitetura Eco + Déjà Vu)")
                fragment_duration_slider = gr.Slider(label="Duração de Cada Fragmento (% da Geração Bruta)", minimum=1, maximum=100, value=75, step=1)
                eco_frames_slider = gr.Slider(label="Tamanho do Eco Cinético (Frames)", minimum=4, maximum=48, value=8, step=1)
                mid_cond_strength_slider = gr.Slider(label="Força do 'Caminho'", minimum=0.1, maximum=1.0, value=0.5, step=0.05)
                dest_cond_strength_slider = gr.Slider(label="Força do 'Destino'", minimum=0.1, maximum=1.0, value=1.0, step=0.05)
                gr.Markdown("---"); gr.Markdown("#### Controles do VAE (Avançado)")
                decode_timestep_slider = gr.Slider(label="VAE Decode Timestep", minimum=0.0, maximum=0.2, value=0.05, step=0.005)
                image_cond_noise_scale_slider = gr.Slider(label="VAE Image Cond Noise Scale", minimum=0.0, maximum=0.1, value=0.025, step=0.005)
                
            animator_button = gr.Button("▶️ 3. Produzir Cenas", variant="primary")
            with gr.Accordion("Visualização das Mídias de Condicionamento (Ao Vivo)", open=True):
                with gr.Row():
                    prod_media_start_output = gr.Video(label="Mídia Inicial (Eco/K1)", interactive=False)
                    prod_media_mid_output = gr.Image(label="Mídia do Caminho (K_i-1)", interactive=False, visible=False)
                    prod_media_end_output = gr.Image(label="Mídia de Destino (K_i)", interactive=False)
            production_log_output = gr.Textbox(label="Diário de Bordo da Produção", lines=10, interactive=False)
        with gr.Column(scale=1): video_gallery_output = gr.Gallery(label="Fragmentos Gerados", object_fit="contain", height="auto", type="video")

    gr.Markdown(f"--- \n ## ETAPA 4: PÓS-PRODUÇÃO (Montagem Final)")
    with gr.Row():
        with gr.Column():
            editor_button = gr.Button("▶️ 4. Montar Vídeo Final", variant="primary")
            final_video_output = gr.Video(label="A Obra-Prima Final")

    def process_and_update_storyboard(num_fragments, prompt, image_path, keyframe_resolution):
        processed_path = process_image_to_square(image_path, keyframe_resolution, f"initial_ref_{keyframe_resolution}x{keyframe_resolution}.png")
        if not processed_path: raise gr.Error("A imagem de referência é inválida ou não foi fornecida.")
        storyboard = run_storyboard_generation(num_fragments, prompt, processed_path)
        return storyboard, prompt, processed_path

    director_button.click(
        fn=process_and_update_storyboard,
        inputs=[num_fragments_input, prompt_input, image_input, keyframe_resolution_selector],
        outputs=[scene_storyboard_state, prompt_geral_state, processed_ref_path_state]
    ).success(fn=lambda s, p: (s, p), inputs=[scene_storyboard_state, processed_ref_path_state], outputs=[storyboard_to_show, ref1_image])

    @photographer_button.click(
        inputs=[scene_storyboard_state, ref1_image, ref1_task, ref2_image, ref2_task, keyframe_resolution_selector],
        outputs=[keyframe_log_output, keyframe_gallery_output, keyframe_images_state]
    )
    def run_keyframe_generation_wrapper(storyboard, ref1_img, ref1_tsk, ref2_img, ref2_tsk, keyframe_res, progress=gr.Progress()):
        ref_data = [{'image': ref1_img, 'task': ref1_tsk}, {'image': ref2_img, 'task': ref2_tsk}]
        yield from run_keyframe_generation(storyboard, ref_data, keyframe_res, progress)

    keyframe_gallery_output.select(fn=on_select_keyframe, inputs=[keyframe_images_state], outputs=[selected_keyframe_index_state, edit_status_textbox], api_name=False, queue=False)
    
    replace_keyframe_button.upload(
        fn=replace_selected_keyframe,
        inputs=[keyframe_images_state, selected_keyframe_index_state, replace_keyframe_button, keyframe_resolution_selector],
        outputs=[keyframe_images_state, keyframe_gallery_output, edit_status_textbox]
    )
    
    def updated_animator_click(
        video_resolution,
        video_duration_seconds, video_fps, eco_video_frames, use_attention_slicing,
        fragment_duration_percentage, mid_cond_strength, dest_cond_strength, num_inference_steps,
        decode_timestep, image_cond_noise_scale,
        prompt_geral, keyframe_images_state, scene_storyboard, cfg, progress=gr.Progress()):
        
        total_frames = video_duration_seconds * video_fps
        fragment_duration_in_frames = int(math.floor((fragment_duration_percentage / 100.0) * total_frames))
        fragment_duration_in_frames = max(1, fragment_duration_in_frames)

        for update in run_video_production(
            video_resolution,
            video_duration_seconds, video_fps, eco_video_frames, use_attention_slicing,
            fragment_duration_in_frames, mid_cond_strength, dest_cond_strength, num_inference_steps,
            decode_timestep, image_cond_noise_scale,
            prompt_geral, keyframe_images_state, scene_storyboard, cfg, progress):
            yield update
        
        yield {
            fragment_duration_state: fragment_duration_in_frames,
            eco_frames_state: eco_video_frames
        }

    animator_button.click(
        fn=updated_animator_click,
        inputs=[
            video_resolution_selector,
            video_duration_slider, video_fps_radio, eco_frames_slider, slicing_checkbox,
            fragment_duration_slider, mid_cond_strength_slider, dest_cond_strength_slider, num_inference_steps_slider,
            decode_timestep_slider, image_cond_noise_scale_slider,
            prompt_geral_state, keyframe_images_state, scene_storyboard_state, cfg_slider
        ],
        outputs=[
            production_log_output, video_gallery_output, fragment_list_state,
            prod_media_start_output, prod_media_mid_output, prod_media_end_output,
            fragment_duration_state, eco_frames_state
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