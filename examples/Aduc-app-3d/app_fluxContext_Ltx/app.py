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

# --- app.py (ADUC-SDR-2.9: Diretor de Cena com Prompt Único e Extração) ---

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

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

from flux_kontext_helpers import flux_kontext_singleton
from ltx_manager_helpers import ltx_manager_singleton

WORKSPACE_DIR = "aduc_workspace"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ======================================================================================
# SEÇÃO 1: FUNÇÕES UTILITÁRIAS E DE PROCESSAMENTO DE MÍDIA
# ======================================================================================

def robust_json_parser(raw_text: str) -> dict:
    """
    Analisa uma string de texto bruto para encontrar e decodificar o primeiro objeto JSON válido.
    É essencial para extrair respostas estruturadas de modelos de linguagem.

    Args:
        raw_text (str): A string completa retornada pela IA.

    Returns:
        dict: Um dicionário Python representando o objeto JSON.
    
    Raises:
        ValueError: Se nenhum objeto JSON válido for encontrado ou a decodificação falhar.
    """
    clean_text = raw_text.strip()
    try:
        start_index = clean_text.find('{'); end_index = clean_text.rfind('}')
        if start_index != -1 and end_index != -1 and end_index > start_index:
            json_str = clean_text[start_index : end_index + 1]
            return json.loads(json_str)
        else: raise ValueError("Nenhum objeto JSON válido encontrado na resposta da IA.")
    except json.JSONDecodeError as e: raise ValueError(f"Falha ao decodificar JSON: {e}")

def process_image_to_square(image_path: str, size: int, output_filename: str = None) -> str:
    """
    Processa uma imagem para um formato quadrado, redimensionando e cortando centralmente.

    Args:
        image_path (str): Caminho para a imagem de entrada.
        size (int): A dimensão (altura e largura) da imagem de saída.
        output_filename (str, optional): Nome do arquivo de saída.

    Returns:
        str: O caminho para a imagem processada.
    """
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
    """
    Usa o FFmpeg para cortar um vídeo, mantendo um número específico de frames do início.

    Args:
        input_path (str): Caminho para o vídeo de entrada.
        output_path (str): Caminho para salvar o vídeo cortado.
        frames_to_keep (int): Número de frames a serem mantidos.

    Returns:
        str: O caminho para o vídeo cortado.
    """
    try:
        subprocess.run(f"ffmpeg -y -v error -i \"{input_path}\" -vf \"select='lt(n,{frames_to_keep})'\" -an \"{output_path}\"", shell=True, check=True, text=True)
        return output_path
    except subprocess.CalledProcessError as e: raise gr.Error(f"FFmpeg falhou ao cortar vídeo: {e.stderr}")

def extract_last_n_frames_as_video(input_path: str, output_path: str, n_frames: int) -> str:
    """
    Usa o FFmpeg para extrair os últimos N frames de um vídeo para criar o "Eco Cinético".

    Args:
        input_path (str): Caminho para o vídeo de entrada.
        output_path (str): Caminho para salvar o vídeo de saída (o eco).
        n_frames (int): Número de frames a serem extraídos do final.

    Returns:
        str: O caminho para o vídeo de eco gerado.
    """
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
    """
    Concatena os fragmentos de vídeo gerados em uma única "Obra-Prima" final.
    Fragmentos marcados como 'cut' (identificados pelo nome do arquivo)
    não terão sua duração cortada para preservar a intenção do corte.

    Args:
        fragment_paths (list): Lista de caminhos para os fragmentos de vídeo.
                                Cada caminho pode conter '_cut.mp4' no nome se for um corte.
        fragment_duration_frames (int): A duração esperada de cada clipe (usado apenas para
                                        fragmentos que NÃO são cortes).
        eco_video_frames (int): O tamanho da sobreposição que deve ser cortada para fragmentos
                                que NÃO são cortes (usado para o 'eco').
        progress (gr.Progress): Objeto do Gradio para atualizar a barra de progresso.

    Returns:
        str: O caminho para o vídeo final montado.
    """
    if not fragment_paths:
        raise gr.Error("Nenhum fragmento de vídeo para concatenar.")

    progress(0.1, desc="Preparando fragmentos para a montagem final...");

    try:
        list_file_path = os.path.abspath(os.path.join(WORKSPACE_DIR, f"concat_list_final_{time.time()}.txt"))
        final_output_path = os.path.abspath(os.path.join(WORKSPACE_DIR, "masterpiece_final.mp4"))
        temp_files_for_concat = []
        
        # Calculamos a duração a ser mantida APENAS para fragmentos que NÃO são cortes
        # Se for um corte, consideramos a duração total do fragmento original
        duration_for_non_cut_fragments = int(fragment_duration_frames - eco_video_frames)
        duration_for_non_cut_fragments = max(1, duration_for_non_cut_fragments) # Garantir que seja pelo menos 1 frame

        for i, p in enumerate(fragment_paths):
            is_last_fragment = (i == len(fragment_paths) - 1)
            
            # Verificamos se o nome do arquivo contém "_cut.mp4" para identificar um corte
            if "_cut.mp4" in os.path.basename(p) or is_last_fragment:
                # Se for um corte ou o último fragmento, usamos o arquivo original sem cortar o fim
                temp_files_for_concat.append(os.path.abspath(p))
                # Apenas para o último fragmento, garantimos que ele também seja considerado
                if is_last_fragment and "_cut.mp4" not in os.path.basename(p):
                    pass # O último fragmento original já foi adicionado
            else:
                # Para fragmentos que não são cortes e não são o último, cortamos o fim
                temp_path = os.path.join(WORKSPACE_DIR, f"final_temp_concat_{i}.mp4")
                # Aqui usamos a duração calculada para não-cortes (fragment_duration - eco)
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
        error_output = e.stderr if e.stderr else "Nenhuma saída de erro do FFmpeg."
        raise gr.Error(f"FFmpeg falhou na concatenação final: {error_output}")
    except Exception as e:
        raise gr.Error(f"Um erro ocorreu durante a concatenação final: {e}")

def concatenate_final_video1(fragment_paths: list, fragment_duration_frames: int, eco_video_frames: int, progress=gr.Progress()):
    """
    Concatena os fragmentos de vídeo gerados em uma única "Obra-Prima" final.

    Args:
        fragment_paths (list): Lista de caminhos para os fragmentos de vídeo.
        fragment_duration_frames (int): A duração de cada clipe na montagem final.
        eco_video_frames (int): O tamanho da sobreposição que deve ser cortada.
        progress (gr.Progress): Objeto do Gradio para atualizar a barra de progresso.

    Returns:
        str: O caminho para o vídeo final montado.
    """
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
    """
    Extrai metadados EXIF relevantes de uma imagem.

    Args:
        image_path (str): O caminho para o arquivo de imagem.

    Returns:
        str: Uma string formatada contendo os metadados EXIF.
    """
    try:
        img = Image.open(image_path); exif_data = img._getexif()
        if not exif_data: return "No EXIF metadata found."
        exif = { ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS }
        relevant_tags = ['DateTimeOriginal', 'Model', 'LensModel', 'FNumber', 'ExposureTime', 'ISOSpeedRatings', 'FocalLength']
        metadata_str = ", ".join(f"{key}: {exif[key]}" for key in relevant_tags if key in exif)
        return metadata_str if metadata_str else "No relevant EXIF metadata found."
    except Exception: return "Could not read EXIF data."

# ======================================================================================
# SEÇÃO 2: ORQUESTRADORES DE IA (As "Etapas" da Geração)
# ======================================================================================

def run_storyboard_generation(num_fragments: int, prompt: str, reference_paths: list):
    """
    Orquestra a Etapa 1: O Roteiro.
    Chama a IA (Gemini) para atuar como "Roteirista", analisando o prompt do usuário e
    todas as imagens de referência para criar uma narrativa coesa dividida em atos.

    Args:
        num_fragments (int): O número de keyframes (atos) a serem gerados no roteiro.
        prompt (str): A ideia geral do usuário.
        reference_paths (list): Lista de caminhos para todas as imagens de referência fornecidas.

    Returns:
        list: Uma lista de strings, onde cada string é a descrição de uma cena.
    """
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
    """
    Orquestra a Etapa 2: Os Keyframes.
    A cada iteração, chama a IA (Gemini) para atuar como "Diretor de Cena". A IA analisa
    o roteiro, as referências fixas e as últimas 3 imagens geradas para criar um prompt
    de composição. O prompt usa tags [IMG-X] para referenciar as fontes, que são então
    mapeadas para os arquivos reais e enviadas ao `FluxKontext` para a geração da imagem.

    Args:
        storyboard (list): A lista de atos do roteiro.
        fixed_reference_paths (list): Lista de caminhos para as imagens de referência fixas.
        keyframe_resolution (int): A resolução para os keyframes a serem gerados.
        global_prompt (str): A ideia geral do usuário para dar contexto à IA.
        progress (gr.Progress): Objeto do Gradio para a barra de progresso.

    Yields:
        dict: Atualizações para os componentes da UI do Gradio durante a geração.
    """
    if not storyboard: raise gr.Error("Nenhum roteiro para gerar keyframes.")
    if not fixed_reference_paths: raise gr.Error("A imagem de referência inicial é obrigatória.")
    
    initial_ref_image_path = fixed_reference_paths[0]
    log_history = ""; generated_images_for_gallery = []
    width, height = keyframe_resolution, keyframe_resolution
    
    keyframe_paths_for_video = [] 
    scene_history = "N/A"

    wrapper_prompt_path = os.path.join(os.path.dirname(__file__), "prompts/flux_composition_wrapper_prompt.txt")
    with open(wrapper_prompt_path, "r", encoding="utf-8") as f:
        kontext_template = f.read()
        
    director_prompt_path = os.path.join(os.path.dirname(__file__), "prompts/director_composition_prompt.txt")
    with open(director_prompt_path, "r", encoding="utf-8") as f:
        director_template = f.read()

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-2.5-flash')

        for i, scene_description in enumerate(storyboard):
            progress(i / len(storyboard), desc=f"Compondo Keyframe {i+1}/{len(storyboard)} ({width}x{height})")
            log_history += f"\n--- COMPONDO KEYFRAME {i+1}/{len(storyboard)} ---\n"
            
            last_three_paths = ([initial_ref_image_path] + keyframe_paths_for_video)[-3:]
            
            log_history += f"  - Diretor de Cena está analisando o contexto...\n"
            yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=generated_images_for_gallery), keyframe_images_state: gr.update(value=generated_images_for_gallery)}
            
            director_prompt = director_template.format(
                global_prompt=global_prompt,
                scene_history=scene_history,
                current_scene_desc=scene_description,
            )
            
            model_contents = []
            image_map = {}
            current_image_index = 1
            
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
            if not current_reference_paths:
                current_reference_paths = [last_three_paths[-1]]

            reference_images_pil = [Image.open(p) for p in current_reference_paths]
            final_kontext_prompt = re.sub(r'\[IMG-\d+\]', '', composition_prompt_with_tags).strip()
            
            log_history += f"  - Diretor de Cena decidiu usar as imagens: {[os.path.basename(p) for p in current_reference_paths]}\n"
            log_history += f"  - Prompt Final do Diretor: \"{final_kontext_prompt}\"\n"
            scene_history += f"Scene {i+1}: {final_kontext_prompt}\n"
            
            yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: gr.update(value=generated_images_for_gallery), keyframe_images_state: gr.update(value=generated_images_for_gallery)}

            final_kontext_prompt_wrapped = kontext_template.format(target_prompt=final_kontext_prompt)
            output_path = os.path.join(WORKSPACE_DIR, f"keyframe_{i+1}.png")
            
            image = flux_kontext_singleton.generate_image(
                reference_images=reference_images_pil, 
                prompt=final_kontext_prompt_wrapped, 
                width=width, height=height, seed=int(time.time())
            )
            
            image.save(output_path)
            keyframe_paths_for_video.append(output_path)
            generated_images_for_gallery.append(output_path)

    except Exception as e: 
        raise gr.Error(f"O Compositor (FluxKontext) ou o Diretor de Cena (Gemini) falhou: {e}")
            
    log_history += "\nComposição de todos os keyframes concluída.\n"
    final_keyframes = keyframe_paths_for_video
    yield {keyframe_log_output: gr.update(value=log_history), keyframe_gallery_output: final_keyframes, keyframe_images_state: final_keyframes}

def get_initial_motion_prompt(user_prompt: str, start_image_path: str, destination_image_path: str, dest_scene_desc: str):
    """
    Chama a IA (Gemini) para atuar como "Cineasta Inicial".
    Gera o prompt de movimento para o primeiro fragmento de vídeo, que não possui um eco anterior.

    Args:
        user_prompt (str): A ideia geral da história.
        start_image_path (str): Caminho para o primeiro keyframe.
        destination_image_path (str): Caminho para o segundo keyframe.
        dest_scene_desc (str): A descrição do roteiro para a cena de destino.

    Returns:
        str: O prompt de movimento gerado.
    """
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
    """
    Chama a IA (Gemini) para atuar como "Diretor de Continuidade".
    Analisa o eco, o keyframe atual e o próximo para decidir entre uma transição contínua
    ou um corte de cena, e gera o prompt de movimento apropriado.

    Args:
        (Vários argumentos de contexto sobre a história e as imagens)

    Returns:
        dict: Um dicionário contendo 'transition_type' e 'motion_prompt'.
    """
    if not GEMINI_API_KEY: raise gr.Error("Chave da API Gemini não configurada!")
    try:
        genai.configure(api_key=GEMINI_API_KEY); model = genai.GenerativeModel('gemini-2.5-flash'); prompt_file = "prompts/transition_decision_prompt.txt"
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

def run_video_production(
    video_resolution,
    video_duration_seconds, video_fps, eco_video_frames, use_attention_slicing,
    fragment_duration_frames, mid_cond_strength, dest_cond_strength, num_inference_steps,
    decode_timestep, image_cond_noise_scale,
    prompt_geral, keyframe_images_state, scene_storyboard, cfg,
    progress=gr.Progress()
):
    """
    Orquestra a Etapa 3: A Produção.
    Itera sobre os keyframes e chama os cineastas de IA para gerar os fragmentos de vídeo.

    Args:
        (Vários parâmetros da UI para controlar a geração de vídeo)

    Yields:
        dict: Atualizações para os componentes da UI do Gradio.
    """
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

# ======================================================================================
# SEÇÃO 3: DEFINIÇÃO DA INTERFACE GRÁFICA (UI com Gradio)
# ======================================================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# NOVIM-13.1 (Painel de Controle do Diretor)\n*Arquitetura ADUC-SDR com Documentação Completa*")

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
        video_resolution_selector = gr.Radio([512, 720, 1024], value=512, label="Resolução de Geração do Vídeo (px)")
        keyframe_resolution_selector = gr.Radio([512, 720, 1024], value=512, label="Resolução dos Keyframes (px)")

    gr.Markdown("--- \n ## ETAPA 1: O ROTEIRO (IA Roteirista)")
    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(label="Ideia Geral (Prompt)")
            num_fragments_input = gr.Slider(2, 50, 4, step=1, label="Nº de Keyframes a Gerar")
            reference_gallery = gr.Gallery(
                label="Imagens de Referência (A primeira é a principal)",
                type="filepath",
                columns=4, rows=1, object_fit="contain", height="auto"
            )
            director_button = gr.Button("▶️ 1. Gerar Roteiro", variant="primary")
        with gr.Column(scale=2): storyboard_to_show = gr.JSON(label="Roteiro de Cenas Gerado (em Inglês)")

    gr.Markdown("--- \n ## ETAPA 2: OS KEYFRAMES (IA Compositor & Diretor de Cena)")
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("O Diretor de Cena IA irá analisar as referências e o roteiro para compor cada keyframe de forma autônoma.")
            photographer_button = gr.Button("▶️ 2. Compor Imagens-Chave em Cadeia", variant="primary")
            keyframe_gallery_output = gr.Gallery(label="Galeria de Keyframes Gerados", object_fit="contain", height="auto", type="filepath", interactive=False)
        with gr.Column(scale=1): 
            keyframe_log_output = gr.Textbox(label="Diário de Bordo do Compositor", lines=25, interactive=False)

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

    gr.Markdown(
        """
        ---
        ### A Arquitetura: ADUC-SDR
        **ADUC (Arquitetura de Unificação Compositiva):** O sistema não usa um único modelo, mas uma equipe de IAs especializadas. Um **Roteirista** cria a história. Um **Diretor de Cena** decide a composição de cada keyframe, selecionando elementos de um "álbum" de referências visuais. Um **Compositor** (`FluxKontext`) cria as imagens.
        
        **SDR (Escala Dinâmica e Resiliente):** A geração de vídeo é dividida em fragmentos, permitindo criar vídeos de longa duração. A continuidade é garantida pela arquitetura **Eco + Déjà Vu**:
        - **O Eco:** Os últimos frames de um clipe são passados para o próximo, transferindo o *momentum* físico e a iluminação.
        - **O Déjà Vu:** Uma IA **Cineasta** analisa o Eco e os keyframes futuros para criar uma instrução de movimento que seja ao mesmo tempo contínua e narrativamente coerente, sabendo até quando realizar um corte de cena.
        """
    )
    # --- Lógica de Conexão dos Componentes ---
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

    director_button.click(
        fn=process_and_run_storyboard,
        inputs=[num_fragments_input, prompt_input, reference_gallery, keyframe_resolution_selector],
        outputs=[scene_storyboard_state, prompt_geral_state, processed_ref_paths_state]
    ).success(fn=lambda s: s, inputs=[scene_storyboard_state], outputs=[storyboard_to_show])

    photographer_button.click(
        fn=run_keyframe_generation,
        inputs=[scene_storyboard_state, processed_ref_paths_state, keyframe_resolution_selector, prompt_geral_state],
        outputs=[keyframe_log_output, keyframe_gallery_output, keyframe_images_state]
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