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

from PIL import Image
import os
import time
import logging
import gradio as gr
import yaml

from flux_kontext_helpers import flux_kontext_singleton
from gemini_helpers import gemini_singleton

logger = logging.getLogger(__name__)

class ImageSpecialist:
    """
    Especialista ADUC para a geração de imagens estáticas (keyframes).
    É responsável por todo o processo de transformar um roteiro em uma galeria de keyframes.
    """
    def __init__(self, workspace_dir):
        self.workspace_dir = workspace_dir
        self.image_generation_helper = flux_kontext_singleton
        logger.info("Especialista de Imagem (Flux) pronto para receber ordens do Maestro.")

    def _generate_single_keyframe(self, prompt: str, reference_images: list[Image.Image], output_filename: str, width: int, height: int, callback: callable = None) -> str:
        """
        Função de baixo nível que gera uma única imagem.
        """
        logger.info(f"Gerando keyframe '{output_filename}' com prompt: '{prompt}'")
        generated_image = self.image_generation_helper.generate_image(
            reference_images=reference_images, prompt=prompt, width=width,
            height=height, seed=int(time.time()), callback=callback
        )
        final_path = os.path.join(self.workspace_dir, output_filename)
        generated_image.save(final_path)
        logger.info(f"Keyframe salvo com sucesso em: {final_path}")
        return final_path

    def generate_keyframes_from_storyboard(self, storyboard: list, initial_ref_path: str, global_prompt: str, keyframe_resolution: int, general_ref_paths: list, progress_callback_factory: callable = None):
        """
        Orquestra a geração de todos os keyframes a partir de um storyboard.
        """
        current_base_image_path = initial_ref_path
        previous_prompt = "N/A (imagem inicial de referência)"
        final_keyframes = [current_base_image_path]
        width, height = keyframe_resolution, keyframe_resolution
        
        # O número de keyframes a gerar é len(storyboard) - 1, pois o primeiro keyframe já existe (initial_ref_path)
        # E o storyboard tem o mesmo número de elementos que o número total de keyframes desejados.
        num_keyframes_to_generate = len(storyboard) - 1
        
        logger.info(f"ESPECIALISTA DE IMAGEM: Recebi ordem para gerar {num_keyframes_to_generate} keyframes.")

        for i in range(num_keyframes_to_generate):
            # A cena atual é a transição de storyboard[i] para storyboard[i+1]
            current_scene = storyboard[i]
            future_scene = storyboard[i+1]
            progress_callback = progress_callback_factory(i + 1, num_keyframes_to_generate) if progress_callback_factory else None
            
            logger.info(f"--> Gerando Keyframe {i+1}/{num_keyframes_to_generate}...")
            
            # O próprio especialista consulta o Gemini para o prompt de imagem
            new_flux_prompt = gemini_singleton.get_anticipatory_keyframe_prompt(
                global_prompt=global_prompt, scene_history=previous_prompt,
                current_scene_desc=current_scene, future_scene_desc=future_scene,
                last_image_path=current_base_image_path, fixed_ref_paths=general_ref_paths
            )
            
            images_for_flux_paths = list(set([current_base_image_path] + general_ref_paths))
            images_for_flux = [Image.open(p) for p in images_for_flux_paths]
            
            new_keyframe_path = self._generate_single_keyframe(
                prompt=new_flux_prompt, reference_images=images_for_flux,
                output_filename=f"keyframe_{i+1}.png", width=width, height=height,
                callback=progress_callback
            )

            final_keyframes.append(new_keyframe_path)
            current_base_image_path = new_keyframe_path
            previous_prompt = new_flux_prompt
            
        logger.info(f"ESPECIALISTA DE IMAGEM: Geração de keyframes concluída.")
        return final_keyframes

# Singleton instantiation - usa o workspace_dir da config
try:
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    WORKSPACE_DIR = config['application']['workspace_dir']
    image_specialist_singleton = ImageSpecialist(workspace_dir=WORKSPACE_DIR)
except Exception as e:
    logger.error(f"Não foi possível inicializar o ImageSpecialist: {e}", exc_info=True)
    image_specialist_singleton = None