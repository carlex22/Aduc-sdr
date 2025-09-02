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
import logging
from typing import List, Dict, Any, Generator, Tuple

import gradio as gr
from PIL import Image, ImageOps

from deformes4D_engine import Deformes4DEngine
from ltx_manager_helpers import ltx_manager_singleton
from gemini_helpers import gemini_singleton
from image_specialist import image_specialist_singleton

# O logger é configurado no app.py, aqui apenas obtemos a instância.
logger = logging.getLogger(__name__)

class AducDirector:
    """
    Representa o Diretor de Cena, responsável pelo gerenciamento do estado da produção.
    Atua como a "partitura" da orquestra, mantendo o controle de todos os artefatos
    gerados (roteiro, keyframes, etc.) durante o processo criativo.
    """
    def __init__(self, workspace_dir: str):
        """
        Inicializa o Diretor, criando o diretório de trabalho.

        Args:
            workspace_dir (str): O caminho para o diretório onde todos os artefatos
                                 de geração serão armazenados.
        """
        self.workspace_dir = workspace_dir
        os.makedirs(self.workspace_dir, exist_ok=True)
        self.state: Dict[str, Any] = {}
        logger.info(f"O palco está pronto. Workspace em '{self.workspace_dir}'.")

    def update_state(self, key: str, value: Any) -> None:
        """
        Anota uma nova informação na "partitura", atualizando o estado da produção.

        Args:
            key (str): A chave para o estado a ser salvo (ex: "storyboard").
            value (Any): O valor do estado (ex: a lista de cenas do roteiro).
        """
        logger.info(f"Anotando na partitura: Estado '{key}' atualizado.")
        self.state[key] = value

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Consulta uma informação da "partitura", recuperando um estado salvo.

        Args:
            key (str): A chave do estado a ser recuperado.
            default (Any, optional): O valor a ser retornado se a chave não existir.

        Returns:
            Any: O valor do estado salvo ou o valor padrão.
        """
        return self.state.get(key, default)

class AducOrchestrator:
    """
    Implementa o Maestro (Γ), a camada central de orquestração da arquitetura ADUC.
    Ele não executa as tarefas de IA diretamente, mas delega cada etapa do processo
    criativo (roteiro, arte, cinematografia) aos Especialistas apropriados.
    """
    def __init__(self, workspace_dir: str):
        """
        Inicializa o Maestro e seus músicos (os especialistas de IA).

        Args:
            workspace_dir (str): O caminho para o diretório de trabalho, que será
                                 gerenciado pelo AducDirector.
        """
        self.director = AducDirector(workspace_dir)
        self.editor = Deformes4DEngine(ltx_manager_singleton, workspace_dir)
        self.painter = image_specialist_singleton
        logger.info("Maestro ADUC está no pódio. Músicos (especialistas) prontos.")

    def process_image_for_story(self, image_path: str, size: int, filename: str) -> str:
        """
        Pré-processa uma imagem de referência, padronizando-a para uso pelos Especialistas.
        Converte para RGB, redimensiona para um formato quadrado e salva no workspace.

        Args:
            image_path (str): Caminho da imagem original.
            size (int): Tamanho (largura e altura) da imagem final.
            filename (str): Nome do arquivo para salvar a imagem processada.

        Returns:
            str: O caminho para a imagem processada e salva.
        """
        img = Image.open(image_path).convert("RGB")
        img_square = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
        processed_path = os.path.join(self.director.workspace_dir, filename)
        img_square.save(processed_path)
        logger.info(f"Imagem de referência processada e salva em: {processed_path}")
        return processed_path

    def task_generate_storyboard(self, prompt: str, num_keyframes: int, ref_image_paths: List[str], 
                                 progress: gr.Progress) -> Tuple[List[str], str, Any]:
        """
        Delega ao Roteirista (Gemini) a tarefa de criar o roteiro (storyboard).

        Args:
            prompt (str): A ideia geral do filme fornecida pelo usuário.
            num_keyframes (int): O número de cenas a serem criadas.
            ref_image_paths (List[str]): Caminhos para as imagens de referência.
            progress (gr.Progress): Objeto do Gradio para reportar progresso.

        Returns:
            Tuple[List[str], str, Any]: O storyboard, o caminho da primeira imagem de referência,
                                        e um update para a UI do Gradio.
        """
        logger.info(f"Ato 1, Cena 1: Roteiro. Instruindo o Roteirista (Gemini) a criar {num_keyframes} cenas a partir de: '{prompt}'")
        progress(0.2, desc="Consultando Roteirista IA (Gemini)...")
        
        storyboard = gemini_singleton.generate_storyboard(prompt, num_keyframes, ref_image_paths)
        
        logger.info(f"Roteirista retornou a partitura: {storyboard}")
        self.director.update_state("storyboard", storyboard)
        self.director.update_state("processed_ref_paths", ref_image_paths)
        
        # Retorna o storyboard, a primeira imagem como referência inicial, e um comando
        # para tornar o próximo acordeão da UI visível.
        return storyboard, ref_image_paths[0], gr.update(visible=True, open=True)

    def task_select_keyframes(self, storyboard: List[str], base_ref_paths: List[str], 
                              pool_ref_paths: List[str]) -> List[str]:
        """
        Delega ao Editor/Fotógrafo (Gemini) a tarefa de selecionar as melhores imagens
        de um "banco de cenas" para corresponder ao roteiro (Modo Fotógrafo).

        Args:
            storyboard (List[str]): O roteiro gerado.
            base_ref_paths (List[str]): Imagens de referência principais.
            pool_ref_paths (List[str]): O conjunto de imagens para seleção.

        Returns:
            List[str]: Uma lista ordenada de caminhos de imagem selecionados como keyframes.
        """
        logger.info(f"Ato 1, Cena 2 (Modo Fotógrafo): Instruindo o Editor (Gemini) a selecionar {len(storyboard)} keyframes.")
        
        selected_paths = gemini_singleton.select_keyframes_from_pool(storyboard, base_ref_paths, pool_ref_paths)
        
        logger.info(f"Editor selecionou as seguintes cenas: {[os.path.basename(p) for p in selected_paths]}")
        self.director.update_state("keyframes", selected_paths)
        return selected_paths

    def task_generate_keyframes(self, storyboard: List[str], initial_ref_path: str, global_prompt: str, 
                                keyframe_resolution: int, progress_callback_factory=None) -> List[str]:
        """
        Delega ao Diretor de Arte (ImageSpecialist) a tarefa de gerar os keyframes
        visuais a partir do roteiro (Modo Diretor de Arte).

        Args:
            storyboard (List[str]): O roteiro gerado.
            initial_ref_path (str): A imagem inicial que serve como base para o primeiro keyframe.
            global_prompt (str): O prompt geral do filme.
            keyframe_resolution (int): A resolução (em pixels) para os keyframes gerados.
            progress_callback_factory (callable, optional): Uma função para criar callbacks de progresso.

        Returns:
            List[str]: Uma lista de caminhos para os keyframes gerados e salvos.
        """
        logger.info("Ato 1, Cena 2 (Modo Diretor de Arte): Delegando ao Especialista de Imagem.")
        
        general_ref_paths = self.director.get_state("processed_ref_paths", [])
        
        final_keyframes = self.painter.generate_keyframes_from_storyboard(
            storyboard=storyboard,
            initial_ref_path=initial_ref_path,
            global_prompt=global_prompt,
            keyframe_resolution=keyframe_resolution,
            general_ref_paths=general_ref_paths,
            progress_callback_factory=progress_callback_factory
        )
        
        self.director.update_state("keyframes", final_keyframes)
        logger.info("Maestro: Especialista de Imagem concluiu a geração dos keyframes.")
        return final_keyframes
    
    def task_produce_final_movie_with_feedback(self, keyframes: List[str], global_prompt: str, seconds_per_fragment: float, 
                                               trim_percent: int, handler_strength: float, 
                                               destination_convergence_strength: float, video_resolution: int, 
                                               use_continuity_director: bool, progress: gr.Progress) -> Generator[Dict[str, Any], None, None]:
        """
        Delega ao Editor/Cineasta (Deformes4DEngine) a produção final do filme,
        gerando e retornando feedback sobre cada fragmento criado.

        Args:
            (Vários): Parâmetros de controle para a geração de vídeo.
            progress (gr.Progress): Objeto do Gradio para reportar progresso.

        Yields:
            Generator[Dict[str, Any], None, None]: Dicionários contendo o caminho para
                                                   cada fragmento gerado ou o caminho final do filme.
        """
        logger.info("Maestro: Delegando a produção do filme completo ao Deformes4DEngine.")
        storyboard = self.director.get_state("storyboard", [])

        # O gerador do Deformes4DEngine retorna atualizações de progresso.
        for update in self.editor.generate_full_movie(
            keyframes=keyframes, 
            global_prompt=global_prompt, 
            storyboard=storyboard,
            seconds_per_fragment=seconds_per_fragment, 
            trim_percent=trim_percent,
            handler_strength=handler_strength, 
            destination_convergence_strength=destination_convergence_strength,
            video_resolution=video_resolution, 
            use_continuity_director=use_continuity_director, 
            progress=progress
        ):
            if "fragment_path" in update and update["fragment_path"]:
                yield {"fragment_path": update["fragment_path"]}
            elif "final_path" in update and update["final_path"]:
                final_movie_path = update["final_path"]
                self.director.update_state("final_video_path", final_movie_path)
                yield {"final_path": final_movie_path}
                break

        logger.info("Maestro: Produção do filme concluída e estado do diretor atualizado.")