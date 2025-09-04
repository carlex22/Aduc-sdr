# 🛠️ managers/ - Ferramentas de IA de Terceiros para orquestração ADUC-SDR

Esta pasta contém implementações adaptadas de modelos e utilitários de IA de terceiros, que servem como "especialistas" ou "ferramentas" de baixo nível para a arquitetura ADUC-SDR.

**IMPORTANTE:** O conteúdo desta pasta é de autoria de seus respectivos idealizadores e desenvolvedores originais. Esta pasta **NÃO FAZ PARTE** do projeto principal ADUC-SDR em termos de sua arquitetura inovadora. Ela serve como um repositório para as **dependências diretas e modificadas** que os `Deformes enginers` (os estágios do "foguete" ADUC-SDR) invocam para realizar tarefas específicas (geração de imagem, vídeo, áudio).

As modificações realizadas nos arquivos aqui presentes visam principalmente:
1.  **Adaptação de Interfaces:** Padronizar as interfaces para que se encaixem no fluxo de orquestração do ADUC-SDR.
2.  **Gerenciamento de Recursos:** Integrar lógicas de carregamento/descarregamento de modelos (GPU management) e configurações via arquivos YAML.
3.  **Otimização de Fluxo:** Ajustar as pipelines para aceitar formatos de entrada mais eficientes (ex: tensores pré-codificados em vez de caminhos de mídia, pulando etapas de codificação/decodificação redundantes).

---

## 📄 Licenciamento

O conteúdo original dos projetos listados abaixo é licenciado sob a **Licença Apache 2.0**, ou outra licença especificada pelos autores originais. Todas as modificações e o uso desses arquivos dentro da estrutura `helpers/` do projeto ADUC-SDR estão em conformidade com os termos da **Licença Apache 2.0**.

As licenças originais dos projetos podem ser encontradas nas suas respectivas fontes ou nos subdiretórios `incl_licenses/` dentro de cada módulo adaptado.

---

## 🛠️ API dos Helpers e Guia de Uso

Esta seção detalha como cada helper (agente especialista) deve ser utilizado dentro do ecossistema ADUC-SDR. Todos os agentes são instanciados como **singletons** no `hardware_manager.py` para garantir o gerenciamento centralizado de recursos de GPU.

### **gemini_helpers.py (GeminiAgent)**

*   **Propósito:** Atua como o "Oráculo de Síntese Adaptativo", responsável por todas as tarefas de processamento de linguagem natural, como criação de storyboards, geração de prompts, e tomada de decisões narrativas.
*   **Singleton Instance:** `gemini_agent_singleton`
*   **Construtor:** `GeminiAgent()`
    *   Lê `configs/gemini_config.yaml` para obter o nome do modelo, parâmetros de inferência e caminhos de templates de prompt. A chave da API é lida da variável de ambiente `GEMINI_API_KEY`.
*   **Métodos Públicos:**
    *   `generate_storyboard(prompt: str, num_keyframes: int, ref_image_paths: list[str])`
        *   **Inputs:**
            *   `prompt`: A ideia geral do filme (string).
            *   `num_keyframes`: O número de cenas a serem geradas (int).
            *   `ref_image_paths`: Lista de caminhos para as imagens de referência (list[str]).
        *   **Output:** `tuple[list[str], str]` (Uma tupla contendo a lista de strings do storyboard e um relatório textual da operação).
    *   `select_keyframes_from_pool(storyboard: list, base_image_paths: list[str], pool_image_paths: list[str])`
        *   **Inputs:**
            *   `storyboard`: A lista de strings do storyboard gerado.
            *   `base_image_paths`: Imagens de referência base (list[str]).
            *   `pool_image_paths`: O "banco de imagens" de onde selecionar (list[str]).
        *   **Output:** `tuple[list[str], str]` (Uma tupla contendo a lista de caminhos de imagens selecionadas e um relatório textual).
    *   `get_anticipatory_keyframe_prompt(...)`
        *   **Inputs:** Contexto narrativo e visual para gerar um prompt de imagem.
        *   **Output:** `tuple[str, str]` (Uma tupla contendo o prompt gerado para o modelo de imagem e um relatório textual).
    *   `get_initial_motion_prompt(...)`
        *   **Inputs:** Contexto narrativo e visual para a primeira transição de vídeo.
        *   **Output:** `tuple[str, str]` (Uma tupla contendo o prompt de movimento gerado e um relatório textual).
    *   `get_transition_decision(...)`
        *   **Inputs:** Contexto narrativo e visual para uma transição de vídeo intermediária.
        *   **Output:** `tuple[dict, str]` (Uma tupla contendo um dicionário `{"transition_type": "...", "motion_prompt": "..."}` e um relatório textual).
    *   `generate_audio_prompts(...)`
        *   **Inputs:** Contexto narrativo global.
        *   **Output:** `tuple[dict, str]` (Uma tupla contendo um dicionário `{"music_prompt": "...", "sfx_prompt": "..."}` e um relatório textual).

### **flux_kontext_helpers.py (FluxPoolManager)**

*   **Propósito:** Especialista em geração de imagens de alta qualidade (keyframes) usando a pipeline FluxKontext. Gerencia um pool de workers para otimizar o uso de múltiplas GPUs.
*   **Singleton Instance:** `flux_kontext_singleton`
*   **Construtor:** `FluxPoolManager(device_ids: list[str], flux_config_file: str)`
    *   Lê `configs/flux_config.yaml`.
*   **Método Público:**
    *   `generate_image(prompt: str, reference_images: list[Image.Image], width: int, height: int, seed: int = 42, callback: callable = None)`
        *   **Inputs:**
            *   `prompt`: Prompt textual para guiar a geração (string).
            *   `reference_images`: Lista de objetos `PIL.Image` como referência visual.
            *   `width`, `height`: Dimensões da imagem de saída (int).
            *   `seed`: Semente para reprodutibilidade (int).
            *   `callback`: Função de callback opcional para monitorar o progresso.
        *   **Output:** `PIL.Image.Image` (O objeto da imagem gerada).

### **dreamo_helpers.py (DreamOAgent)**

*   **Propósito:** Especialista em geração de imagens de alta qualidade (keyframes) usando a pipeline DreamO, com capacidades avançadas de edição e estilo a partir de referências.
*   **Singleton Instance:** `dreamo_agent_singleton`
*   **Construtor:** `DreamOAgent(device_id: str = None)`
    *   Lê `configs/dreamo_config.yaml`.
*   **Método Público:**
    *   `generate_image(prompt: str, reference_images: list[Image.Image], width: int, height: int)`
        *   **Inputs:**
            *   `prompt`: Prompt textual para guiar a geração (string).
            *   `reference_images`: Lista de objetos `PIL.Image` como referência visual. A lógica interna atribui a primeira imagem como `style` e as demais como `ip`.
            *   `width`, `height`: Dimensões da imagem de saída (int).
        *   **Output:** `PIL.Image.Image` (O objeto da imagem gerada).

### **ltx_manager_helpers.py (LtxPoolManager)**

*   **Propósito:** Especialista na geração de fragmentos de vídeo no espaço latente usando a pipeline LTX-Video. Gerencia um pool de workers para otimizar o uso de múltiplas GPUs.
*   **Singleton Instance:** `ltx_manager_singleton`
*   **Construtor:** `LtxPoolManager(device_ids: list[str], ltx_model_config_file: str, ltx_global_config_file: str)`
    *   Lê o `ltx_global_config_file` e o `ltx_model_config_file` para configurar a pipeline.
*   **Método Público:**
    *   `generate_latent_fragment(**kwargs)`
        *   **Inputs:** Dicionário de keyword arguments (`kwargs`) contendo todos os parâmetros da pipeline LTX, incluindo:
            *   `height`, `width`: Dimensões do vídeo (int).
            *   `video_total_frames`: Número total de frames a serem gerados (int).
            *   `video_fps`: Frames por segundo (int).
            *   `motion_prompt`: Prompt de movimento (string).
            *   `conditioning_items_data`: Lista de objetos `LatentConditioningItem` contendo os tensores latentes de condição.
            *   `guidance_scale`, `stg_scale`, `num_inference_steps`, etc.
        *   **Output:** `tuple[torch.Tensor, tuple]` (Uma tupla contendo o tensor latente gerado e os valores de padding utilizados).

### **mmaudio_helper.py (MMAudioAgent)**

*   **Propósito:** Especialista em geração de áudio para um determinado fragmento de vídeo.
*   **Singleton Instance:** `mmaudio_agent_singleton`
*   **Construtor:** `MMAudioAgent(workspace_dir: str, device_id: str = None, mmaudio_config_file: str)`
    *   Lê `configs/mmaudio_config.yaml`.
*   **Método Público:**
    *   `generate_audio_for_video(video_path: str, prompt: str, negative_prompt: str, duration_seconds: float)`
        *   **Inputs:**
            *   `video_path`: Caminho para o arquivo de vídeo silencioso (string).
            *   `prompt`: Prompt textual para guiar a geração de áudio (string).
            *   `negative_prompt`: Prompt negativo para áudio (string).
            *   `duration_seconds`: Duração exata do vídeo (float).
        *   **Output:** `str` (O caminho para o novo arquivo de vídeo com a faixa de áudio integrada).


### **seedvr_helpers.py (SeedVrManager)**

*   **Propósito:** Especialista em pós-produção de vídeo, aplicando super-resolução com IA (`Video Super-Resolution`) para adicionar detalhes finos, nitidez e texturas realistas a um vídeo já renderizado.
*   **Singleton Instance:** `seedvr_manager_singleton`
*   **Construtor:** `SeedVrManager(workspace_dir: str, device_id: str = None)`
    *   Lê `configs/seedvr_config.yaml`.
*   **Método Público:**
    *   `process_video(input_video_path: str, output_video_path: str, prompt: str, model_version: str = '7B', steps: int = 100, seed: int = 666)`
        *   **Inputs:**
            *   `input_video_path`: Caminho para o vídeo de entrada a ser aprimorado (string).
            *   `output_video_path`: Caminho onde o vídeo finalizado será salvo (string).
            *   `prompt`: Um prompt de estilo geral para guiar o aprimoramento (string).
            *   `model_version`: A versão do modelo a ser usada, '3B' ou '7B' (string).
            *   `steps`: Número de passos de inferência para o processo de aprimoramento (int).
            *   `seed`: Semente para reprodutibilidade (int).
        *   **Output:** `str` (O caminho para o vídeo finalizado em alta definição).

---

## 🔗 Projetos Originais e Atribuições
(A seção de atribuições e licenças permanece a mesma que definimos anteriormente)

### DreamO
*   **Repositório Original:** [https://github.com/bytedance/DreamO](https://github.com/bytedance/DreamO)
...

### LTX-Video
*   **Repositório Original:** [https://github.com/Lightricks/LTX-Video](https://github.com/Lightricks/LTX-Video)
...

### MMAudio
*   **Repositório Original:** [https://github.com/hkchengrex/MMAudio](https://github.com/hkchengrex/MMAudio)
...

### SeedVr
*   **Repositório Original:** [https://github.com/ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR)