# Aduc-sdr: Arquitetura de Unificação Compositiva

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Carlexxx/Novinho)

**Aduc-sdr** é uma implementação aberta e funcional da **Arquitetura de Unificação Compositiva (ADUC) - Morte, Destilação e Renascimento (SDR)**. Este projeto apresenta um pipeline completo para a geração de vídeo por IA com foco em alta coerência temporal e continuidade narrativa, resolvendo um dos maiores desafios da área.

Em vez de um processo monolítico, a geração opera em um ciclo causal, onde a "alma" de cada clipe gerado é destilada e usada para informar o nascimento do próximo.

---

## Resultados em Destaque

Abaixo estão alguns exemplos gerados pela arquitetura ADUC-SDR. Cada vídeo é composto por múltiplos fragmentos, unidos de forma coerente através do método do "eco causal".

*(**Nota:** Para que os players de vídeo abaixo funcionem, você precisa fazer o upload dos seus arquivos de vídeo para uma pasta `examples` neste repositório e nomeá-los como `exemplo_01.mp4`, `exemplo_02.mp4`, etc.)*

| Exemplo 1: O Robô e o Trem | Exemplo 2: [Adicione um Título] |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <video src="https://github.com/carlex22/Aduc-sdr/raw/main/examples/exemplo_01.mp4" controls="controls" style="max-width: 400px;"></video> | <video src="https://github.com/carlex22/Aduc-sdr/raw/main/examples/exemplo_02.mp4" controls="controls" style="max-width: 400px;"></video> |

---

## A Arquitetura ADUC-SDR: O Esquema Matemático

A geração de vídeo é governada por uma função seccional que define como cada fragmento (`V_i`) é criado, operando em dois regimes distintos: o **"Gênesis"** da história e a **"Cadeia Causal"** que se segue.

---

#### **FÓRMULA 1: O FRAGMENTO INICIAL (Gênesis, `i=1`)**
*Define a criação do primeiro clipe a partir de imagens estáticas.*

**Planejamento:** `P_1 = Γ_initial( K_1, K_2, P_geral )`
        
**Execução:** `V_1 = Ψ( { (K_1, F_start), (K_2, F_end) }, P_1 )`

---
#### **FÓRMULA 2: A CADEIA CAUSAL (Momentum, `i > 1`)**
*Define a criação dos fragmentos subsequentes, garantindo a continuidade através do "eco".*

**Destilação:** `C_(i-1) = Δ(V_(i-1))`

**Planejamento:** `P_i = Γ_transition( C_(i-1), K_(i+1), P_geral, H_(i-1) )`

**Execução:** `V_i = Ψ( { (C_(i-1), F_start), (K_(i+1), F_end) }, P_i )`

---
#### **Componentes (O Léxico da Arquitetura):**
- **`V_i`**: Fragmento de Vídeo
- **`K_i`**: Keyframe (Imagem Estática)
- **`C_i`**: "Eco" Causal (Clipe de Vídeo ou Vetor de Frames)
- **`P_i`**: Prompt de Movimento
- **`P_geral`**: Prompt Geral (Intenção do Diretor)
- **`H_i`**: Histórico Narrativo
- **`Γ`**: Cineasta (Gerador de Prompt, ex: Gemini)
- **`Ψ`**: Câmera (Gerador de Vídeo, ex: LTX)
- **`Δ`**: Editor (Extrator de "Eco", ex: FFmpeg)
- **`F_start`, `F_end`**: Constantes de Frame (Âncoras Temporais)

---

## Como Funciona: O Pipeline de Produção

O sistema imita um estúdio de cinema de IA com especialistas para cada etapa:

1.  **O Roteiro (Sonhador):** A partir de uma ideia geral e uma imagem de referência, o Gemini (`photographer_prompt`) cria um roteiro visual (storyboard) com `N` cenas.

2.  **Os Keyframes (Pintor):** O DreamO (`run_keyframe_generation`) pinta os `N` keyframes. O primeiro é baseado na imagem do usuário, e os seguintes são gerados em cadeia, usando o keyframe anterior como referência para manter a consistência.

3.  **A Produção (Cineasta e Câmera):** Esta é a fase ADUC-SDR, que gera `N-1` fragmentos de vídeo.
    -   **Fragmento 1:** O Cineasta (Gemini com `director_motion_prompt.txt`) planeja a transição de `K1` para `K2`. A Câmera (LTX) filma.
    -   **Fragmentos Seguintes:** O Editor (FFmpeg) extrai o "Eco" do vídeo anterior. O Cineasta (Gemini com `director_motion_prompt_vector.txt`) analisa o "Eco" e o próximo keyframe para planejar um movimento que continue a inércia. A Câmera (LTX) filma.

4.  **Pós-Produção (Editor):** O Editor (FFmpeg) apara a sobreposição dos "ecos" de cada fragmento e une tudo em um único vídeo final coerente.

## Como Usar

1.  **Clonar o Repositório:**
    ```bash
    git clone https://github.com/carlex22/Aduc-sdr.git
    cd Aduc-sdr
    ```

2.  **Instalar Dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configurar a Chave da API:**
    Exporte sua chave da API do Google AI Studio como uma variável de ambiente.
    ```bash
    export GEMINI_API_KEY='SUA_CHAVE_API_AQUI'
    ```

4.  **Executar a Aplicação:**
    ```bash
    python app.py
    ```
    A aplicação estará disponível em uma URL local.

## Licença

Este projeto é distribuído sob a **Licença Pública Geral Affero da GNU v3.0**. Veja o arquivo `LICENSE` para mais detalhes.

## Contato

**Carlos Rodrigues dos Santos**
-   Email: [carlex22@gmail.com](mailto:carlex22@gmail.com)
-   GitHub: [@carlex22](https://github.com/carlex22)
