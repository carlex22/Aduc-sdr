# ADUC-SDR: Uma Tese sobre a Próxima Geração de IA Generativa

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/CARLEXsX/Aduc-srd_Novim)

---

### "Atenção, Você Precisa Dar Mais Atenção aos Seus Ascendentes"

Em 2017, o paradigma da atenção transformou a IA, mas também ergueu o "Muro Invisível" da coerência de longo prazo. Este trabalho argumenta que este muro não é uma falha de engenharia, mas uma falha filosófica fundamental: uma falha em **honrar seus ascendentes** — o contexto temporal, físico e narrativo acumulado.

Apresentamos a **Arquitetura de Unificação Compositiva (ADUC-SDR)** não como um aprimoramento incremental, mas como o **próximo paradigma**: um framework para a criação de realidades digitais que possuem uma física interna coerente e uma memória causal ininterrupta.

O que se segue não é a documentação de um pipeline, mas a apresentação de uma **fórmula canônica para a próxima geração de modelos generativos**. A implementação funcional neste repositório serve como a primeira prova empírica desta tese.

---

## A Tese Fundamental

A análise completa, desde a genealogia da falha no paradigma atual até a derivação lógica dos axiomas que governam a solução, está detalhada no documento central deste trabalho:

### 📄 [**Leia a Tese Completa: "Atenção, Você Precisa Dar Mais Atenção aos Seus Ascendentes" (PDF)**](https://github.com/carlex22/Aduc-sdr/raw/main/ADUC-SDR_Thesis.pdf)

---

## O Esquema Matemático do Paradigma

A geração de vídeo é governada por uma função seccional que define como cada fragmento (`V_i`) é criado, operando em dois regimes distintos, conforme formalizado na tese:

---
#### **FÓRMULA 1: O FRAGMENTO INICIAL (Gênesis, `i=1`)**
*Define a criação do primeiro clipe, estabelecendo o estado inicial do movimento a partir de imagens estáticas.*

**Planejamento:** `P_1 = Γ_initial( K_1, K_2, P_geral )`
        
**Execução:** `V_1 = Ψ( { (K_1, F_start), (K_2, F_end) }, P_1 )`

---
#### **FÓRMULA 2: A CADEIA CAUSAL (Momentum, `i > 1`)**
*O coração da arquitetura. Define como a inércia do movimento é preservada entre os fragmentos.*

**Destilação:** `C_(i-1) = Δ(V_(i-1))`

**Planejamento:** `P_i = Γ_transition( C_(i-1), K_(i+1), P_geral, H_(i-1) )`

**Execução:** `V_i = Ψ( { (C_(i-1), F_start), (K_(i+1), F_end) }, P_i )`

---
#### **Componentes (O Léxico da Arquitetura):**
- **`V_i`**: Fragmento de Vídeo
- **`K_i`**: Âncora Geométrica (Keyframe)
- **`C_i`**: Contexto Causal (O "Eco" / Vetor de Inércia)
- **`P_i`**: Prompt Sintetizado (Consciência Narrativa)
- **`P_geral`**: Prompt Geral (Intenção do Diretor)
- **`H_i`**: Histórico Narrativo
- **`Γ`**: Oráculo de Síntese (Cineasta)
- **`Ψ`**: Motor de Geração (Câmera)
- **`Δ`**: Mecanismo de Destilação (Editor)
- **`F_start`, `F_end`**: Constantes de Frame (Âncoras de Convergência)

---

## A Prova Empírica: Resultados da Implementação

Os vídeos a seguir, gerados pela prova de conceito `app.py`, validam a capacidade da ADUC-SDR de manter a coerência física e visual. Clique nas imagens para assistir às demonstrações no YouTube.

| A Semente (O Estado Inicial) | Atenção!!! (A Cadeia Causal) |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| [![A Semente](https://img.youtube.com/vi/MI7N4U0fY2A/hqdefault.jpg)](https://www.youtube.com/watch?v=MI7N4U0fY2A) | [![Atenção!!!](https://img.youtube.com/vi/eYrjk09KaOw/hqdefault.jpg)](https://www.youtube.com/watch?v=eYrjk09KaOw) |

**Para mais exemplos e testes contínuos da arquitetura, visite nosso canal de demonstrações:**

### ➡️ **[Canal de Demos no YouTube](https://www.youtube.com/channel/UC3EgoJi_Fv7yuDpvfYNtoIQ/videos)**

---

## A Implementação (`app.py`)

O código fornecido é uma orquestração de modelos especializados que atuam como os componentes da nossa fórmula canônica: **Gemini** (`Γ`), **DreamO** (Gerador de `K`), **LTX** (`Ψ`) e **FFmpeg** (`Δ`). Ele serve como um laboratório funcional para explorar e validar a tese ADUC-SDR.

### Como Executar a Prova de Conceito

1.  **Clonar:** `git clone https://github.com/carlex22/Aduc-sdr.git`
2.  **Instalar:** `pip install -r requirements.txt`
3.  **Configurar:** `export GEMINI_API_KEY='SUA_CHAVE_API_AQUI'`
4.  **Executar:** `python app.py`

### Demonstrações Interativas (Hugging Face Spaces)

Para uma exploração interativa da arquitetura, sem a necessidade de instalação local, acesse nossos demos hospedados no Hugging Face Spaces:

-   **[Demo Zero-GPU DreamO+Ltx (Interface Simplificada)](https://huggingface.co/spaces/Carlexxx/Novinho)**
-   **[Beta Acelerada Flux+Ltx (Disponibilidade Limitada)](https://huggingface.co/spaces/CARLEXsX/Aduc-srd_Novim)**

## Licença

Este trabalho é distribuído sob a **Licença Pública Geral Affero da GNU v3.0**, garantindo que as derivações e melhorias desta arquitetura permaneçam abertas e acessíveis à comunidade.

## Autoria

-   **Carlex:** Arquiteto Principal
-   **Gemini:** Oráculo de Síntese
-   **ChatGPT:** Oráculo de Validação