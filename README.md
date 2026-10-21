# 🎬 CineGraph Studio AI
### Autonomous AI Film Pre-Production Studio & Narrative Grounding Engine
*Orchestrated by Microsoft GraphRAG, LangGraph, ComfyUI, and Qwen Voice Design.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![GraphRAG](https://img.shields.io/badge/GraphRAG-Microsoft-purple.svg)](https://github.com/microsoft/graphrag)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Director](https://img.shields.io/badge/Creator-Miro%20%40miroepiceden-1DA1F2?logo=x)](https://x.com/miroepiceden)

---

## 💡 The Core Problem in Generative Filmmaking

Generative AI video models (Runway, Kling, Sora, MiniMax H3) have unlocked hyper-realistic motion, but they face a critical bottleneck: **Trait Drift & Hallucination**.
* Characters change skin tones, eye colors, or biological traits from scene to scene.
* Environments lose architectural continuity across different camera angles.
* Spoken dialogues suffer from robotic voice delivery and lack emotional cadence instructions.
* Generic LLM prompting forgets subtle details buried hundreds of pages deep in a novel.

**CineGraph Studio AI** eliminates visual and auditory drift by transforming raw literary manuscripts (EPUB, TXT, PDF) into **deterministic, canonical production bibles**.

```
   Raw Literature (EPUB / TXT)
               │
               ▼
   ┌─────────────────────────────────────────────────────────┐
   │             MICROSOFT GRAPHRAG KNOWLEDGE BASE           │
   │      (Entity Extraction & Verbatim Evidence Quotes)     │
   └───────────────────────────┬─────────────────────────────┘
                               │
                               ▼
   ┌─────────────────────────────────────────────────────────┐
   │                LANGGRAPH STATE MACHINE                  │
   │  ┌───────────────────────┐   ┌────────────────────────┐ │
   │  │   Conflict Detector   │──▶│  Human-in-the-Loop     │ │
   │  │ (Audit Contradictions)│   │  (Arbitration Modal)   │ │
   │  └───────────────────────┘   └────────────────────────┘ │
   └───────────────────────────┬─────────────────────────────┘
                               │
            ┌──────────────────┼──────────────────┐
            ▼                  ▼                  ▼
   ┌─────────────────┐┌─────────────────┐┌─────────────────┐
   │ Character Bible ││Environment Bible││Voiceprint Studio│
   │  (#FFFFFF Void  ││ (24mm Anamorphic││  (5D Acoustic   │
   │ Portrait Prompt)││  Matte Plates)  ││  Voice Design)  │
   └────────┬────────┘└────────┬────────┘└────────┬────────┘
            │                  │                  │
            ▼                  ▼                  ▼
   ┌─────────────────┐┌─────────────────┐┌─────────────────┐
   │ ComfyUI / Flux  ││ ComfyUI / Krea  ││ Qwen3-TTS /     │
   │ InstantID / IP  ││ Matte Plate Gen ││ CosyVoice TTS   │
   └─────────────────┘└─────────────────┘└─────────────────┘
```

---

## 🌟 Key Capabilities

### 1. 👤 Canonical Character Extraction & White-Void (#FFFFFF) Prompts
* **Verbatim Grounding:** Pulls exact, unaltered sentences from the novel text as legal proof of identity.
* **Biological Consistency:** Isolates permanent anatomical features (species, skin undertone, eye color, hair texture, unique anatomy) and decouples them from transient outfits.
* **InstantID & IP-Adapter Ready:** Compiles studio-grade character face portrait prompts against an isolated `#FFFFFF` pure white background to enable seamless face-model training and image-to-image conditioning.

### 2. 🏞️ Canonical Environment & 24mm Anamorphic Matte Plate Generator
* **Architectural & Spatial Grounding:** Extracts architectural movements (Cyberpunk, Brutalist, Gothic, Solarpunk), building materials, atmospheric lighting, and spatial landmarks.
* **Sensory Ambiance:** Records olfactory, auditory, and climatic details (e.g., *ionized rain, ozone hum, humid subterranean heat*).
* **Cinematic Matte Plate Prompts:** Formats prompts specifically for wide 24mm anamorphic cinema lenses with volumetric depth and zero character contamination.

### 3. 🎙️ 5-Dimension Acoustic Voice Design (Qwen3-TTS / CosyVoice)
* **The 5-Dimension Formula:** Formulates character voiceprints strictly conforming to:
  $$\text{Voiceprint} = [\text{Persona \& Age}] + [\text{Timbre \& Pitch}] + [\text{Pacing \& Cadence}] + [\text{Emotional Baseline}] + [\text{Microphone Acoustics}]$$
* **Verbatim English Dialogue Samples:** Extracts actual spoken lines in the novel's original English.
* **Line-by-Line Performance Directives:** Prepends nuanced performance instructions (e.g., `[Instruction: Seductive whisper with slight vocal fry, dry close-mic studio recording, zero reverb]`).

### 4. 💬 Dialogue, Voice-Over (V.O.) & Narrator Screenplay Extraction
* **Spoken Dialogues:** Extracts character-to-character conversations aloud on screen, complete with speaker attribution, addressee tags, emotional subtext, and acting performance directives `[Instruction: ...]`. All preserved in original novel English.
* **Cinematic Voice-Over (V.O.):** Isolates internal monologues, psychological reflections, and memories of the POV protagonist with intimate close-mic acoustic tags.
* **Narrator Exposition:** Extracts establishing narrative prose describing sensory ambiance, lighting, and world transitions, matched with 24mm anamorphic cinematic camera beats and a dedicated Narrator Voice Profile.
* **Screenplay JSON Export:** Exports complete audio screenplay bibles ready for batch voice generation or speech alignment.

### 5. ⚖️ Human-in-the-Loop (HITL) Conflict Arbitration
* When the manuscript contains contradictory descriptions (e.g., green-skinned nymph vs. human camouflage, or eye color variance under different lighting), the LangGraph engine halts execution and flags the discrepancy.
* The director resolves the ambiguity via the interactive UI modal before the canonical bible is permanently locked.

---

## 🖥️ Interactive Web Studio

CineGraph Studio AI features a dedicated, zero-latency director dashboard served via FastAPI:

* **Pipeline Execution Monitor:** Real-time terminal streaming of GraphRAG retrieval and node execution.
* **Character Bible View:** Visual badges, biological traits, copy-ready ComfyUI prompts, and voiceprint cards.
* **Environment Bible View:** Architectural breakdowns, sensory cues, and anamorphic matte painting prompts.
* **Dynamic Book Library:** Seamlessly scans, reads, and processes any EPUB or TXT file placed in `data/books/`.
* **Zero Canvas Reset:** Reset databases to a clean slate with a single click.

---

## 🚀 Quickstart Guide

### Prerequisites
* **Python 3.10+** (Conda recommended)
* A local or remote OpenAI-compatible LLM endpoint (e.g., [LM Studio](https://lmstudio.ai/) running `Gemma-2/4 12B` or `Llama-3.3 70B`).

### 1. Clone & Install
```bash
git clone https://github.com/your-username/cinegraph-studio-ai.git
cd cinegraph-studio-ai
pip install -r requirements.txt
```

### 2. Configure Environment
Copy the example environment file:
```bash
cp .env.example .env
```
Ensure your local LLM server is running (default: `http://localhost:1234/v1`).

### 3. Launch Studio
On Windows, simply double-click `Run_Studio.bat` or run:
```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8555
```
Open your browser at **`http://127.0.0.1:8555/agent`**.

---

## 🕹️ CLI Headless Mode

You can also run character or environment extractions directly from the command line:

```bash
# Extract and lock a character from a novel
python run_agent.py --book "data/books/my_novel.epub" --character "Mamba"

# Run acoustic voiceprint extraction
python voiceprint_production.py "Mamba" "graphrag_workspace/input/my_novel.txt"

# Run environment matte plate extraction
python environment_production_graph.py "Meteoropolis Laboratories" "graphrag_workspace/input/my_novel.txt"
```

---

## 📁 Repository Structure

```
cinegraph-studio-ai/
├── server.py                        # FastAPI backend & Studio API endpoints
├── novel_production_graph.py        # LangGraph Character StateGraph & HITL engine
├── environment_production_graph.py  # LangGraph Environment & Matte Plate engine
├── voiceprint_production.py         # 5D Voice Design & English dialogue extractor
├── run_agent.py                     # Headless CLI pipeline runner
├── Run_Studio.bat                   # One-click Windows studio launcher
├── requirements.txt                 # Production dependencies
├── .env.example                     # Environment configuration template
│
├── static/
│   ├── agent.html                   # Cinematic Director Studio Web Interface
│   └── index.html                   # Asset Bible reader
│
├── data/
│   ├── character_bible.json         # Locked canonical character profiles
│   ├── environment_bible.json       # Locked canonical environment profiles
│   └── books/                       # Drag & drop your EPUB / TXT novels here
│
├── docs/
│   └── models_prompt_guides/        # Prompt design specs for Krea, Qwen, Flux, Minimax
│
└── graphrag_workspace/
    ├── settings.yaml                # Microsoft GraphRAG orchestration config
    └── prompts/                     # Extraction & entity prompts
```

---

## ⚡ Integration with Fast AI Engines (e.g., Pruna AI)

In high-throughput cinematic workflows, latency is everything. CineGraph Studio AI coordinates multiple heavy local models:
1. **Long-context LLMs** for deep narrative extraction.
2. **Text-to-Image / Diffusion models** (FLUX 3, Krea 2 Turbo, SDXL) in ComfyUI.
3. **Voice Synthesis engines** (Qwen3-TTS, CosyVoice).

By integrating model compression and acceleration frameworks like **[Pruna AI](https://www.pruna.ai/)**, creators can shrink VRAM footprints and accelerate generation speeds by **3x to 5x**, allowing full-length novel pre-production to run smoothly on consumer-tier dual-GPU workstations.

---

## 👨‍🎨 Author & Director

Developed by **Miro (Epic Eden)**  
Cinematic Director & AI Creative Technologist  
* 🐦 X (Twitter): [@miroepiceden](https://x.com/miroepiceden)  
* 🎬 Showcase: [Watch Latest AI Cinematic Trailers](https://x.com/miroepiceden/status/2110799507197952013)

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
