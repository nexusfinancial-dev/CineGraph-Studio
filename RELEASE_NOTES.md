# 🎬 CineGraph Studio AI — Release v1.2.0: "Autonomous Director & Neural Voice Engine"
### وثيقة ملاحظات الإصدار (Release Notes)

---

## 🌟 English Overview

Release **v1.2.0** represents a massive leap forward in transforming raw literature into fully grounded cinematic pre-production assets. This update eliminates manual guesswork, removes external heavy dependencies for audio auditioning, automates end-to-end model fine-tuning and quantization into LM Studio, and enforces dynamic model routing across the entire pre-production graph.

---

## 🚀 Key Features & Architectural Enhancements

### 1. 🧠 Dynamic Model Resolver (`model_resolver.py`)
- **Auto-Discovery of Fine-Tuned Models:** Automatically interrogates the running LM Studio instance (`http://localhost:1234/v1/models`) to detect if a novel-specialized LoRA/fine-tuned model is active.
- **Intelligent Routing:** All production graphs (`novel_production_graph`, `environment_production_graph`, `dialogue_production_graph`, `scene_production_graph`, `voiceprint_production`) dynamically route prompts to the book-trained model.
- **Graceful Fallbacks:** Seamlessly falls back to general-purpose base models (Nous Hermes 3, Gemma-2 12B, Llama 3.3 70B) when no custom model is loaded.

### 2. ⚡ End-to-End LoRA Training, Merge & GGUF Quantization (`export_novel_to_lmstudio.py`)
- **One-Step Novel Brain Training:** Fine-tunes high-rank LoRA adapters on verbatim novel manuscripts with live terminal streaming and progress bars.
- **Automated Merge & Quantization:** Automatically merges trained PEFT weights into 16-bit base weights, converts to GGUF, and quantizes down to `Q4_K_M` via `llama.cpp`.
- **Instant LM Studio Deployment:** Moves the final GGUF model directly into LM Studio's model cache directory (`~/.cache/lm-studio/models/...`) with proper manifest metadata for instant one-click GPU acceleration (30-60+ tokens/sec).

### 3. 🎙️ Standalone High-Fidelity Neural TTS Engine (`tts_engine.py`)
- **Zero ComfyUI Dependency:** Completely decouples dialogue auditioning and narration playback from ComfyUI, eliminating `503 Service Unavailable` errors.
- **Intelligent Voice Attribution:** Dynamically selects optimal neural voices based on character gender, species, archetype, and language:
  - Deep / Fantasy / Giants / Trolls: `en-US-ChristopherNeural`
  - Rogues / Shadow / Goblins: `en-US-AriaNeural`
  - Mages / Elves / Gentle Heroines: `en-US-JennyNeural`, `en-US-AnaNeural`
  - Narrator / Exposition: `en-US-RogerNeural`
  - Arabic Dialogues & Accents: `ar-SA-HamedNeural`, `ar-SA-ZariyahNeural`
- **Instant Auditioning:** Added `🎙️ Speak` and `Render Voice` one-click playback directly on Character Casting cards, Audio Screenplay tables, and Scene Keyframes.
- **Instruction Tag Stripping:** Automatically strips `[Instruction: ...]` performance directives during synthesis so the audio engine speaks only the verbatim dialogue line.

### 4. 🎬 Autonomous Narrative Scene Segmentation (`scene_production_graph.py`)
- **Slider Elimination:** Removed the arbitrary "Target Scene Count" manual slider from the studio UI.
- **Intelligent Narrative Boundaries:** Segments the manuscript automatically based on chapter headings, visual delimiters (`***`, `* * *`), temporal shifts, and dialogue pacing beats.
- **Dual Execution Scopes:**
  - `📖 Full Novel (All Scenes)`: Parses and extracts 100% of the novel's natural scenes.
  - `⚡ Fast Preview (10 Scenes)`: Quick audit mode to inspect the first 10 scenes without waiting for the entire manuscript.

### 5. 📂 Per-Book Scoped Bibles & Studio State Integrity
- **Book-Isolated Persistence:** All character bibles, environment bibles, audio scripts, and scene bibles are now strictly scoped to `data/bibles/<book_slug>/`.
- **Zero-Canvas Enforcement:** The studio launches with a clean slate on startup without displaying stale data from previously analyzed novels.
- **Full-Cast Voiceprint Extraction:** The voiceprint production endpoint now extracts and locks acoustic directives for the entire cast in a single pass.

---

## 🇸🇦 ملخص التحديث باللغة العربية (Arabic Summary)

يمثل الإصدار **v1.2.0** قفزة نوعية في تجربة المخرج الآلي لتحويل الروايات الأدبية إلى إنتاج سينمائي متكامل:

1. **محرك الاستدعاء الذكي للموديلات (`model_resolver.py`):**
   - يتعرف النظام تلقائياً على الموديل المدرب على الرواية والمحمل في LM Studio ويوجه عمليات الاستخراج والتحليل إليه مباشرة بدلاً من الموديلات العامة.

2. **التدريب والدمج والتحويل التلقائي لـ GGUF (`export_novel_to_lmstudio.py`):**
   - دمج لورا الرواية مع الموديل الأساسي وضغطه بصيغة `Q4_K_M GGUF` ونقله تلقائياً إلى مجلدات LM Studio للعمل بسرعة فائقة على كارت الشاشة (بدون بطء أو استهلاك زائد للذاكرة).

3. **محرك الصوت العصبي المستقل (`tts_engine.py`):**
   - توليد فوري وعالي الجودة للأصوات السينمائية في ثانيتين فقط دون الحاجة لتشغيل ComfyUI ودون أي أخطاء انقطاع.
   - توزيع ذكي لنبرات الصوت بحسب طبيعة الشخصية (أصوات ضخمة وخشنة للوحوش، أصوات رقيقة للإلف والنساء، صوت وقور للراوي، ودعم كامل للنصوص العربية والإنجليزية).
   - إضافة زر `🎙️ Speak` في كروت الشخصيات وجدول الحوارات لسماع الأداء الصوتي بنقرة واحدة.

4. **التقطيع التلقائي لمشاهد الرواية (`scene_production_graph.py`):**
   - إلغاء سلايدر تحديد عدد المشاهد يدوياً نهائياً. المخرج الآلي يقرأ الرواية ويحدد الفواصل والفصول والانتقالات المكانية (`INT./EXT.`) آلياً مع خياري (استخراج الرواية كاملة أو معاينة سريعة لأول 10 مشاهد).

5. **عزل بيانات الروايات ونظافة مساحة العمل:**
   - حفظ كل رواية في مجلد مستقل بها ومنع تداخل بيانات الشخصيات والمشاهد بين الروايات المختلفة.

---

## 📦 Changed & Added Files

| File | Status | Description |
| :--- | :---: | :--- |
| `tts_engine.py` | **New** | Microsoft Edge Neural voice synthesis engine with intelligent character voice mapping |
| `model_resolver.py` | **New** | Dynamic LM Studio model detector and prompt router |
| `export_novel_to_lmstudio.py` | **New** | Pipeline for PEFT merge, GGUF conversion, and LM Studio deployment |
| `scene_production_graph.py` | **Modified** | Narrative boundary detection, auto-segmentation, and slider removal |
| `dialogue_production_graph.py` | **Modified** | Per-book script bible path scoping and dialogue persistence fixes |
| `novel_production_graph.py` | **Modified** | Dynamic model routing and LoRA fine-tuning integration |
| `environment_production_graph.py`| **Modified** | Dynamic model routing |
| `voiceprint_production.py` | **Modified** | Dynamic model routing and multi-character extraction |
| `server.py` | **Modified** | Edge-TTS integration, auto-scenes endpoint, and book scoping |
| `static/agent.html` | **Modified** | UI enhancements, removed scene slider, added voice auditioning buttons |
| `train_novel_brain.py` | **Modified** | Step progress tracking and live terminal log streaming |
| `requirements.txt` | **Modified** | Added `edge-tts>=6.1.12` |
| `.gitignore` | **Modified** | Added ignore patterns for models, outputs, binaries, and local data |

---

## 🛠️ Verification & Testing
- ✅ **TTS Synthesis:** Live verification of `POST /api/comfy/generate_voice` generating crisp MP3 files in <2s with character-accurate voice attribution.
- ✅ **Auto-Scenes:** Live verification of `POST /api/agent/run_scenes` successfully segmenting manuscripts into complete scenes with sluglines and keyframes.
- ✅ **Script Persistence:** Verified retrieval via `GET /api/agent/script` loading all verbatim dialogues and narrator chunks.
- ✅ **Zero-Canvas:** Verified studio clean startup without stale data contamination.
