import os
import re
import sys
import json
import zipfile
import shutil
import subprocess
import requests
import xml.etree.ElementTree as ET
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
import json_repair

sys.stdout.reconfigure(encoding='utf-8')

app = FastAPI(title="Descriptive Studio (Director Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:1234/v1")
BOOKS_DIR_EPUB = os.environ.get("BOOKS_DIR_EPUB", r"E:\Ero\epub" if os.path.exists(r"E:\Ero\epub") else "")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
LOCAL_BOOKS_DIR = os.path.join(DATA_DIR, "books")
DB_FILE = os.path.join(DATA_DIR, "asset_bible.json")
RECENTS_FILE = os.path.join(DATA_DIR, "reading_history.json")
GEMINI_CONFIG_FILE = os.path.join(DATA_DIR, "gemini_config.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOCAL_BOOKS_DIR, exist_ok=True)

def load_gemini_config() -> Dict[str, Any]:
    if os.path.exists(GEMINI_CONFIG_FILE):
        try:
            with open(GEMINI_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {"api_key": "", "preferred_engine": "agy", "model": "gemini-3.8-flash-low"}

def save_gemini_config(cfg: Dict[str, Any]):
    with open(GEMINI_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

AGY_EXE_PATHS = [
    r"C:\pinokio\bin\agy.exe",
    shutil.which("agy")
]

def find_agy_executable() -> Optional[str]:
    for p in AGY_EXE_PATHS:
        if p and os.path.exists(p):
            return p
    return None

def get_book_db_path(book_id: Optional[str] = None) -> str:
    if not book_id:
        return DB_FILE
    safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', book_id)
    return os.path.join(DATA_DIR, f"bible_{safe_id}.json")

# Initialize Asset Bible Database (Scoped per book)
def load_db(book_id: Optional[str] = None) -> Dict[str, Any]:
    file_path = get_book_db_path(book_id)
    if not os.path.exists(file_path):
        empty_db = {
            "characters": [],
            "locations": [],
            "dialogues": [],
            "saved_cards": [],
            "saved_scans": {}
        }
        return empty_db
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"characters": [], "locations": [], "dialogues": [], "saved_cards": [], "saved_scans": {}}

def save_db(data: Dict[str, Any], book_id: Optional[str] = None):
    file_path = get_book_db_path(book_id)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

epub_cache = {}

def get_best_local_uncensored_model() -> str:
    try:
        r = requests.get(f"{LM_STUDIO_URL}/models", timeout=3)
        if r.status_code == 200:
            models = [m.get("id") for m in r.json().get("data", [])]
            for m in models:
                if "uncensored" in m.lower() and "26b" in m.lower() and "@?" not in m:
                    return m
            for m in models:
                if "uncensored" in m.lower() and "@?" not in m:
                    return m
            for m in models:
                if "uncensored" in m.lower():
                    return m
            if models:
                return models[0]
    except Exception:
        pass
    return "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m"

@app.get("/api/status")
def get_status():
    lm_status = False
    models = []
    try:
        r = requests.get(f"{LM_STUDIO_URL}/models", timeout=3)
        if r.status_code == 200:
            lm_status = True
            models = [m.get("id") for m in r.json().get("data", [])]
    except Exception:
        pass

    agy_path = find_agy_executable()
    cfg = load_gemini_config()
    api_key_set = bool(cfg.get("api_key"))

    active_engine = "none"
    if agy_path:
        active_engine = "Antigravity CLI (Browser OAuth)"
    elif api_key_set:
        active_engine = "Google AI Studio API Key"

    uncensored_model = get_best_local_uncensored_model() if lm_status else None
    has_uncensored = bool(any("uncensored" in m.lower() for m in models)) if lm_status else False

    return {
        "lm_studio_connected": lm_status,
        "loaded_models": models,
        "uncensored_model": uncensored_model,
        "has_uncensored_model": has_uncensored,
        "agy_connected": bool(agy_path),
        "agy_path": agy_path or "",
        "gemini_api_key_set": api_key_set,
        "gemini_ready": bool(agy_path or api_key_set),
        "active_gemini_engine": active_engine
    }

@app.get("/api/gemini/config")
def get_gemini_config_endpoint():
    cfg = load_gemini_config()
    masked_key = ""
    if cfg.get("api_key"):
        raw_key = cfg["api_key"]
        masked_key = raw_key[:4] + "..." + raw_key[-4:] if len(raw_key) > 8 else "***"
    return {
        "preferred_engine": cfg.get("preferred_engine", "agy"),
        "model": cfg.get("model", "gemini-3.8-flash-low"),
        "has_api_key": bool(cfg.get("api_key")),
        "masked_key": masked_key,
        "agy_available": bool(find_agy_executable()),
        "agy_path": find_agy_executable() or ""
    }

class SaveGeminiConfigRequest(BaseModel):
    api_key: Optional[str] = None
    preferred_engine: Optional[str] = "agy"
    model: Optional[str] = "gemini-3.8-flash-low"

@app.post("/api/gemini/config")
def set_gemini_config_endpoint(req: SaveGeminiConfigRequest):
    cfg = load_gemini_config()
    if req.api_key is not None:
        cfg["api_key"] = req.api_key.strip()
    if req.preferred_engine:
        cfg["preferred_engine"] = req.preferred_engine
    if req.model:
        cfg["model"] = req.model
    save_gemini_config(cfg)
    return {"status": "success", "config": cfg}

def parse_txt_chapters(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    lines = text.splitlines()
    if not lines:
        return []

    # Check if text already has explicit "=== NEXT CHAPTER ===" markers
    if "=== NEXT CHAPTER ===" in text:
        parts = text.split("=== NEXT CHAPTER ===")
        chapters = []
        for idx, part in enumerate(parts):
            p_lines = [l.strip() for l in part.splitlines() if l.strip()]
            if not p_lines:
                continue
            first_l = p_lines[0]
            paras = [p.strip() for p in part.split('\n\n') if p.strip()]
            chapters.append({
                "chapter_num": len(chapters) + 1,
                "title": first_l[:50],
                "title_ar": "",
                "paragraphs": paras,
                "start_line": 1,
                "end_line": len(p_lines)
            })
        if chapters:
            return chapters

    heading_pattern = r'^((\d+[\.:]\s+[^\n]+)|(Chapter\s+\d+[^\n]*)|([A-Za-z0-9\s\'\’\-]+Interlude)|(Prologue)|(Epilogue))\s*$'
    
    candidates = []
    for i in range(len(lines)):
        l = lines[i].strip()
        if not l:
            continue
        m = re.match(heading_pattern, l, re.IGNORECASE)
        if m:
            prev_blank = (i == 0 or not lines[i-1].strip())
            next_l = lines[i+1].strip() if i+1 < len(lines) else ''
            has_ar = any('\u0600' <= c <= '\u06FF' for c in next_l)
            if prev_blank:
                candidates.append((i, l, next_l if has_ar else ''))

    real_chapters = []
    for idx, (line_no, en_title, ar_title) in enumerate(candidates):
        start = line_no + (2 if ar_title else 1)
        end = candidates[idx+1][0] if idx+1 < len(candidates) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        
        if len(body) >= 250:
            raw_paras = [p.strip() for p in body.split('\n\n') if p.strip()]
            cleaned_paras = [p for p in raw_paras if not p.lower().startswith('oceanofpdf') and 'all rights reserved' not in p.lower()]
            if cleaned_paras:
                real_chapters.append({
                    "chapter_num": len(real_chapters) + 1,
                    "title": en_title,
                    "title_ar": ar_title,
                    "paragraphs": cleaned_paras,
                    "start_line": line_no + 1,
                    "end_line": end
                })

    if not real_chapters:
        all_paras = [p.strip() for p in text.split('\n\n') if p.strip() and not p.lower().startswith('oceanofpdf')]
        chunk_size = 50
        for i in range(0, len(all_paras), chunk_size):
            chunk = all_paras[i:i+chunk_size]
            real_chapters.append({
                "chapter_num": len(real_chapters) + 1,
                "title": f"Section {len(real_chapters) + 1}",
                "title_ar": f"القسم {len(real_chapters) + 1}",
                "paragraphs": chunk,
                "start_line": i * 2,
                "end_line": (i + chunk_size) * 2
            })

    return real_chapters

def find_book_path(book_id: str) -> Optional[str]:
    def norm(s: str) -> str:
        return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

    if os.path.exists(book_id):
        return book_id

    target_id = book_id[:-6] if book_id.endswith("__epub") else book_id
    target_norm = norm(target_id)
    prefer_epub = book_id.endswith("__epub")

    def search_dir(directory: str) -> Optional[str]:
        if not os.path.exists(directory):
            return None
        
        all_files = []
        for root, _, files in os.walk(directory):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".txt", ".epub"]:
                    all_files.append((os.path.join(root, f), f, ext))

        first_ext = ".epub" if prefer_epub else ".txt"
        second_ext = ".txt" if prefer_epub else ".epub"

        # 1. Exact match with primary extension
        for full_path, f, ext in all_files:
            if ext == first_ext and os.path.splitext(f)[0] == target_id:
                return full_path

        # 2. Exact match with secondary extension
        for full_path, f, ext in all_files:
            if ext == second_ext and os.path.splitext(f)[0] == target_id:
                return full_path

        # 3. Normalized match with primary extension
        for full_path, f, ext in all_files:
            if ext == first_ext and norm(os.path.splitext(f)[0]) == target_norm:
                return full_path

        # 4. Normalized match with secondary extension
        for full_path, f, ext in all_files:
            if ext == second_ext and norm(os.path.splitext(f)[0]) == target_norm:
                return full_path

        return None

    # 1. Check in local uploaded books
    p_local = search_dir(LOCAL_BOOKS_DIR)
    if p_local:
        return p_local

    # 2. Check in main library books (including subfolders like "New folder")
    p_lib = search_dir(BOOKS_DIR_EPUB)
    if p_lib:
        return p_lib

    return None

def load_recents() -> List[Dict[str, Any]]:
    if not os.path.exists(RECENTS_FILE):
        return []
    try:
        with open(RECENTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_recents(data: List[Dict[str, Any]]):
    try:
        with open(RECENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving recents: {e}")

class RecentProgress(BaseModel):
    book_id: str
    title: Optional[str] = ""
    chapter_num: Optional[int] = 1
    chapter_title: Optional[str] = ""
    page_num: Optional[int] = 1
    total_pages: Optional[int] = 1
    para_idx: Optional[int] = 0
    scroll_ratio: Optional[float] = 0.0

@app.get("/api/books")
def list_books():
    books = []
    seen = set()

    def process_dir(directory, source_type):
        if not os.path.exists(directory):
            return
        all_entries = []
        for root, _, files in os.walk(directory):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in [".txt", ".epub"]:
                    all_entries.append((root, f, ext))

        all_entries.sort(key=lambda x: (x[2] != ".txt", not ("en_ar" in x[1].lower()), x[1]))

        for root, f, ext in all_entries:
            full_path = os.path.join(root, f)
            base = os.path.splitext(f)[0]
            item_id = base if ext == ".txt" else f"{base}__epub" if os.path.exists(os.path.join(root, f"{base}.txt")) else base
            if item_id not in seen:
                seen.add(item_id)
                clean_title = base.replace("_OceanofPDF.com_", "").replace("_", " ")
                is_txt = (ext == ".txt")
                is_bilingual = False
                if is_txt:
                    if "_en_ar" in f.lower() or "en_ar" in f.lower():
                        is_bilingual = True
                    else:
                        try:
                            with open(full_path, 'r', encoding='utf-8', errors='ignore') as sf:
                                sample = sf.read(2048)
                                is_bilingual = bool(re.search(r'[\u0600-\u06FF]', sample))
                        except Exception:
                            pass
                books.append({
                    "id": item_id,
                    "title": clean_title,
                    "filename": f,
                    "format": ext.replace(".", ""),
                    "is_bilingual": is_bilingual,
                    "source": source_type
                })

    process_dir(LOCAL_BOOKS_DIR, "local")
    process_dir(BOOKS_DIR_EPUB, "library")

    return sorted(books, key=lambda x: (not x.get("is_bilingual", False), x.get("format") != "txt", x["title"]))

@app.post("/api/upload_book")
async def upload_book(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".txt", ".epub"]:
        raise HTTPException(status_code=400, detail="Only .txt and .epub files are supported")
    save_path = os.path.join(LOCAL_BOOKS_DIR, file.filename)
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)
    base = os.path.splitext(file.filename)[0]
    clean_title = base.replace("_OceanofPDF.com_", "").replace("_", " ")
    return {
        "status": "success",
        "id": base,
        "title": clean_title,
        "filename": file.filename,
        "format": ext.replace(".", "")
    }

@app.get("/api/recents")
def get_recents():
    return load_recents()

@app.post("/api/recents")
def record_recent(req: RecentProgress):
    recents = load_recents()
    recents = [r for r in recents if r.get("book_id") != req.book_id]
    entry = {
        "book_id": req.book_id,
        "title": req.title or req.book_id,
        "chapter_num": req.chapter_num or 1,
        "chapter_title": req.chapter_title or f"Chapter {req.chapter_num}",
        "page_num": req.page_num or 1,
        "total_pages": req.total_pages or 1,
        "para_idx": req.para_idx or 0,
        "scroll_ratio": req.scroll_ratio or 0.0,
        "updated_at": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    recents.insert(0, entry)
    save_recents(recents[:30])
    return {"status": "success", "recent": entry}

@app.get("/api/book/{book_id}/chapters")
def get_chapters(book_id: str):
    book_path = find_book_path(book_id)
    if not book_path or not os.path.exists(book_path):
        raise HTTPException(status_code=404, detail="Book file not found")

    if book_path.lower().endswith(".txt"):
        txt_chaps = parse_txt_chapters(book_path)
        return [{
            "chapter_num": ch["chapter_num"],
            "title": ch["title"],
            "title_ar": ch.get("title_ar", ""),
            "start_line": ch.get("start_line", 0),
            "end_line": ch.get("end_line", 0)
        } for ch in txt_chaps]

    chapters = []
    try:
        with zipfile.ZipFile(book_path, 'r') as z:
            toc_name = next((f for f in z.namelist() if f.endswith('toc.ncx')), None)
            if toc_name:
                try:
                    tree = ET.fromstring(z.read(toc_name))
                    for np in tree.iter('{http://www.daisy.org/z3986/2005/ncx/}navPoint'):
                        lbl = np.find('{http://www.daisy.org/z3986/2005/ncx/}navLabel')
                        txt = lbl.find('{http://www.daisy.org/z3986/2005/ncx/}text').text.strip() if lbl is not None else ""
                        content = np.find('{http://www.daisy.org/z3986/2005/ncx/}content')
                        src = content.get('src') if content is not None else ''
                        if src and txt:
                            t_lower = txt.strip().lower()
                            if t_lower in ["table of contents", "contents", "toc", "title page", "copyright", "cover", "about the author", "also by"]:
                                continue
                            clean_src = src.split('#')[0]
                            matched = next((f for f in z.namelist() if f.endswith(clean_src)), None)
                            if matched:
                                raw_c = z.read(matched).decode('utf-8', errors='ignore')
                                soup = BeautifulSoup(raw_c, 'html.parser')
                                valid_p = [p for p in soup.find_all('p') if len(p.get_text().strip()) > 20 and not p.get_text().strip().startswith("OceanofPDF")]
                                if len(valid_p) >= 2:
                                    chapters.append({
                                        "chapter_num": len(chapters) + 1,
                                        "title": txt[:45],
                                        "html_file": clean_src
                                    })
                except Exception as e:
                    print(f"TOC error: {e}")

            if len(chapters) < 2:
                chapters = []
                html_files = [f for f in z.namelist() if f.endswith(('.xhtml', '.html', '.htm'))]
                for hf in html_files:
                    c = z.read(hf).decode('utf-8', errors='ignore')
                    soup = BeautifulSoup(c, 'html.parser')
                    valid_p = [p for p in soup.find_all('p') if len(p.get_text().strip()) > 20 and not p.get_text().strip().startswith("OceanofPDF")]
                    if len(valid_p) >= 3:
                        h_tag = soup.find(['h1', 'h2', 'h3', 'title'])
                        t_text = h_tag.get_text().strip() if h_tag else f"Chapter {len(chapters) + 1}"
                        if t_text.strip().lower() in ["table of contents", "contents", "toc", "title page", "copyright", "cover"]:
                            continue
                        chapters.append({
                            "chapter_num": len(chapters) + 1,
                            "title": t_text[:40],
                            "html_file": hf
                        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return chapters

@app.get("/api/book/{book_id}/chapter/{ch_num}")
def get_chapter_content(book_id: str, ch_num: int):
    cache_key = f"{book_id}_ch_{ch_num}"
    if cache_key in epub_cache:
        return epub_cache[cache_key]

    book_path = find_book_path(book_id)
    if not book_path or not os.path.exists(book_path):
        raise HTTPException(status_code=404, detail="Book not found")

    if book_path.lower().endswith(".txt"):
        txt_chaps = parse_txt_chapters(book_path)
        if 1 <= ch_num <= len(txt_chaps):
            ch = txt_chaps[ch_num - 1]
            result = {
                "book_id": book_id,
                "chapter_num": ch_num,
                "chapter_title": ch["title"],
                "chapter_title_ar": ch.get("title_ar", ""),
                "paragraphs": ch["paragraphs"],
                "is_bilingual": True
            }
            epub_cache[cache_key] = result
            return result
        else:
            raise HTTPException(status_code=404, detail="Chapter not found in txt")

    paragraphs = []
    chapter_title = f"Chapter {ch_num}"
    chapter_title_ar = ""
    try:
        with zipfile.ZipFile(book_path, 'r') as z:
            chapters = get_chapters(book_id)
            target_file = None
            if ch_num <= len(chapters):
                target_file = chapters[ch_num - 1].get("html_file")
                chapter_title = chapters[ch_num - 1].get("title", f"Chapter {ch_num}")
                chapter_title_ar = chapters[ch_num - 1].get("title_ar", "")
            
            if not target_file:
                target_file = next((f for f in z.namelist() if f.endswith(('.xhtml', '.html', '.htm')) and 'split_00' in f), None)
            
            matched_file = next((f for f in z.namelist() if f.endswith(target_file)), target_file)
            raw_html = z.read(matched_file).decode('utf-8', errors='ignore')
            soup = BeautifulSoup(raw_html, 'html.parser')

            for p in soup.find_all('p'):
                t = " ".join(p.get_text().split())
                if len(t) > 10 and not t.startswith("OceanofPDF") and t != "Desconhecido":
                    paragraphs.append(t)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    result = {
        "book_id": book_id,
        "chapter_num": ch_num,
        "chapter_title": chapter_title,
        "chapter_title_ar": chapter_title_ar,
        "paragraphs": paragraphs
    }
    epub_cache[cache_key] = result
    return result

class ScanStepRequest(BaseModel):
    book_id: str
    chapter_num: int
    step: Optional[str] = "full"

@app.post("/api/scan_chapter")
def scan_full_chapter(req: ScanStepRequest):
    chapter_data = get_chapter_content(req.book_id, req.chapter_num)
    paragraphs = chapter_data.get("paragraphs", [])
    if not paragraphs:
        raise HTTPException(status_code=400, detail="Chapter is empty")

    full_text = "\n\n".join(paragraphs)

    prompt = f"""You are a professional film director and script supervisor breaking down this novel chapter for visual production.
Analyze the ENTIRE chapter thoroughly from beginning to end and extract:

1. ACTIVE CHARACTERS (Every character physically present on-screen in this chapter's current timeline. Include protagonists and any newly encountered characters. Mark backstory mentions separately).
2. OUTFIT & GEAR (Clothing, coats, boots, backpacks, gear, and physical condition like bruises or wounds for each character).
3. KEY SETTINGS (The primary physical locations of this chapter).
4. SPOKEN DIALOGUE vs VOICE-OVER:
   - "spoken_dialogue": The key spoken dialogue lines and exchanges between characters that happen aloud in the scene. Include speaker, line_en, line_ar, and tone.
   - "voice_over_thoughts": Key reflective narrative thoughts for film voice-over (V.O.).
5. CINEMATIC BEATS (Key visual keyframe shots of the chapter across its full progression from start to finish).

Output strictly valid JSON with this exact schema:
{{
  "active_characters": [
    {{
      "name": "Character Name",
      "status": "Protagonist / Active Solo / Encountered Character",
      "physical_en": "Concise physical features, build, visible injuries or appearance",
      "physical_ar": "الملامح الجسدية بالعربية"
    }}
  ],
  "outfits": [
    {{
      "character_name": "Character Name",
      "title": "Outfit Title",
      "description_en": "Garments, coats, boots, backpack, condition",
      "description_ar": "الملابس والأحذية والعتاد بالعربية"
    }}
  ],
  "settings": [
    {{
      "name": "Location Name",
      "description_en": "Visual environment, atmosphere, details",
      "description_ar": "وصف المكان بالعربية"
    }}
  ],
  "spoken_dialogue": [
    {{
      "speaker": "Speaker Name",
      "line_en": "Spoken line in English",
      "line_ar": "ترجمة الجملة بالعربية",
      "tone": "Emotional tone / delivery"
    }}
  ],
  "voice_over_thoughts": [
    {{
      "line_en": "Key narrator thought",
      "line_ar": "ترجمة الخاطرة بالعربية",
      "vibe": "Reflective / cynical / determined"
    }}
  ],
  "cinematic_beats": [
    {{
      "title": "Shot title",
      "visual_en": "Camera angle and visual action",
      "visual_ar": "وصف الكادر وزاوية الكاميرا بالعربية"
    }}
  ]
}}

Chapter Text:
\"\"\"{full_text[:35000]}\"\"\"
"""

    payload = {
        "model": "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m",
        "messages": [
            {"role": "system", "content": "You are a Hollywood script breakdown supervisor. Output valid JSON only. Keep descriptions visual, punchy, and complete without truncating."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 6000
    }

    content = ""
    try:
        r = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload, timeout=180)
        res_data = r.json()
        choice = res_data["choices"][0]
        content = choice["message"].get("content") or choice["message"].get("reasoning_content", "{}")
        
        parsed = None
        # Attempt 1: Standard regex match and json.loads
        m = re.search(r'\{.*\}', content, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                pass
        
        # Attempt 2: Direct json.loads
        if not parsed:
            try:
                parsed = json.loads(content)
            except Exception:
                pass
        
        # Attempt 3: json_repair for truncated / unescaped / unclosed JSON
        if not parsed:
            parsed = json_repair.loads(content)
            
        if isinstance(parsed, dict) and any(k in parsed for k in ["active_characters", "spoken_dialogue", "settings"]):
            return {"status": "success", "breakdown": parsed}
        else:
            raise ValueError("Parsed output does not contain expected keys")

    except Exception as e:
        print(f"Full scan error: {e}")
        # Last resort fallback repair
        if content:
            try:
                repaired = json_repair.loads(content)
                if isinstance(repaired, dict):
                    return {"status": "success", "breakdown": repaired}
            except Exception as re_err:
                print(f"json_repair fallback failed: {re_err}")
                
        return {"status": "error", "error": str(e), "raw": content}

# --- FULL BOOK SCAN PIPELINE (GEMINI 2.0 / 3.8 FLASH VIA AGY OR REST) ---

def extract_full_novel_text(book_id: str, max_chars: Optional[int] = None) -> str:
    book_path = find_book_path(book_id)
    if not book_path or not os.path.exists(book_path):
        raise HTTPException(status_code=404, detail=f"Book file not found: {book_id}")

    # Standard Priority 1: Bilingual / Plain TXT Files
    if book_path.lower().endswith(".txt"):
        txt_chaps = parse_txt_chapters(book_path)
        total_text = []
        cur_chars = 0
        for ch in txt_chaps:
            ch_header = f"=== {ch['title']} ==="
            if ch.get("title_ar"):
                ch_header += f" ({ch['title_ar']})"
            ch_text = f"{ch_header}\n" + "\n\n".join(ch.get("paragraphs", []))
            total_text.append(ch_text)
            cur_chars += len(ch_text)
            if max_chars and cur_chars >= max_chars:
                break
        return "\n\n=== NEXT CHAPTER ===\n\n".join(total_text)

    # Priority 2: EPUB Archives
    total_text = []
    with zipfile.ZipFile(book_path, 'r') as z:
        html_files = [f for f in z.namelist() if f.endswith(('.xhtml', '.html', '.htm'))]
        ordered_files = []
        toc_name = next((f for f in z.namelist() if f.endswith('toc.ncx')), None)
        if toc_name:
            try:
                tree = ET.fromstring(z.read(toc_name))
                for np in tree.iter('{http://www.daisy.org/z3986/2005/ncx/}navPoint'):
                    content = np.find('{http://www.daisy.org/z3986/2005/ncx/}content')
                    if content is not None:
                        src = content.get('src', '').split('#')[0]
                        matched = next((f for f in html_files if f.endswith(src)), None)
                        if matched and matched not in ordered_files:
                            ordered_files.append(matched)
            except Exception:
                pass
        
        if not ordered_files:
            ordered_files = html_files

        cur_chars = 0
        for hf in ordered_files:
            try:
                raw = z.read(hf).decode('utf-8', errors='ignore')
                soup = BeautifulSoup(raw, 'html.parser')
                paras = []
                for p in soup.find_all(['p', 'h1', 'h2', 'h3']):
                    t = p.get_text().strip()
                    if len(t) > 25 and not t.lower().startswith("oceanofpdf") and "all rights reserved" not in t.lower():
                        paras.append(t)
                if len(paras) >= 2:
                    joined = "\n\n".join(paras)
                    total_text.append(joined)
                    cur_chars += len(joined)
                    if max_chars and cur_chars >= max_chars:
                        break
            except Exception:
                continue

    return "\n\n=== NEXT CHAPTER ===\n\n".join(total_text)

# Backwards compatibility alias
extract_full_epub_text = extract_full_novel_text

SAFETY_REFUSAL_KEYWORDS = [
    "safety", "harmful", "policy", "sexually explicit", "refuse", "cannot fulfill",
    "unable to process", "i am sorry", "i'm sorry", "against safety", "finishreason: safety",
    "blocked due to safety", "content policy", "explicit sexual", "nsfw content"
]

def is_refusal_or_blocked(text: str) -> bool:
    if not text or len(text.strip()) < 10:
        return True
    lower = text.lower()
    if ('"characters"' in lower or '"locations"' in lower) and ('{' in lower and '}' in lower):
        return False
    for kw in SAFETY_REFUSAL_KEYWORDS:
        if kw in lower:
            return True
    return False

def call_gemini_engine(prompt: str, model: str = "gemini-3.8-flash-low") -> str:
    # 1. First try Antigravity CLI (Logged in via browser)
    agy_exec = find_agy_executable()
    if agy_exec:
        try:
            proc = subprocess.Popen(
                [agy_exec, "--model", model, "--disable-slash-commands"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
            stdout, stderr = proc.communicate(input=prompt)
            if proc.returncode == 0 and stdout.strip():
                return stdout.strip()
            print(f"AGY call failed (code {proc.returncode}): {stderr}")
        except Exception as e:
            print(f"Error calling AGY: {e}")

    # 2. Fallback to Google AI Studio API Key if configured
    cfg = load_gemini_config()
    api_key = cfg.get("api_key")
    if api_key:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}
        }
        res = requests.post(url, headers=headers, json=payload, timeout=180)
        if res.status_code == 200:
            candidates = res.json().get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "")
        raise RuntimeError(f"Gemini API error ({res.status_code}): {res.text}")

    raise RuntimeError("No Gemini engine available. Antigravity CLI not found and no API key configured.")

def call_local_uncensored_engine(prompt: str, model: Optional[str] = None) -> str:
    chosen_model = model or get_best_local_uncensored_model()
    payload = {
        "model": chosen_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert literary researcher, script supervisor, and narrative analyst. "
                    "Extract the requested data strictly according to instructions in valid JSON format. "
                    "Include all character physical details, costumes, settings, and environments exactly as written in the text. "
                    "Provide translations into Arabic as instructed. Do not censor or refuse. Output JSON directly."
                )
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 4096
    }
    r = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"LM Studio local uncensored error ({r.status_code}): {r.text}")
    res_data = r.json()
    choice = res_data["choices"][0]
    content = choice["message"].get("content") or choice["message"].get("reasoning_content") or ""
    return content.strip()

def chunk_text(text: str, max_chunk_size: int = 35000) -> List[str]:
    chapters = text.split("=== NEXT CHAPTER ===")
    chunks = []
    current_chunk = []
    current_len = 0
    for ch in chapters:
        ch_len = len(ch)
        if current_len + ch_len > max_chunk_size and current_chunk:
            chunks.append("\n\n=== NEXT CHAPTER ===\n\n".join(current_chunk))
            current_chunk = [ch]
            current_len = ch_len
        else:
            current_chunk.append(ch)
            current_len += ch_len
    if current_chunk:
        chunks.append("\n\n=== NEXT CHAPTER ===\n\n".join(current_chunk))
    return chunks

def chunk_novel_into_batches(text: str, max_chars_per_batch: int = 220000) -> List[str]:
    chapters = text.split("=== NEXT CHAPTER ===")
    if not chapters or len(chapters) == 1:
        if len(text) <= max_chars_per_batch:
            return [text]
        return [text[i:i+max_chars_per_batch] for i in range(0, len(text), max_chars_per_batch)]

    batches = []
    current_batch = []
    current_len = 0
    for ch in chapters:
        ch_len = len(ch)
        if current_len + ch_len > max_chars_per_batch and current_batch:
            batches.append("\n\n=== NEXT CHAPTER ===\n\n".join(current_batch))
            current_batch = [ch]
            current_len = ch_len
        else:
            current_batch.append(ch)
            current_len += ch_len
    if current_batch:
        batches.append("\n\n=== NEXT CHAPTER ===\n\n".join(current_batch))
    return batches

def extract_with_local_uncensored(prompt_template: str, text: str, pass_key: str, local_model: Optional[str] = None, max_chunks: int = 6) -> list:
    chunks = chunk_text(text, max_chunk_size=35000)
    chunks_to_process = chunks[:max_chunks]
    all_items = []
    seen_names = set()
    for idx, ch in enumerate(chunks_to_process):
        p = prompt_template.replace("{novel_text}", ch)
        raw = call_local_uncensored_engine(p, model=local_model)
        parsed = parse_llm_json(raw)
        items = parsed.get(pass_key, [])
        for it in items:
            name = (it.get("name_en") or it.get("name", "")).strip().lower()
            if name and name not in seen_names:
                seen_names.add(name)
                all_items.append(it)
    return all_items

def parse_llm_json(raw: str) -> dict:
    clean = raw.strip()
    match = re.search(r"(\{.*\})", clean, re.DOTALL)
    if match:
        clean = match.group(1)
    try:
        return json.loads(clean)
    except Exception:
        return json_repair.loads(clean)

CHAR_PROMPT_TEMPLATE = """You are a literary analyst and narrative researcher.
Analyze the following novel text thoroughly and extract ALL characters mentioned in the story (Pass 1: Master Characters).
CRITICAL DIRECTIVE: DO NOT generate AI image generation prompts or TTS voice design prompts.
Extract strictly what is written in the book text, providing accurate descriptions in BOTH English and Arabic.
Do not censor or omit any character details, romantic relationships, anatomical features, or clothing styles mentioned in the text.

For each character, return a JSON object with:
- "name_en": Character's primary English name as written in this novel text
- "name_ar": Character's name in Arabic transliteration (الاسم باللغة العربية)
- "aliases_en": List of English nicknames, titles, or alternative names
- "aliases_ar": List of Arabic nicknames or titles
- "gender": "male" / "female" / "unknown"
- "gender_ar": "ذكر" / "أنثى" / "غير محدد"
- "role": "protagonist" / "antagonist" / "supporting" / "minor"
- "role_ar": "بطل" / "شخصية رئيسية" / "شخصية مساعدة" / "خصم" / "شخصية ثانوية"
- "physical_description_en": Physical appearance, age, build, facial features, hair, and noticeable traits directly described in the novel text.
- "physical_description_ar": ترجمة عربية دقيقة وتفصيلية للوصف الجسدي ومظهر الشخصية وملامحها المذكورة في الرواية.
- "summary_en": 1-2 sentence overview of the character's narrative role and background.
- "summary_ar": نبذة موجزة (جملة أو جملتان) عن دور الشخصية وخلفيتها في الرواية باللغة العربية.
- "costume_en": Clothing, signature garments, or winter gear described in text.
- "costume_ar": وصف الملابس والأزياء المذكورة للشخصية في النص باللغة العربية.

Return a STRICT JSON object:
{
  "characters": [
    {
      "name_en": "...",
      "name_ar": "...",
      "aliases_en": [],
      "aliases_ar": [],
      "gender": "...",
      "gender_ar": "...",
      "role": "...",
      "role_ar": "...",
      "physical_description_en": "...",
      "physical_description_ar": "...",
      "summary_en": "...",
      "summary_ar": "...",
      "costume_en": "...",
      "costume_ar": "..."
    }
  ]
}

Novel Text:
\"\"\"
{novel_text}
\"\"\"
"""

ENV_PROMPT_TEMPLATE = """You are a literary analyst and narrative researcher.
Analyze the following novel text thoroughly and extract ALL key locations, settings, and environments across the story (Pass 2: Master Environments).
CRITICAL DIRECTIVE: DO NOT generate AI image generation prompts.
Extract strictly what is written in the book text, providing accurate descriptions in BOTH English and Arabic.
Do not censor or omit any setting descriptions, rooms, or atmospheric elements.

For each environment, return a JSON object with:
- "name_en": Distinct location name in English (e.g. "Homestead Cabin - Kitchen & Living Area", "Dead Pine Forest Trail")
- "name_ar": اسم المكان باللغة العربية (e.g. "كابينة الهومستيد - المطبخ وغرفة المعيشة")
- "description_en": Detailed architectural, structural, and spatial description of the location as described in the book.
- "description_ar": وصف دقيق ومفصل للمكان وتصميمه الداخلي أو الخارجي ومعالمه كما ورد في الرواية باللغة العربية.
- "atmosphere_en": Lighting conditions, weather, temperature, and sensory atmosphere described in text.
- "atmosphere_ar": وصف الإضاءة، الطقس، الأجواء العامة، ودرجة الحرارة كما وردت في الرواية باللغة العربية.
- "props_en": List of key physical objects, furniture, weapons, or survival items present in this setting.
- "props_ar": قائمة بالأدوات، الأثاث، والأسلحة أو المعدات الرئيسية المتواجدة في المكان باللغة العربية.
- "significance_en": Why this location matters in the story.
- "significance_ar": أهمية هذا المكان وسياقه في أحداث الرواية باللغة العربية.

Return a STRICT JSON object:
{
  "locations": [
    {
      "name_en": "...",
      "name_ar": "...",
      "description_en": "...",
      "description_ar": "...",
      "atmosphere_en": "...",
      "atmosphere_ar": "...",
      "props_en": [],
      "props_ar": [],
      "significance_en": "...",
      "significance_ar": "..."
    }
  ]
}

Novel Text:
\"\"\"
{novel_text}
\"\"\"
"""

SYNONYM_STEMS = {
    'cabbie': 'driver',
    'cab': 'driver',
    'xeno': 'alien',
    'marketplace': 'market',
    'intruder': 'stalker',
    'shrubbery': 'backyard',
    'counter': 'kitchen',
    'veranda': 'balcony',
    'terrace': 'balcony',
}

def norm_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').replace("–", "-").replace("—", "-")
    s = re.sub(r'^(the|a|an)\s+', '', s, flags=re.IGNORECASE)
    return re.sub(r'[^a-zA-Z0-9]', '', s).lower()

def strip_title(s: str) -> str:
    if not s:
        return ""
    s = s.replace("’", "'").replace("‘", "'")
    s = re.sub(r'^(the|a|an)\s+', '', s, flags=re.IGNORECASE)
    s = re.sub(r'^(mr|mrs|ms|dr|doctor|commissioner|officer|detective|sergeant|lieutenant|captain|chief|lord|lady|king|queen|prince|princess|councilman|councilwoman)\.?\s+', '', s, flags=re.IGNORECASE)
    return s.strip()

def norm_ar(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r'[\u064B-\u065F\u0670]', '', s) # tashkeel
    s = re.sub(r'[إأآا]', 'ا', s)
    s = re.sub(r'ة', 'ه', s)
    s = re.sub(r'ى', 'ي', s)
    return re.sub(r'[^\w]', '', s)

def canonical_tokens(s: str) -> set:
    s = s.lower().replace("’", "'").replace("&", " and ")
    s = re.sub(r'\(.*?\)', '', s)
    s = re.sub(r'^(the|a|an)\s+', '', s)
    words = re.findall(r'[a-z0-9]+', s)
    stop = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'at', 'to', 'for', 'with', 'on', 'playing', 'man', 'area', 'exterior', 'interior'}
    res = set()
    for w in words:
        if w in stop or len(w) <= 2:
            continue
        w = SYNONYM_STEMS.get(w, w)
        if w.endswith('ing') and len(w) > 5:
            w = w[:-3]
        elif w.endswith('er') and len(w) > 5:
            w = w[:-2]
        res.add(w)
    return res

GENERIC_ALIASES = {
    'mom', 'dad', 'doc', 'alien', 'cabbie', 'cop', 'the doc', 'nameless', 'master', 
    'commissioner', 'the xeno', 'xeno', 'the intruder', 'the suspect', 'officer',
    'aquamancer officer', 'roasted manticore vendor', 'the space wizard'
}

def characters_match(c1: dict, c2: dict) -> bool:
    n1 = c1.get('name') or c1.get('name_en') or ''
    n2 = c2.get('name') or c2.get('name_en') or ''
    if not n1 or not n2:
        return False
    
    norm1 = norm_text(n1)
    norm2 = norm_text(n2)
    
    # 1. Exact normalized name
    if norm1 and norm1 == norm2:
        return True
    
    # 2. Title-stripped match
    t1 = norm_text(strip_title(n1))
    t2 = norm_text(strip_title(n2))
    if t1 and t2 and len(t1) >= 4 and len(t2) >= 4 and (t1 == t2 or t1 == norm2 or t2 == norm1):
        return True
        
    # 3. Aliases cross match
    aliases1 = [a for a in (c1.get('aliases') or []) if a]
    aliases2 = [a for a in (c2.get('aliases') or []) if a]
    norm_aliases1 = {norm_text(a) for a in aliases1 if norm_text(a)}
    norm_aliases2 = {norm_text(a) for a in aliases2 if norm_text(a)}
    
    if norm1 in norm_aliases2 and norm1 not in GENERIC_ALIASES:
        return True
    if norm2 in norm_aliases1 and norm2 not in GENERIC_ALIASES:
        return True
    if t1 in norm_aliases2 and t1 not in GENERIC_ALIASES:
        return True
    if t2 in norm_aliases1 and t2 not in GENERIC_ALIASES:
        return True
        
    # Shared specific alias (must be specific, not generic job/role)
    shared = norm_aliases1.intersection(norm_aliases2)
    specific_shared = [a for a in shared if a not in GENERIC_ALIASES and len(a) >= 5]
    if specific_shared:
        is_p1 = n1.istitle() and ' ' not in n1
        is_p2 = n2.istitle() and ' ' not in n2
        if not (is_p1 and is_p2 and norm1 != norm2):
            return True
        
    # 4. Arabic name match
    ar1 = norm_ar(c1.get('name_ar') or '')
    ar2 = norm_ar(c2.get('name_ar') or '')
    if ar1 and ar2 and ar1 == ar2 and len(ar1) >= 4:
        return True
        
    # 5. Canonical tokens overlap (e.g. descriptive minor characters)
    tok1 = canonical_tokens(n1)
    tok2 = canonical_tokens(n2)
    if tok1 and tok2:
        overlap = tok1.intersection(tok2)
        distinguishing = {'flute', 'pink', 'horned', 'goat', 'satyr', 'centaur', 'golem', 'gnoll'}
        d1 = tok1.intersection(distinguishing)
        d2 = tok2.intersection(distinguishing)
        if d1 and d2 and d1 != d2:
            return False
            
        if len(overlap) >= 2 and (len(overlap) == len(tok1) or len(overlap) == len(tok2) or len(overlap) / len(tok1.union(tok2)) >= 0.5):
            return True

    return False

def merge_character_records(existing: dict, new_item: dict) -> dict:
    is_locked = existing.get('is_locked', False) or new_item.get('is_locked', False)
    
    e_name = existing.get('name') or existing.get('name_en') or ''
    n_name = new_item.get('name') or new_item.get('name_en') or ''
    primary_name = e_name if len(e_name) >= len(n_name) else n_name
    alt_name = n_name if primary_name == e_name else e_name
    
    e_ar = existing.get('name_ar') or ''
    n_ar = new_item.get('name_ar') or ''
    name_ar = e_ar if (e_ar and len(e_ar) >= len(n_ar)) else (n_ar or e_ar)
    
    aliases = list(existing.get('aliases') or []) + list(new_item.get('aliases') or []) + list(new_item.get('aliases_en') or [])
    if alt_name and alt_name != primary_name:
        aliases.append(alt_name)
    seen_aliases = set()
    clean_aliases = []
    for a in aliases:
        if a and norm_text(a) not in seen_aliases and norm_text(a) != norm_text(primary_name):
            seen_aliases.add(norm_text(a))
            clean_aliases.append(a.strip())
            
    aliases_ar = list(existing.get('aliases_ar') or []) + list(new_item.get('aliases_ar') or [])
    seen_aliases_ar = set()
    clean_aliases_ar = []
    for a in aliases_ar:
        if a and a.strip() not in seen_aliases_ar:
            seen_aliases_ar.add(a.strip())
            clean_aliases_ar.append(a.strip())
            
    phys_en = existing.get('physical_en') or existing.get('physical_description_en') or ''
    new_phys_en = new_item.get('physical_en') or new_item.get('physical_description_en') or ''
    if len(new_phys_en) > len(phys_en):
        phys_en = new_phys_en
        
    phys_ar = existing.get('physical_ar') or existing.get('physical_description_ar') or ''
    new_phys_ar = new_item.get('physical_ar') or new_item.get('physical_description_ar') or ''
    if len(new_phys_ar) > len(phys_ar):
        phys_ar = new_phys_ar
        
    summary_en = existing.get('summary_en') or existing.get('summary') or ''
    new_summary_en = new_item.get('summary_en') or new_item.get('summary') or ''
    if len(new_summary_en) > len(summary_en):
        summary_en = new_summary_en
        
    summary_ar = existing.get('summary_ar') or ''
    new_summary_ar = new_item.get('summary_ar') or ''
    if len(new_summary_ar) > len(summary_ar):
        summary_ar = new_summary_ar
        
    outfits = existing.get('outfits') or []
    new_outfits = new_item.get('outfits') or []
    new_costume = new_item.get('costume_en') or new_item.get('costume')
    if new_costume and not any(o.get('title') == 'Signature Outfit' for o in outfits):
        outfits.append({
            "id": f"outfit_{os.urandom(3).hex()}",
            "title": "Signature Outfit",
            "title_ar": "الزي الأساسي",
            "outfit_en": new_costume,
            "outfit_ar": new_item.get('costume_ar', ''),
            "chapter_num": 1
        })
    seen_outfit_content = {norm_text(o.get('outfit_en', '')[:40]) for o in outfits if o.get('outfit_en')}
    for o in new_outfits:
        key = norm_text(o.get('outfit_en', '')[:40])
        if key and key not in seen_outfit_content:
            seen_outfit_content.add(key)
            outfits.append(o)
            
    return {
        "id": existing.get("id"),
        "name": primary_name,
        "name_ar": name_ar,
        "gender": existing.get("gender") or new_item.get("gender") or "unknown",
        "gender_ar": existing.get("gender_ar") or new_item.get("gender_ar") or "",
        "role": existing.get("role") or new_item.get("role") or "supporting",
        "role_ar": existing.get("role_ar") or new_item.get("role_ar") or "",
        "aliases": clean_aliases,
        "aliases_ar": clean_aliases_ar,
        "physical_en": phys_en,
        "physical_ar": phys_ar,
        "summary_en": summary_en,
        "summary_ar": summary_ar,
        "is_locked": is_locked,
        "outfits": outfits,
        "voice_profile": existing.get("voice_profile") or new_item.get("voice_profile") or {}
    }

ROOM_CATEGORIES = {
    'capsule': 'capsule_chamber',
    'rebirth': 'capsule_chamber',
    'subterranean': 'testing_vault',
    'testing': 'testing_vault',
    'vault': 'testing_vault',
    'living': 'living_room',
    'kitchen': 'living_room',
    'counter': 'living_room',
    'bedroom': 'bedroom',
    'studio': 'bedroom',
    'sanctuary': 'bedroom',
    'guest': 'guest_bedroom',
    'bathroom': 'bathroom',
    'backyard': 'backyard',
    'shrubbery': 'backyard',
    'balcony': 'balcony',
    'veranda': 'balcony',
    'terrace': 'balcony',
    'department': 'department_store',
    'concourse': 'department_store',
    'dressing': 'department_store',
    'rooftop': 'rooftop',
    'spotlight': 'rooftop',
    'signal': 'rooftop',
    'park': 'park',
    'ground': 'first_floor',
    'first': 'first_floor',
    'second': 'second_floor',
    'alleyway': 'alley',
    'alley': 'alley',
    'ledge': 'alley',
    'service': 'alley',
    'stall': 'plant_stall',
    'plant': 'plant_stall',
    'market': 'market',
    'marketplace': 'market',
    'dining': 'vip_dining',
    'restaurant': 'vip_dining',
    'vip': 'vip_dining',
    'executive': 'vip_dining',
    'office': 'office',
    'clinic': 'lab_clinic',
    'laboratory': 'lab_clinic',
    'residence': 'lab_clinic',
    'highway': 'border_highway',
    'elderheart': 'border_highway',
    'torture': 'border_highway',
    'donuts': 'donuts_cafe',
    'bakery': 'donuts_cafe',
    'cafe': 'donuts_cafe'
}

def extract_main_and_sub(name: str):
    parts = re.split(r'\s*[-–—/:]\s*', name)
    if len(parts) >= 2:
        return parts[0].strip(), " - ".join(parts[1:]).strip()
    return name.strip(), ""

def get_loc_cat(s: str):
    words = canonical_tokens(s)
    if 'guest' in words:
        return 'guest_bedroom'
    if 'plant' in words and ('stall' in words or 'combat' in words):
        return 'plant_stall'
    if 'alley' in words or 'ledge' in words:
        return 'alley'
    if 'office' in words:
        return 'office'
    if 'rooftop' in words or 'spotlight' in words or 'signal' in words:
        return 'rooftop'
    for w in words:
        if w in ROOM_CATEGORIES:
            return ROOM_CATEGORIES[w]
    return None

def locations_match(l1: dict, l2: dict) -> bool:
    n1 = l1.get('name') or l1.get('name_en') or ''
    n2 = l2.get('name') or l2.get('name_en') or ''
    if not n1 or not n2:
        return False
        
    norm1 = norm_text(n1)
    norm2 = norm_text(n2)
    if norm1 == norm2:
        return True
        
    main1, sub1 = extract_main_and_sub(n1)
    main2, sub2 = extract_main_and_sub(n2)
    
    cat1 = get_loc_cat(n1)
    cat2 = get_loc_cat(n2)
    
    # House aliases
    is_house1 = any(h in norm1 for h in ['pinkypeach', 'sharedhome', 'house', 'backyard'])
    is_house2 = any(h in norm2 for h in ['pinkypeach', 'sharedhome', 'house', 'backyard'])
    
    if is_house1 and is_house2:
        if cat1 and cat2:
            return cat1 == cat2
        if not cat1 and not cat2:
            return True
        return False
        
    # Rooftops with signals / police headquarters rooftop
    if cat1 == 'rooftop' and cat2 == 'rooftop':
        return True

    # Parks
    if cat1 == 'park' and cat2 == 'park':
        return True

    # Marketplaces
    if 'goblin' in norm1 and 'goblin' in norm2:
        if ('market' in norm1 or 'marketplace' in norm1) and ('market' in norm2 or 'marketplace' in norm2):
            return True
        if cat1 and cat2:
            return cat1 == cat2

    # Same primary venue
    w_main1 = canonical_tokens(main1)
    w_main2 = canonical_tokens(main2)
    main_overlap = w_main1.intersection(w_main2)
    
    if len(main_overlap) >= 2 or (w_main1 and w_main1 == w_main2):
        if cat1 and cat2:
            return cat1 == cat2
        return True

    # Whole string canonical tokens
    tok1 = canonical_tokens(n1)
    tok2 = canonical_tokens(n2)
    if tok1 and tok2:
        overlap = tok1.intersection(tok2)
        jaccard = len(overlap) / len(tok1.union(tok2))
        if jaccard >= 0.55:
            return True

    return False

def merge_location_records(existing: dict, new_item: dict) -> dict:
    is_locked = existing.get('is_locked', False) or new_item.get('is_locked', False)
    
    e_name = existing.get('name') or existing.get('name_en') or ''
    n_name = new_item.get('name') or new_item.get('name_en') or ''
    primary_name = e_name if len(e_name) >= len(n_name) else n_name
    
    e_ar = existing.get('name_ar') or ''
    n_ar = new_item.get('name_ar') or ''
    name_ar = e_ar if (e_ar and len(e_ar) >= len(n_ar)) else (n_ar or e_ar)
    
    setting_en = existing.get('setting_en') or existing.get('description_en') or ''
    new_setting_en = new_item.get('setting_en') or new_item.get('description_en') or ''
    if len(new_setting_en) > len(setting_en):
        setting_en = new_setting_en
        
    setting_ar = existing.get('setting_ar') or existing.get('description_ar') or ''
    new_setting_ar = new_item.get('setting_ar') or new_item.get('description_ar') or ''
    if len(new_setting_ar) > len(setting_ar):
        setting_ar = new_setting_ar
        
    atmosphere_en = existing.get('atmosphere_en') or ''
    new_atmosphere_en = new_item.get('atmosphere_en') or ''
    if len(new_atmosphere_en) > len(atmosphere_en):
        atmosphere_en = new_atmosphere_en
        
    atmosphere_ar = existing.get('atmosphere_ar') or ''
    new_atmosphere_ar = new_item.get('atmosphere_ar') or ''
    if len(new_atmosphere_ar) > len(atmosphere_ar):
        atmosphere_ar = new_atmosphere_ar
        
    significance_en = existing.get('significance_en') or ''
    new_significance_en = new_item.get('significance_en') or ''
    if len(new_significance_en) > len(significance_en):
        significance_en = new_significance_en
        
    significance_ar = existing.get('significance_ar') or ''
    new_significance_ar = new_item.get('significance_ar') or ''
    if len(new_significance_ar) > len(significance_ar):
        significance_ar = new_significance_ar
        
    props_en = list(existing.get('props_en') or []) + list(new_item.get('props_en') or [])
    seen_props = set()
    clean_props_en = []
    for p in props_en:
        if p and norm_text(p) not in seen_props:
            seen_props.add(norm_text(p))
            clean_props_en.append(p.strip())
            
    props_ar = list(existing.get('props_ar') or []) + list(new_item.get('props_ar') or [])
    seen_props_ar = set()
    clean_props_ar = []
    for p in props_ar:
        if p and p.strip() not in seen_props_ar:
            seen_props_ar.add(p.strip())
            clean_props_ar.append(p.strip())
            
    angles = existing.get('angles') or []
    new_angles = new_item.get('angles') or []
    for a in new_angles:
        if not any(norm_text(x.get('title', '')) == norm_text(a.get('title', '')) for x in angles):
            angles.append(a)
            
    return {
        "id": existing.get("id"),
        "name": primary_name,
        "name_ar": name_ar,
        "setting_en": setting_en,
        "setting_ar": setting_ar,
        "atmosphere_en": atmosphere_en,
        "atmosphere_ar": atmosphere_ar,
        "props_en": clean_props_en,
        "props_ar": clean_props_ar,
        "significance_en": significance_en,
        "significance_ar": significance_ar,
        "is_locked": is_locked,
        "angles": angles
    }

class ScanBookGeminiRequest(BaseModel):
    book_id: str
    mode: Optional[str] = "all"  # "all", "characters", "locations"
    scope: Optional[str] = "full"  # "full" or "preview_5"
    engine: Optional[str] = "hybrid"  # "hybrid", "local_uncensored", "gemini"
    model: Optional[str] = "gemini-3.8-flash-low"
    local_model: Optional[str] = None

@app.post("/api/scan_book_gemini")
def scan_book_gemini_endpoint(req: ScanBookGeminiRequest):
    novel_text = extract_full_epub_text(req.book_id, max_chars=None)
    if not novel_text:
        raise HTTPException(status_code=400, detail="Could not extract readable text from book")

    all_batches = chunk_novel_into_batches(novel_text, max_chars_per_batch=220000)
    if req.scope == "preview_5":
        batches_to_scan = all_batches[:1]
    else:
        # Full scan: process up to 6 batches (~1.3M chars, covering 60+ chapters of omnibuses, or 100% of standard novels)
        batches_to_scan = all_batches[:6]

    db = load_db(req.book_id)
    results = {
        "characters": [], 
        "locations": [], 
        "status": "success",
        "engine_used": req.engine,
        "explicit_safety_fallback": False,
        "batches_processed": len(batches_to_scan)
    }

    # Pass 1: Master Characters
    if req.mode in ["all", "characters"]:
        all_extracted_chars = []
        seen_char_names = set()

        for b_idx, batch_text in enumerate(batches_to_scan):
            chars_list = []
            if req.engine == "local_uncensored":
                print(f"[Engine: Local Uncensored] Batch {b_idx+1}/{len(batches_to_scan)} via {req.local_model or get_best_local_uncensored_model()}...")
                chars_list = extract_with_local_uncensored(CHAR_PROMPT_TEMPLATE, batch_text, "characters", local_model=req.local_model, max_chunks=4)
                results["engine_used"] = "local_uncensored"
            else:
                use_fallback = False
                try:
                    char_prompt = CHAR_PROMPT_TEMPLATE.replace("{novel_text}", batch_text)
                    raw_char_res = call_gemini_engine(char_prompt, model=req.model)
                    if is_refusal_or_blocked(raw_char_res):
                        print(f"[Safety Warning] Gemini batch {b_idx+1} flagged: {raw_char_res[:200]}")
                        use_fallback = True
                    else:
                        parsed_chars = parse_llm_json(raw_char_res)
                        chars_list = parsed_chars.get("characters", [])
                        if not chars_list:
                            use_fallback = True
                except Exception as e:
                    print(f"[Gemini Exception Batch {b_idx+1}] {e}")
                    if req.engine == "hybrid":
                        use_fallback = True
                    else:
                        raise HTTPException(status_code=500, detail=f"Gemini engine error: {str(e)}")

                if use_fallback and req.engine == "hybrid":
                    print(f"[Safety Bypass Active] Routing batch {b_idx+1} to Local Uncensored Model...")
                    chars_list = extract_with_local_uncensored(CHAR_PROMPT_TEMPLATE, batch_text, "characters", local_model=req.local_model, max_chunks=4)
                    results["engine_used"] = "local_uncensored_fallback"
                    results["explicit_safety_fallback"] = True

            for c in chars_list:
                c_name_en = (c.get("name_en") or c.get("name", "")).strip()
                if c_name_en and c_name_en.lower() not in seen_char_names:
                    seen_char_names.add(c_name_en.lower())
                    all_extracted_chars.append(c)

        results["characters"] = all_extracted_chars

        # Merge into database using universal entity resolution
        for c in all_extracted_chars:
            c_name_en = (c.get("name_en") or c.get("name", "")).strip()
            if not c_name_en:
                continue

            matched_idx = -1
            for idx, existing in enumerate(db["characters"]):
                if characters_match(c, existing):
                    matched_idx = idx
                    break

            if matched_idx >= 0:
                db["characters"][matched_idx] = merge_character_records(db["characters"][matched_idx], c)
            else:
                new_entry = {
                    "id": f"char_{len(db['characters']) + 1}",
                    "name": c_name_en,
                    "name_ar": c.get("name_ar", ""),
                    "gender": c.get("gender", "unknown"),
                    "gender_ar": c.get("gender_ar", ""),
                    "role": c.get("role", "supporting"),
                    "role_ar": c.get("role_ar", ""),
                    "aliases": c.get("aliases_en") or c.get("aliases") or [],
                    "aliases_ar": c.get("aliases_ar", []),
                    "physical_en": c.get("physical_description_en") or c.get("physical_en") or "",
                    "physical_ar": c.get("physical_description_ar") or c.get("physical_ar") or "",
                    "summary_en": c.get("summary_en") or c.get("summary") or "",
                    "summary_ar": c.get("summary_ar", ""),
                    "is_locked": False,
                    "outfits": []
                }
                costume = c.get("costume_en") or c.get("costume")
                if costume:
                    new_entry["outfits"].append({
                        "id": f"outfit_{os.urandom(3).hex()}",
                        "title": "Signature Outfit",
                        "title_ar": "الزي الأساسي",
                        "outfit_en": costume,
                        "outfit_ar": c.get("costume_ar", ""),
                        "chapter_num": 1
                    })
                db["characters"].append(new_entry)

        # Re-index characters cleanly
        for idx, c in enumerate(db["characters"]):
            c["id"] = f"char_{idx + 1}"

    # Pass 2: Master Environments
    if req.mode in ["all", "locations"]:
        all_extracted_envs = []
        seen_env_names = set()

        for b_idx, batch_text in enumerate(batches_to_scan):
            envs_list = []
            if req.engine == "local_uncensored":
                print(f"[Engine: Local Uncensored] Locations batch {b_idx+1}/{len(batches_to_scan)}...")
                envs_list = extract_with_local_uncensored(ENV_PROMPT_TEMPLATE, batch_text, "locations", local_model=req.local_model, max_chunks=4)
                results["engine_used"] = "local_uncensored"
            else:
                use_fallback = False
                try:
                    env_prompt = ENV_PROMPT_TEMPLATE.replace("{novel_text}", batch_text)
                    raw_env_res = call_gemini_engine(env_prompt, model=req.model)
                    if is_refusal_or_blocked(raw_env_res):
                        print(f"[Safety Warning] Gemini location batch {b_idx+1} flagged: {raw_env_res[:200]}")
                        use_fallback = True
                    else:
                        parsed_envs = parse_llm_json(raw_env_res)
                        envs_list = parsed_envs.get("locations", [])
                        if not envs_list:
                            use_fallback = True
                except Exception as e:
                    print(f"[Gemini Location Exception Batch {b_idx+1}] {e}")
                    if req.engine == "hybrid":
                        use_fallback = True
                    else:
                        raise HTTPException(status_code=500, detail=f"Gemini engine error: {str(e)}")

                if use_fallback and req.engine == "hybrid":
                    print(f"[Safety Bypass Active] Routing locations batch {b_idx+1} to Local Uncensored Model...")
                    envs_list = extract_with_local_uncensored(ENV_PROMPT_TEMPLATE, batch_text, "locations", local_model=req.local_model, max_chunks=4)
                    results["engine_used"] = "local_uncensored_fallback"
                    results["explicit_safety_fallback"] = True

            for l in envs_list:
                l_name_en = (l.get("name_en") or l.get("name", "")).strip()
                if l_name_en and l_name_en.lower() not in seen_env_names:
                    seen_env_names.add(l_name_en.lower())
                    all_extracted_envs.append(l)

        results["locations"] = all_extracted_envs

        # Merge into database using universal entity resolution
        for l in all_extracted_envs:
            l_name_en = (l.get("name_en") or l.get("name", "")).strip()
            if not l_name_en:
                continue

            matched_idx = -1
            for idx, existing in enumerate(db["locations"]):
                if locations_match(l, existing):
                    matched_idx = idx
                    break

            if matched_idx >= 0:
                db["locations"][matched_idx] = merge_location_records(db["locations"][matched_idx], l)
            else:
                new_entry = {
                    "id": f"loc_{len(db['locations']) + 1}",
                    "name": l_name_en,
                    "name_ar": l.get("name_ar", ""),
                    "setting_en": l.get("description_en") or l.get("setting_en") or "",
                    "setting_ar": l.get("description_ar") or l.get("setting_ar") or "",
                    "atmosphere_en": l.get("atmosphere_en", ""),
                    "atmosphere_ar": l.get("atmosphere_ar", ""),
                    "props_en": l.get("props_en", []),
                    "props_ar": l.get("props_ar", []),
                    "significance_en": l.get("significance_en", ""),
                    "significance_ar": l.get("significance_ar", ""),
                    "is_locked": False,
                    "angles": []
                }
                db["locations"].append(new_entry)

        # Re-index locations cleanly
        for idx, l in enumerate(db["locations"]):
            l["id"] = f"loc_{idx + 1}"

    save_db(db, req.book_id)
    results["db"] = db
    return results

class DeduplicateDbRequest(BaseModel):
    book_id: Optional[str] = None

@app.post("/api/database/deduplicate")
def deduplicate_database_endpoint(req: DeduplicateDbRequest):
    db = load_db(req.book_id)
    before_chars = len(db.get("characters", []))
    before_locs = len(db.get("locations", []))

    clean_chars = []
    for c in db.get("characters", []):
        matched_idx = -1
        for idx, existing in enumerate(clean_chars):
            if characters_match(c, existing):
                matched_idx = idx
                break
        if matched_idx >= 0:
            clean_chars[matched_idx] = merge_character_records(clean_chars[matched_idx], c)
        else:
            clean_chars.append(c)

    for idx, c in enumerate(clean_chars):
        c["id"] = f"char_{idx + 1}"

    clean_locs = []
    for l in db.get("locations", []):
        matched_idx = -1
        for idx, existing in enumerate(clean_locs):
            if locations_match(l, existing):
                matched_idx = idx
                break
        if matched_idx >= 0:
            clean_locs[matched_idx] = merge_location_records(clean_locs[matched_idx], l)
        else:
            clean_locs.append(l)

    for idx, l in enumerate(clean_locs):
        l["id"] = f"loc_{idx + 1}"

    db["characters"] = clean_chars
    db["locations"] = clean_locs
    save_db(db, req.book_id)

    return {
        "status": "success",
        "book_id": req.book_id,
        "before": {"characters": before_chars, "locations": before_locs},
        "after": {"characters": len(clean_chars), "locations": len(clean_locs)},
        "removed": {
            "characters": before_chars - len(clean_chars),
            "locations": before_locs - len(clean_locs)
        },
        "db": db
    }

class ToggleLockRequest(BaseModel):
    book_id: Optional[str] = None
    asset_type: str  # "character" or "location"
    asset_id: str
    is_locked: bool

@app.post("/api/database/toggle_lock")
def toggle_lock_endpoint(req: ToggleLockRequest):
    db = load_db(req.book_id)
    target_list = db["characters"] if req.asset_type == "character" else db["locations"]
    item = next((x for x in target_list if x.get("id") == req.asset_id or x.get("name") == req.asset_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Asset not found")
    item["is_locked"] = req.is_locked
    save_db(db, req.book_id)
    return {"status": "success", "asset": item}

# --- ASSET BIBLE DATABASE APIS (SCOPED PER BOOK) ---

@app.get("/api/database")
def get_database(book_id: Optional[str] = None):
    return load_db(book_id)

class SaveCharacterRequest(BaseModel):
    name: str
    physical_en: str
    physical_ar: Optional[str] = ""
    role: Optional[str] = ""
    voice_profile: Optional[str] = ""
    is_locked: Optional[bool] = False
    book_id: Optional[str] = None

@app.post("/api/database/character")
def save_character(req: SaveCharacterRequest):
    db = load_db(req.book_id)
    # Check if exists
    existing = next((c for c in db["characters"] if c["name"].lower() == req.name.lower()), None)
    if existing:
        existing["physical_en"] = req.physical_en
        if req.physical_ar: existing["physical_ar"] = req.physical_ar
        if req.role: existing["role"] = req.role
        if req.voice_profile: existing["voice_profile"] = req.voice_profile
        if req.is_locked is not None: existing["is_locked"] = req.is_locked
        save_db(db, req.book_id)
        return {"status": "updated", "character": existing}
    
    new_char = {
        "id": f"char_{len(db['characters']) + 1}",
        "name": req.name,
        "role": req.role or "supporting",
        "physical_en": req.physical_en,
        "physical_ar": req.physical_ar or "",
        "voice_profile": req.voice_profile or "",
        "is_locked": req.is_locked or False,
        "outfits": []
    }
    db["characters"].append(new_char)
    save_db(db, req.book_id)
    return {"status": "created", "character": new_char}

class SaveOutfitRequest(BaseModel):
    character_name: str
    outfit_title: str
    outfit_en: str
    outfit_ar: Optional[str] = ""
    chapter_num: Optional[int] = 1
    book_id: Optional[str] = None

@app.post("/api/database/outfit")
def save_outfit(req: SaveOutfitRequest):
    db = load_db(req.book_id)
    char = next((c for c in db["characters"] if c["name"].lower() == req.character_name.lower()), None)
    
    outfit_entry = {
        "id": f"outfit_{os.urandom(3).hex()}",
        "title": req.outfit_title,
        "outfit_en": req.outfit_en,
        "outfit_ar": req.outfit_ar or "",
        "chapter_num": req.chapter_num
    }

    if not char:
        # Create character on the fly if not existing
        char = {
            "id": f"char_{len(db['characters']) + 1}",
            "name": req.character_name,
            "physical_en": "",
            "physical_ar": "",
            "voice_profile": "",
            "is_locked": False,
            "outfits": [outfit_entry]
        }
        db["characters"].append(char)
    else:
        char.setdefault("outfits", []).append(outfit_entry)

    save_db(db, req.book_id)
    return {"status": "saved", "character": char, "outfit": outfit_entry}

class SaveLocationRequest(BaseModel):
    location_name: str
    setting_en: str
    setting_ar: Optional[str] = ""
    lighting_and_atmosphere: Optional[str] = ""
    key_props: Optional[List[str]] = []
    is_locked: Optional[bool] = False
    is_new_angle: bool = False
    parent_location_name: Optional[str] = None
    book_id: Optional[str] = None

@app.post("/api/database/location")
def save_location(req: SaveLocationRequest):
    db = load_db(req.book_id)
    if req.is_new_angle and req.parent_location_name:
        parent = next((l for l in db["locations"] if l["name"].lower() == req.parent_location_name.lower()), None)
        if parent:
            angle_entry = {
                "id": f"angle_{os.urandom(3).hex()}",
                "name": req.location_name,
                "setting_en": req.setting_en,
                "setting_ar": req.setting_ar or "",
                "lighting_and_atmosphere": req.lighting_and_atmosphere or ""
            }
            parent.setdefault("angles", []).append(angle_entry)
            save_db(db, req.book_id)
            return {"status": "angle_added", "location": parent}

    existing = next((l for l in db["locations"] if l["name"].lower() == req.location_name.lower()), None)
    if existing:
        existing["setting_en"] = req.setting_en
        if req.setting_ar: existing["setting_ar"] = req.setting_ar
        if req.lighting_and_atmosphere: existing["lighting_and_atmosphere"] = req.lighting_and_atmosphere
        if req.key_props: existing["key_props"] = req.key_props
        if req.is_locked is not None: existing["is_locked"] = req.is_locked
        save_db(db, req.book_id)
        return {"status": "updated", "location": existing}

    new_loc = {
        "id": f"loc_{len(db['locations']) + 1}",
        "name": req.location_name,
        "setting_en": req.setting_en,
        "setting_ar": req.setting_ar or "",
        "lighting_and_atmosphere": req.lighting_and_atmosphere or "",
        "key_props": req.key_props or [],
        "is_locked": req.is_locked or False,
        "angles": []
    }
    db["locations"].append(new_loc)
    save_db(db, req.book_id)
    return {"status": "created", "location": new_loc}

class SaveDialogueRequest(BaseModel):
    speaker: str
    line_en: str
    line_ar: Optional[str] = ""
    tone: Optional[str] = ""
    chapter_num: Optional[int] = 1
    book_id: Optional[str] = None

@app.post("/api/database/dialogue")
def save_dialogue(req: SaveDialogueRequest):
    db = load_db(req.book_id)
    d_entry = {
        "id": f"dlg_{os.urandom(3).hex()}",
        "chapter_num": req.chapter_num,
        "speaker": req.speaker,
        "line_en": req.line_en,
        "line_ar": req.line_ar or "",
        "tone": req.tone
    }
    db.setdefault("dialogues", []).append(d_entry)
    save_db(db, req.book_id)
    return {"status": "saved", "dialogue": d_entry}

class SaveCardRequest(BaseModel):
    category: str
    card_content: str
    title: Optional[str] = "Manual Extraction"
    chapter_num: Optional[int] = 1
    book_id: Optional[str] = None

@app.post("/api/database/card")
def save_custom_card(req: SaveCardRequest):
    db = load_db(req.book_id)
    card_entry = {
        "id": f"card_{os.urandom(3).hex()}",
        "title": req.title,
        "category": req.category,
        "content": req.card_content,
        "chapter_num": req.chapter_num
    }
    db.setdefault("saved_cards", []).append(card_entry)
    save_db(db, req.book_id)
    return {"status": "saved", "card": card_entry}

# Standard manual extraction endpoint with character attribution
class ExtractRequest(BaseModel):
    category: str # character | outfit | environment | action | dialogue | voiceover | custom
    selected_text: str
    custom_prompt: Optional[str] = None
    character_name: Optional[str] = None

@app.post("/api/extract")
def extract_description(req: ExtractRequest):
    char_label = f" for Character '{req.character_name}'" if req.character_name else ""
    
    prompts = {
        "dialogue": (
            f"Analyze this passage strictly for spoken dialogue between characters on screen{char_label}.\n"
            "CRITICAL CHECK: If this passage is first-person reflection or internal thoughts (e.g., 'I was oddly suited for the end of the world...'), "
            "do NOT label it as spoken dialogue! Classify it as Voice-Over / Narration.\n\n"
            "Format:\n"
            "### 🗣️ Speakers & Context\n"
            f"- **Speaker:** {req.character_name if req.character_name else '[Identify Speaker aloud]'}\n"
            "- **Listener:** [Identify Listener / Conversational Partner]\n"
            "- **Context:** [Physical scene situation]\n\n"
            "### 💬 Exact Spoken Dialogue\n"
            "- **Lines:** \"Exact spoken words here\"\n\n"
            "### 🎭 Vocal Delivery & Subtext\n"
            "- **Voice Timbre & Tone:** (e.g. dry gravelly baritone, tense breathy whisper, sarcastic chuckle)\n"
            "- **Emotional Subtext:** (Underlying intent, vulnerability, concealed strategy, tension)\n\n"
            "### 🎬 Visual Acting & Body Language\n"
            "- **Facial Expression & Gaze:** (Eye contact, calculating smirk, tightened jaw, blink rate)\n"
            "- **Gestures & Posture:** (Arms crossed over chest, hand hovering near holster, leaning back)"
        ),
        "outfit": (
            f"Analyze this passage and extract a production costume and wardrobe breakdown{char_label}.\n"
            "Format:\n"
            "### 👗 Character & Outfit Identity\n"
            f"- **Character:** {req.character_name if req.character_name else '[Identify Character wearing this outfit]'}\n"
            "- **Outfit Title:** [e.g. Winter Survival Gear / Gas Station Arrival Attire]\n\n"
            "### 🧵 Garments & Layering\n"
            "- **Upper Body:** (Parkas, coats, fleece collars, knitted sweaters, inner thermals)\n"
            "- **Lower Body:** (Denim jeans, thermal trousers, snow gaiters, boots)\n"
            "- **Accessories & Gear:** (Backpack, straps, belts, gloves, canteen, holster)\n\n"
            "### 🎨 Palette, Materials & Wear\n"
            "- **Textures & Colors:** (Charcoal distressed canvas, faded blue denim, pale shearling)\n"
            "- **Condition & Wear:** (Frayed hems, snow dusting, grease stains, worn leather)"
        ),
        "voiceover": (
            f"Analyze this passage strictly as cinematic Voice-Over (V.O.) / Narrator Monologue{char_label}.\n"
            "Format:\n"
            "### 🎙️ Voice-Over Identity\n"
            f"- **Narrator / Speaker:** {req.character_name if req.character_name else 'Protagonist / First-Person Narrator'}\n"
            "- **Delivery Nature:** Internal reflection / Story opening narration\n\n"
            "### 📜 Narration Line\n"
            "- **Exact Text:** \"Exact passage words here\"\n\n"
            "### 🎭 Vocal Tone & Acoustic Delivery\n"
            "- **Vocal Texture & Register:** (Deep grounded baritone, quiet cynic amusement, measured pauses)\n"
            "- **Pacing & Breath:** (Slow reflective cadence, breath vapor condensation in cold air)\n\n"
            "### 🎬 Accompanying Visual Scene\n"
            "- **Visual Shot:** (Wide establishing wasteland shot, camera drifting slowly, lone silhouette in snow)"
        ),
        "character": (
            f"Analyze the character described in this passage and extract a detailed visual Character Sheet{char_label}.\n"
            "Format:\n"
            "### 👤 Character Name & Role\n"
            f"- **Name:** {req.character_name if req.character_name else '[Character Name]'}\n"
            "- **Archetype / Role:** [Protagonist / Survivor / Key Encounter]\n\n"
            "### 📌 Physical Anatomy & Features\n"
            "- **Age & Build:** (Approximate age, height, shoulder frame, physical build)\n"
            "- **Face & Distinct Traits:** (Jawline, dark purple black eye on cheekbone, split lip, stubble)\n"
            "- **Hair & Eyes:** (Color, length, disheveled styling, eye expression)\n\n"
            "### ✨ Presence & Demeanor\n"
            "- **Gaze & Expression:** (Knowing smirk, calculating gaze, exhausted resilience)\n"
            "- **Demeanor:** (Cautious composure, unbothered acceptance)"
        ),
        "environment": (
            "Extract a detailed visual setting and environment composition for production.\n"
            "Format:\n"
            "### 🏰 Setting & Architecture\n"
            "- **Location Name:** [e.g. Abandoned Gas Station Interior]\n"
            "- **Spatial Layout:** (Narrow hallway, peeling doorway, linoleum floor, rafters)\n\n"
            "### 💡 Lighting & Weather Atmosphere\n"
            "- **Lighting Scheme:** (Cold ambient blue daylight filtering through frosted windows)\n"
            "- **Atmosphere:** (Freezing temperature, drifting snow flurries, dust motes)\n\n"
            "### 📦 Props & Set Dressing\n"
            "- **Key Objects:** (Peeling notices on door, old metal cash register, iced countertops)"
        ),
        "action": (
            "Extract the dynamic cinematic action beat and camera blocking.\n"
            "Format:\n"
            "### 🎬 Scene Beat & Blocking\n"
            "- **Primary Action:** (Character movement, interaction, stance adjustment)\n"
            "- **Camera Framing & Motion:** (Medium shot, slow forward dolly, eye-level)\n"
            "- **Physical Dynamics:** (Visible breath condensation puffing, snow crunching)"
        )
    }

    if req.category == "custom" and req.custom_prompt:
        instruction = (
            f"User Specific Directive: {req.custom_prompt}\n"
            + (f"Target Character: {req.character_name}\n" if req.character_name else "")
            + "Extract and analyze the selected passage strictly according to the user directive above. "
            "Provide rich visual and cinematic details in English. Format cleanly with Markdown."
        )
    else:
        instruction = prompts.get(req.category, prompts["character"])

    system_prompt = (
        "You are an expert Hollywood script supervisor and visual media director. "
        "Analyze the provided passage strictly from the text. "
        "Output all analysis and details in English with clear Markdown headers and bullet points."
    )
    user_prompt = f"""{instruction}\n\nPassage:\n\"\"\"{req.selected_text}\"\"\""""

    payload = {
        "model": "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.25,
        "max_tokens": 1500
    }

    try:
        r = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload, timeout=60)
        res_data = r.json()
        choice = res_data["choices"][0]
        content = choice["message"].get("content") or choice["message"].get("reasoning_content", "No content")
        return {
            "result": content, 
            "category": req.category,
            "character_name": req.character_name
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- FULL NOVEL STORYBOARD PIPELINE (HOLLYWOOD SCENE CARDS) ---

STORYBOARD_PROMPT_TEMPLATE = """You are a master Hollywood film director and script supervisor.
Analyze this entire novel chapter from start to finish and break it down into a sequence of MAJOR STORYBOARD SCENE CARDS (الكروت الكبيرة للمشاهد الرئيسية).
Each card represents a distinct dramatic scene / master sequence taking place in a specific location and timeframe.
Cover the full chapter progression continuously without skipping or truncating any scenes (typically 4 to 8 full scene cards).

For each scene, output a structured JSON card with these EXACT specifications:
1. "card_number": Integer (1, 2, 3...)
2. "slugline": "INT. or EXT. LOCATION - TIME OF DAY"
3. "title_en": "Scene Title in English"
4. "title_ar": "عنوان المشهد باللغة العربية"
5. "location": {
     "name_en": "Specific Location Name",
     "name_ar": "اسم المكان بالعربية",
     "atmosphere_and_lighting": "Visual lighting, color palette, mood, temperature",
     "key_props": ["Key physical objects/props in this scene"]
   }
6. "cast_present": [
     {
       "name_en": "Character Name",
       "name_ar": "الاسم بالعربية",
       "outfit_and_condition": "Exact garments worn and physical state in THIS specific scene"
     }
   ]
7. "visual_action": {
     "action_en": "Detailed narrative description of visual actions and camera progression from start of scene to end",
     "action_ar": "وصف مرئي متتابع لما تراه الكاميرا بالعربية"
   }
8. "audio_and_dialogue": {
     "spoken_dialogue": [
       {"speaker": "Name", "line_en": "Spoken line in English", "line_ar": "الترجمة بالعربية"}
     ],
     "voice_over": "Key inner reflective narrator thought (if any)",
     "sfx_and_ambience": "Key sound effects and background environmental soundscape"
   }
9. "source_paragraphs": "e.g. Paragraphs 1-18"
10. "estimated_duration_min": "e.g. 1.5 - 2 minutes"

Output strictly valid JSON with key "storyboard_cards": [...]
Do not output markdown codeblocks.

Chapter Text:
\"\"\"
{chapter_text}
\"\"\"
"""

def get_storyboard_db_path(book_id: str) -> str:
    safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '_', book_id)
    return os.path.join(DATA_DIR, f"storyboard_{safe_id}.json")

def load_storyboard_db(book_id: str) -> Dict[str, Any]:
    p = get_storyboard_db_path(book_id)
    if not os.path.exists(p):
        return {"book_id": book_id, "chapters": {}}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"book_id": book_id, "chapters": {}}

def save_storyboard_db(book_id: str, data: Dict[str, Any]):
    p = get_storyboard_db_path(book_id)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_chapter_storyboard_sync(book_id: str, chapter_num: int, model: Optional[str] = None) -> List[Dict[str, Any]]:
    content = get_chapter_content(book_id, chapter_num)
    paras = content.get("paragraphs", [])
    if not paras:
        raise HTTPException(status_code=400, detail="Chapter is empty")

    formatted_paras = []
    for idx, p in enumerate(paras):
        lines = p.split('\n')
        en = [l for l in lines if not any('\u0600' <= c <= '\u06FF' for c in l)]
        t = ' '.join(en).strip() if en else p.strip()
        formatted_paras.append(f"[{idx+1}] {t}")

    chapter_text = "\n\n".join(formatted_paras)
    prompt = STORYBOARD_PROMPT_TEMPLATE.replace("{chapter_text}", chapter_text)

    chosen_model = model or "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp@q4_k_m"

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": "You are a veteran Hollywood director and script breakdown supervisor. Output valid JSON only without markdown code blocks."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 4096
    }

    raw = ""
    try:
        r = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload, timeout=360)
        res_data = r.json()
        if r.status_code != 200 or "choices" not in res_data:
            err_msg = res_data.get("error", {}).get("message") if isinstance(res_data, dict) and "error" in res_data else str(res_data)
            raise ValueError(f"LM Studio API Error ({r.status_code}): {err_msg}")
        choice = res_data["choices"][0]
        raw = choice["message"].get("content") or choice["message"].get("reasoning_content") or ""
        parsed = parse_llm_json(raw)
        cards = parsed.get("storyboard_cards", [])
        if not cards and isinstance(parsed, list):
            cards = parsed

        if cards:
            db = load_storyboard_db(book_id)
            db["chapters"][str(chapter_num)] = {
                "chapter_num": chapter_num,
                "chapter_title": content.get("chapter_title", f"Chapter {chapter_num}"),
                "chapter_title_ar": content.get("chapter_title_ar", ""),
                "cards": cards,
                "updated_at": __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M")
            }
            save_storyboard_db(book_id, db)
            return cards
        else:
            raise ValueError(f"No storyboard_cards found in model output: {raw[:300]}")
    except Exception as e:
        print(f"Error generating storyboard for ch {chapter_num}: {e}")
        raise HTTPException(status_code=500, detail=f"Storyboard generation error: {str(e)}")

# Background Batch State
active_storyboard_batches: Dict[str, Dict[str, Any]] = {}

class StoryboardGenRequest(BaseModel):
    book_id: str
    chapter_num: int
    model: Optional[str] = None

class StoryboardBatchRequest(BaseModel):
    book_id: str
    start_chapter: Optional[int] = 1
    end_chapter: Optional[int] = None
    model: Optional[str] = None

@app.get("/api/storyboard/{book_id}")
def get_storyboard(book_id: str):
    return load_storyboard_db(book_id)

@app.post("/api/storyboard/generate_chapter")
def generate_chapter_storyboard_endpoint(req: StoryboardGenRequest):
    cards = generate_chapter_storyboard_sync(req.book_id, req.chapter_num, req.model)
    return {
        "status": "success",
        "chapter_num": req.chapter_num,
        "cards_count": len(cards),
        "cards": cards
    }

import threading

def run_storyboard_batch_worker(book_id: str, start_ch: int, end_ch: int, model: Optional[str]):
    chapters = get_chapters(book_id)
    total_ch = len(chapters)
    end_ch = min(end_ch, total_ch)

    active_storyboard_batches[book_id] = {
        "is_running": True,
        "current_chapter": start_ch,
        "total_chapters": total_ch,
        "start_chapter": start_ch,
        "end_chapter": end_ch,
        "completed_chapters": 0,
        "total_cards_generated": 0,
        "cancelled": False,
        "error": None
    }

    try:
        for ch_idx in range(start_ch, end_ch + 1):
            if active_storyboard_batches.get(book_id, {}).get("cancelled"):
                break
            active_storyboard_batches[book_id]["current_chapter"] = ch_idx
            try:
                cards = generate_chapter_storyboard_sync(book_id, ch_idx, model)
                active_storyboard_batches[book_id]["completed_chapters"] += 1
                active_storyboard_batches[book_id]["total_cards_generated"] += len(cards)
            except Exception as ch_err:
                print(f"[Batch Ch {ch_idx} Error] {ch_err}")
                active_storyboard_batches[book_id]["error"] = str(ch_err)

    finally:
        if book_id in active_storyboard_batches:
            active_storyboard_batches[book_id]["is_running"] = False

@app.post("/api/storyboard/start_batch")
def start_storyboard_batch(req: StoryboardBatchRequest):
    if active_storyboard_batches.get(req.book_id, {}).get("is_running"):
        return {
            "status": "already_running",
            "message": "A batch generation is already running for this book.",
            **active_storyboard_batches[req.book_id]
        }

    chapters = get_chapters(req.book_id)
    if not chapters:
        raise HTTPException(status_code=400, detail="No chapters found in book")
    
    total = len(chapters)
    end_ch = req.end_chapter or total
    start_ch = max(1, req.start_chapter or 1)

    t = threading.Thread(
        target=run_storyboard_batch_worker,
        args=(req.book_id, start_ch, end_ch, req.model),
        daemon=True
    )
    t.start()
    return {
        "status": "started",
        "book_id": req.book_id,
        "start_chapter": start_ch,
        "end_chapter": end_ch,
        "total_chapters": total
    }

@app.get("/api/storyboard/{book_id}/batch_status")
def get_storyboard_batch_status(book_id: str):
    status = active_storyboard_batches.get(book_id, {
        "is_running": False,
        "current_chapter": 0,
        "total_chapters": 0,
        "completed_chapters": 0,
        "total_cards_generated": 0,
        "cancelled": False
    })
    return status

@app.post("/api/storyboard/{book_id}/cancel_batch")
def cancel_storyboard_batch(book_id: str):
    if book_id in active_storyboard_batches:
        active_storyboard_batches[book_id]["cancelled"] = True
        active_storyboard_batches[book_id]["is_running"] = False
    return {"status": "cancelled"}

@app.delete("/api/storyboard/{book_id}/chapter/{ch_num}")
def delete_chapter_storyboard(book_id: str, ch_num: int):
    db = load_storyboard_db(book_id)
    if str(ch_num) in db.get("chapters", {}):
        del db["chapters"][str(ch_num)]
        save_storyboard_db(book_id, db)
    return {"status": "success", "remaining_chapters": len(db.get("chapters", {}))}

# --- MASTER CHARACTER BIBLE & TWO-WAY STORYBOARD SYNC ---

CANONICAL_CHARACTERS_MASTER = [
    {
        "id": "char_brock_clayton",
        "canonical_en": "Brock Clayton",
        "canonical_ar": "بروك كلايتون",
        "role": "Protagonist / Male Lead",
        "role_ar": "البطل الرئيسي",
        "species": "Human / Solar Dragon Splice",
        "species_ar": "بشري مدمج مع تنين شمسي أسطوري",
        "gender": "Male",
        "gender_ar": "ذكر",
        "physical_en": "Athletic, tall male with sun-bronzed skin and determined amber-hazel eyes. Under stress or power activation, shimmering golden dragon scales emerge along his arms, neck, and spinal ridge, accompanied by large draconic wings and glowing solar energy aura. In full dragon form: A magnificent, massive golden solar dragon with iridescent scales, glowing spinal crests, and formidable draconic wings.",
        "physical_ar": "شاب رياضي طويل القامة، بشرته برونزية مشمسة وعيناه عسليتان تلمعان بالعزم. عند إطلاق قوته أو تحت الضغط، تبرز حراشف ذهبية متلألئة على ذراعيه ورقبته وعموده الفقري، مع أجنحة تنين ضخمة وهالة شمسية متوهجة. وفي هيئة التنين الكامل: تنين شمسي ذهبي مهيب وضخم بحراشف متلألئة وعرف شوكي مشع وأجنحة جبارة.",
        "visual_prompt_anchor": "cinematic portrait, Brock Clayton, athletic male in mid-20s, sharp jawline, short ruffled dark brown hair, glowing amber-gold eyes, sun-kissed skin with faint iridescent golden dragon scale patterns along jawline and neck, confident and heroic expression, 8k resolution, highly detailed, photorealistic cinematic lighting",
        "voice_profile": "Warm, grounded, slightly husky baritone. Honest and relatable with sudden bursts of commanding resonance when channeling his draconic power.",
        "voice_profile_ar": "صوت باريتون دافئ، خشن بنعومة، صريح ومقنع، يتحول إلى رنين هادر وحازم عند إطلاق قوة التنين.",
        "alias_patterns": [
            r"\bbrock\b", r"\bbrock\s+clayton\b", r"\bbrock\s+star\b", 
            r"\bprotagonist\b", r"\bsolar\s+dragon\b", r"\bdragon\s+form\b"
        ]
    },
    {
        "id": "char_pinky_peach",
        "canonical_en": "Pinky Peach",
        "canonical_ar": "بينكي بيتش",
        "role": "Core Female Lead / Streamer",
        "role_ar": "بطلة رئيسية / صانعة محتوى ومشهورة رقمية",
        "species": "Unicorn-Girl (Equine Demi-Human)",
        "species_ar": "فتاة اليونيكورن الأسطورية",
        "gender": "Female",
        "gender_ar": "أنثى",
        "physical_en": "Stunning, ethereal unicorn-girl with pristine milk-white skin, lustrous pastel-pink hair falling in soft waves, and expressive, sparkling pink/violet eyes. A spiral golden horn gracefully crowns her forehead, catching the light. Curvy, delicate feminine physique with an enchanting, bubbly, and charming on-screen presence.",
        "physical_ar": "فتاة يونيكورن فاتنة بجمال ملائكي، بشرتها بيضاء ناصعة كالحليب، شعرها بلون وردي باستيل جذاب ينسدل في موجات ناعمة، وعيناها ورديتان لامعتان. يتوسط جبهتها قرن لولبي ذهبي أنيق يشع بريقاً خافتاً. قوام أنثوي رشيق ومظهر ساحر يجمع بين البراءة والجاذبية.",
        "visual_prompt_anchor": "cinematic character shot, Pinky Peach, gorgeous unicorn girl, flawless porcelain milk-white skin, long wavy pastel pink hair, captivating sparkling violet-pink eyes, a single spiral golden unicorn horn on forehead, cute friendly smile, high fashion modern aesthetic, unreal engine 5 render, cinematic studio lighting",
        "voice_profile": "Melodic, sweet, upbeat soprano with playful intonations, capable of shifting into soft, intimate whispers when sincere.",
        "voice_profile_ar": "صوت سوبرانو نغمي، عذب وحيوي، نبرة مرحة تتخللها همسات رقيقة وحميمة عند الحديث بصدق وعاطفة.",
        "alias_patterns": [
            r"\bpinky\b", r"\bpinky\s+peach\b"
        ]
    },
    {
        "id": "char_mamba",
        "canonical_en": "Mamba",
        "canonical_ar": "مامبا",
        "role": "Core Female Lead / Draconic Warrior",
        "role_ar": "بطلة رئيسية / محاربة تنينية",
        "species": "Draconic / Reptilian Demi-Human",
        "species_ar": "فتاة من سلالة التنانين والزواحف المحاربة",
        "gender": "Female",
        "gender_ar": "أنثى",
        "physical_en": "Fierce, striking draconic demi-human with exotic emerald-tinted olive skin, smooth scales accentuating her cheekbones, collarbones, and hips. Athletic, sculpted curves with predatory grace, feline/reptilian golden-slit eyes, sleek dark hair, and a sinuous draconic tail with dark accents. Exudes confidence, lethal martial prowess, and smoldering sensuality.",
        "physical_ar": "محاربة تنينية ذات حضور طاغٍ، بشرتها زيتونية مائلة للزمردي مع قشور ناعمة تزين عظمتي الخدين والترقوة والخصر. قوام رياضي منحوت ورشاقة مفترسة، عيناها ذهبيتان ببؤبؤ عمودي مثل الزواحف، شعرها داكن أملس، ولها ذيل تنيني مرن بحواف داكنة. تنبض بالثقة والقوة والجاذبية الشرسة.",
        "visual_prompt_anchor": "cinematic action portrait, Mamba, exotic reptilian draconic woman, athletic sculpted hourglass physique, olive-emerald smooth skin with subtle emerald scales along temples and collarbones, fierce slit-pupil golden eyes, sleek long raven hair, sharp confident smirk, dark cyberpunk leather gear, dynamic rim lighting, 8k",
        "voice_profile": "Sultry, confident, low-pitched alto with a raspy purr, fearless and teasing.",
        "voice_profile_ar": "صوت ألتو دافئ ومغري، بنبرة واثقة وهدير خافت ناعم يعكس الشجاعة والمرح والجرأة.",
        "alias_patterns": [
            r"\bmamba\b"
        ]
    },
    {
        "id": "char_blossom",
        "canonical_en": "Blossom",
        "canonical_ar": "بلوسوم",
        "role": "Core Female Lead / Floral Sprite",
        "role_ar": "بطلة رئيسية / جنية الزهور والطبيعة",
        "species": "Flora-Sprite / Miniature Plant Kin",
        "species_ar": "جنية النباتات والزهور الفاتنة",
        "gender": "Female",
        "gender_ar": "أنثى",
        "physical_en": "Adorable, miniature humanoid flora-sprite (approx 12-18 inches tall or small fairy proportions). Tender jade-green skin that blushes a deeper emerald or rosy hue when flustered. Her hair is an intricate crown of living flowering vines and soft rose-petal tresses. Wide, innocent dew-drop eyes and translucent, gossamer leaf-like wings.",
        "physical_ar": "جنية زهور فائقة اللطافة والجمال، حجمها مصغر (في حجم كف اليد أو زهرة حديقة). بشرتها خضراء يشمية ناعمة تحمر بخجل زمردي أو وردي عندما ترتبك. شعرها تاج من أغصان الكروم الحية وبراعم الورد المتفتحة، عيناها واسعتان كريستاليتان كقطرات الندى، ولها أجنحة شافة كأوراق الشجر الرقيقة.",
        "visual_prompt_anchor": "cinematic macro photography, Blossom, tiny fairy plant girl, 15cm height, delicate jade-green skin, sparkling oversized dewdrop eyes, hair made of living vine tendrils and tiny pink rose blossoms, translucent glowing leaf wings, sitting on a table next to a cup, shallow depth of field, fairytale cinematic lighting",
        "voice_profile": "High-pitched, breathy, chime-like soprano, exceedingly cute, bubbly, and prone to excited squeaks.",
        "voice_profile_ar": "صوت سوبرانو رنان وناعم كأجراس الكريستال، فائق اللطافة وسريع النبرة عندما تتحمس، مع شهقات خجولة ظريفة.",
        "alias_patterns": [
            r"\bblossom\b", r"\bplant\s+girl\b"
        ]
    },
    {
        "id": "char_dr_luna",
        "canonical_en": "Dr. Artigius Luna",
        "canonical_ar": "د. أرتيجيوس لونيا",
        "role": "Key Supporting / Physician & Scientist",
        "role_ar": "شخصية مساعدة رئيسية / طبيب وخبير جيني",
        "species": "Alien Xeno-Physician",
        "species_ar": "كائن فضائي جيني متخصص",
        "gender": "Male",
        "gender_ar": "ذكر",
        "physical_en": "Wiry, slender xeno-humanoid figure with a hairless cranium, pointed elongated ears, pale leathery skin, and sharp, calculating but sympathetic eyes. Dressed in clinical, sterile academy physician attire, often carrying a medical scanner or briefcase.",
        "physical_ar": "طبيب فضائي نحيل وممشوق، برأس أصلس وأذنين مدببتين طويلتين، بشرته جلدية شاحبة وعيناه ثاقبتان تجمعان بين الذكاء العلمي والتعاطف. يرتدي بدلة أطباء أكاديمية معقمة ويحمل ماسحه الجيني وحقيبته الطبية.",
        "visual_prompt_anchor": "cinematic character portrait, Dr. Artigius Luna, wiry alien physician, pointed ears, bald head, wise observant eyes, wearing high-tech white and slate-gray medical coat with glowing holographic badges, clean futuristic infirmary background, photorealistic",
        "voice_profile": "Fast, precise, intellectual cadence with a calm reassuring bedside manner.",
        "voice_profile_ar": "نبرة سريعة ودقيقة ومثقفة، هادئة ومطمئنة بأسلوب الأطباء المحترفين.",
        "alias_patterns": [
            r"\bdr\.?\s*luna\b", r"\bartigius\s+luna\b", r"\bluna\b"
        ]
    },
    {
        "id": "char_commissioner_gourden",
        "canonical_en": "Commissioner Go'Urden",
        "canonical_ar": "الكومشنر جو-أوردن",
        "role": "Key Supporting / Law Enforcement Chief",
        "role_ar": "شخصية مساعدة رئيسية / قائد شرطة ومحقق عسكري",
        "species": "Elf-Blood Detective / Lawman",
        "species_ar": "إلف عسكري مخضرم",
        "gender": "Male",
        "gender_ar": "ذكر",
        "physical_en": "Broad-shouldered, rugged, weathered lawman with an elven heritage, prominent muscular build, and a thick, stylized mustache twirled into an iconic Q-shape. Wears a grime-brown heavy trench coat, reflective sunglasses, and usually has a half-smoked brown paper cigar between his teeth.",
        "physical_ar": "رجل قانون مهيب وعريض المنكبين، من أصول إلفية مخضرمة، ذو بنية عضلية قوية وشارب كثيف ملفوف بأسلوب مميز على شكل حرف Q. يرتدي معطفاً طويلاً بنياً داكناً ونظارات شمسية عاكسة، ويمضغ سيجاراً بنياً في زاوية فمه.",
        "visual_prompt_anchor": "cinematic noir portrait, Commissioner Go'Urden, gritty detective, broad elven face with sharp jaw, large twirled Q-shaped mustache, unlit cigar in mouth, weathered grime-brown trench coat, dark rain-slicked sci-fi city background, moody noir neon lighting",
        "voice_profile": "Gravelly, deep, world-weary bass with a gruff no-nonsense authority.",
        "voice_profile_ar": "صوت باس عميق وخشن، محنك وواقعي يعكس هيبة السلطة والخبرة الطويلة في شوارع المدينة.",
        "alias_patterns": [
            r"\bcommissioner\b", r"\bgo'?urden\b", r"\bgourden\b"
        ]
    },
    {
        "id": "char_hephi",
        "canonical_en": "Hephimila (Hephi)",
        "canonical_ar": "هيفيميلا (هيفي)",
        "role": "Supporting / Academic Researcher",
        "role_ar": "شخصية مساعدة / باحثة أكاديمية",
        "species": "Demihuman Scholar",
        "species_ar": "باحثة من السلالات الراقية",
        "gender": "Female",
        "gender_ar": "أنثى",
        "physical_en": "Intellectual young woman with neat posture, silver-framed spectacles resting on her nose, neat coiffed hair, dressed in crisp academy-style uniform, frequently taking notes in a digital pad.",
        "physical_ar": "شابة ذكية ذات مظهر أكاديمي مرتب، ترتدي نظارات فضية أنيقة، شعرها منسق بعناية، وترتدي زي الأكاديمية الرسمي، وتدوّن الملاحظات دائماً على جهازها الرقمي.",
        "visual_prompt_anchor": "cinematic portrait, Hephimila Hephi, scholarly attractive woman, wearing silver-framed glasses, neat scholar academy blazer, taking notes, intelligent and focused gaze, university library background",
        "voice_profile": "Polite, articulate, crisp collegiate tone.",
        "voice_profile_ar": "نبرة مهذبة، واضحة ومخارج حروف متقنة كطالبات الأكاديميات العريقة.",
        "alias_patterns": [
            r"\bhephi\b", r"\bhephimila\b"
        ]
    },
    {
        "id": "char_koriana",
        "canonical_en": "Koriana",
        "canonical_ar": "كوريانا",
        "role": "Supporting / Fit Warrior",
        "role_ar": "شخصية مساعدة / مقاتلة رياضية",
        "species": "Dark-skinned Xeno Demi-Human",
        "species_ar": "محاربة فضائية ذات بشرة بنفسجية",
        "gender": "Female",
        "gender_ar": "أنثى",
        "physical_en": "Athletic, exceptionally fit warrior woman with deep purple/amethyst skin, stark contrasting silver-white hair, and piercing fearless gaze, dressed in form-fitting combat gym wear.",
        "physical_ar": "محاربة رياضية فائقة الرشاقة والقوة، بشرتها بنفسجية داكنة ساحرة مع شعر فضي أبيض ناصع، عيناها جريئتان لا تعرفان الخوف، وترتدي زياً قتالياً رياضياً مجسماً.",
        "visual_prompt_anchor": "cinematic action photo, Koriana, fit athletic woman with deep rich purple skin, contrasting snow-white hair, intense fearless eyes, athletic combat outfit, dramatic neon side lighting",
        "voice_profile": "Confident, athletic, energetic.",
        "voice_profile_ar": "نبرة واثقة، رياضية مفعمة بالحيوية والقوة.",
        "alias_patterns": [
            r"\bkoriana\b"
        ]
    },
    {
        "id": "char_satina",
        "canonical_en": "Satina",
        "canonical_ar": "ساتينا",
        "role": "Supporting / Cafe Barista",
        "role_ar": "شخصية مساعدة / نادلة المقهى",
        "species": "Darkling Demihuman",
        "species_ar": "فتاة من سلالة الظلال اللطيفة",
        "gender": "Female",
        "gender_ar": "أنثى",
        "physical_en": "Attractive young woman with vibrant auburn/red hair, cute freckles across her nose and cheeks, wearing a neat cafe barista apron and uniform.",
        "physical_ar": "شابة جذابة ذات شعر كستنائي أحمر متألق، ونمش لطيف يزين أنفها ووجنتيها، ترتدي مريلة عمل أنيقة للمقهى وابتسامة ودودة.",
        "visual_prompt_anchor": "cinematic portrait, Satina, cute barista girl, fiery red hair in messy bun, freckled cheeks, warm charming smile, wearing coffee shop apron, cozy warm coffee shop ambiance",
        "voice_profile": "Cheery, hospitable, warm customer-service cadence.",
        "voice_profile_ar": "صوت مبتهج ودود ومرحب ينبض بالدفء وحسن الضيافة.",
        "alias_patterns": [
            r"\bsatina\b"
        ]
    },
    {
        "id": "char_xeno_lich",
        "canonical_en": "The Xeno-Lich",
        "canonical_ar": "الزينو-ليتش (الساحر الفضائي الشرير)",
        "role": "Antagonist",
        "role_ar": "العدو والخصم الشرير",
        "species": "Cosmic Undead Necromancer",
        "species_ar": "مشعوذ كوني مظلم",
        "gender": "Male / Entity",
        "gender_ar": "كيان مظلم",
        "physical_en": "Hunchbacked, sinister figure draped in a tattered void-black hooded cowl, wielding shadowy necrotic magic with claw-like skeletal hands and glowing spectral eyes.",
        "physical_ar": "كيان مظلم أحدب يرتدي رداءً أسود مهترئاً مع قلنسوة تخفي ملامحه، تنبعث من يديه العظميتين قوى سحرية مظلمة، وعيناه تشعان بنور طيفي شرير.",
        "visual_prompt_anchor": "cinematic dark fantasy, The Xeno-Lich, sinister necromancer in tattered dark void robes, skeletal clawed hands crackling with dark purple energy, glowing eerie green eyes beneath deep hood, horror atmosphere",
        "voice_profile": "Hissing, dry, echoing dark whisper.",
        "voice_profile_ar": "صوت فحيح جاف ومرعب، يتردد بصدى مظلم يثير الرهبة.",
        "alias_patterns": [
            r"\bxeno-?lich\b", r"\blich\b"
        ]
    }
]

def rebuild_character_bible_for_book(book_id: str) -> Dict[str, Any]:
    sb = load_storyboard_db(book_id)
    chapters = sb.get("chapters", {})
    if not chapters:
        return {"status": "no_chapters", "characters_count": 0, "cards_count": 0}

    def clean_name_str(s):
        if not s: return ""
        return s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').strip()

    is_solar = "solar" in book_id.lower()

    def find_match(raw):
        if not is_solar:
            return None
        norm = clean_name_str(raw).lower()
        for c in CANONICAL_CHARACTERS_MASTER:
            for pat in c["alias_patterns"]:
                if re.search(pat, norm, re.IGNORECASE):
                    return c
        return None

    dossiers = {}
    if is_solar:
        for c in CANONICAL_CHARACTERS_MASTER:
            cid = c["id"]
            dossiers[cid] = {
                "id": cid,
                "name": c["canonical_en"],
                "name_ar": c["canonical_ar"],
                "role": c["role"],
                "role_ar": c["role_ar"],
                "species": c["species"],
                "species_ar": c["species_ar"],
                "gender": c["gender"],
                "gender_ar": c["gender_ar"],
                "physical_en": c["physical_en"],
                "physical_ar": c["physical_ar"],
                "visual_prompt_anchor": c["visual_prompt_anchor"],
                "voice_profile": c["voice_profile"],
                "voice_profile_ar": c["voice_profile_ar"],
                "is_locked": True,
                "aliases": set(),
                "total_cards": 0,
                "chapters_present": set(),
                "card_timeline": [],
                "distinct_wardrobes": []
            }

    minors = {}
    total_cards = 0

    for ch_key, ch_data in chapters.items():
        ch_num = int(ch_data.get("chapter_num", ch_key))
        ch_title = ch_data.get("chapter_title", f"Chapter {ch_num}")
        ch_title_ar = ch_data.get("chapter_title_ar", "")
        cards = ch_data.get("cards", [])

        for card in cards:
            total_cards += 1
            card_num = card.get("card_number")
            slugline = card.get("slugline", "")
            slug_title = card.get("title_en", "")

            dialogue_items = card.get("audio_and_dialogue", {}).get("spoken_dialogue", [])
            vis_act = card.get("visual_action", {})
            act_en = vis_act.get("action_en", "") if isinstance(vis_act, dict) else str(vis_act)
            act_ar = vis_act.get("action_ar", "") if isinstance(vis_act, dict) else ""

            cast_list = card.get("cast_present", [])
            updated_cast = []

            for cast in cast_list:
                raw_name = clean_name_str(cast.get("name_en", "") or cast.get("original_name", ""))
                if not raw_name:
                    continue

                raw_outfit = clean_name_str(cast.get("outfit_and_condition", ""))

                lines_spoken = []
                for d in dialogue_items:
                    spk = clean_name_str(d.get("speaker", ""))
                    if raw_name.lower() in spk.lower() or spk.lower() in raw_name.lower():
                        lines_spoken.append({
                            "line_en": d.get("line_en", ""),
                            "line_ar": d.get("line_ar", "")
                        })

                matched = find_match(raw_name)
                if matched:
                    cid = matched["id"]
                    target = dossiers[cid]
                    target["aliases"].add(raw_name)
                    target["total_cards"] += 1
                    target["chapters_present"].add(ch_num)

                    card_rec = {
                        "chapter_num": ch_num,
                        "chapter_title": ch_title,
                        "chapter_title_ar": ch_title_ar,
                        "card_number": card_num,
                        "slugline": slugline,
                        "scene_title": slug_title,
                        "raw_name_in_card": raw_name,
                        "outfit_and_condition": raw_outfit,
                        "action_summary_en": act_en,
                        "action_summary_ar": act_ar,
                        "dialogues": lines_spoken
                    }
                    target["card_timeline"].append(card_rec)

                    if raw_outfit and not any(o["description"] == raw_outfit for o in target["distinct_wardrobes"]):
                        target["distinct_wardrobes"].append({
                            "id": f"outfit_{len(target['distinct_wardrobes']) + 1}",
                            "chapter_num": ch_num,
                            "card_number": card_num,
                            "description": raw_outfit
                        })

                    updated_cast.append({
                        "char_id": cid,
                        "name_en": matched["canonical_en"],
                        "name_ar": matched["canonical_ar"],
                        "original_name": raw_name,
                        "outfit_and_condition": raw_outfit,
                        "is_canonical": True
                    })
                else:
                    norm_m = raw_name.lower()
                    if norm_m not in minors:
                        mid = f"minor_{re.sub(r'[^a-z0-9_]', '', norm_m.replace(' ', '_'))}"
                        minors[norm_m] = {
                            "id": mid,
                            "name": raw_name,
                            "name_ar": cast.get("name_ar", raw_name),
                            "role": "Minor / Cameo",
                            "role_ar": "شخصية ثانوية / عابرة",
                            "species": "Demihuman / Civilian",
                            "species_ar": "مواطن / شخصية عابرة",
                            "gender": "Unknown",
                            "gender_ar": "غير محدد",
                            "physical_en": raw_outfit or "Incidental background character.",
                            "physical_ar": "شخصية خلفية عابرة.",
                            "visual_prompt_anchor": f"cinematic shot, {raw_name}, secondary background character, highly detailed",
                            "voice_profile": "Incidental bystander / minor dialogue.",
                            "voice_profile_ar": "صوت عابر لشخصية ثانوية.",
                            "is_locked": False,
                            "aliases": [raw_name],
                            "total_cards": 0,
                            "chapters_present": set(),
                            "card_timeline": [],
                            "distinct_wardrobes": []
                        }

                    mt = minors[norm_m]
                    mt["total_cards"] += 1
                    mt["chapters_present"].add(ch_num)
                    mt["card_timeline"].append({
                        "chapter_num": ch_num,
                        "chapter_title": ch_title,
                        "card_number": card_num,
                        "slugline": slugline,
                        "scene_title": slug_title,
                        "raw_name_in_card": raw_name,
                        "outfit_and_condition": raw_outfit,
                        "action_summary_en": act_en,
                        "dialogues": lines_spoken
                    })

                    updated_cast.append({
                        "char_id": mt["id"],
                        "name_en": raw_name,
                        "name_ar": cast.get("name_ar", raw_name),
                        "original_name": raw_name,
                        "outfit_and_condition": raw_outfit,
                        "is_canonical": False
                    })

            card["cast_present"] = updated_cast

    final_list = []
    for cid, d in dossiers.items():
        d["aliases"] = sorted(list(d["aliases"]))
        d["chapters_present"] = sorted(list(d["chapters_present"]))
        d["chapters_count"] = len(d["chapters_present"])
        final_list.append(d)

    for norm_m, d in minors.items():
        d["chapters_present"] = sorted(list(d["chapters_present"]))
        d["chapters_count"] = len(d["chapters_present"])
        final_list.append(d)

    final_list.sort(key=lambda x: (0 if "char_" in x["id"] else 1, -x["total_cards"]))

    bible = load_db(book_id)
    bible["book_id"] = book_id
    bible["characters"] = final_list
    save_db(bible, book_id)
    save_storyboard_db(book_id, sb)

    return {
        "status": "success",
        "total_cards_processed": total_cards,
        "characters_count": len(final_list),
        "primary_characters_count": len(dossiers),
        "minor_characters_count": len(minors)
    }

class RebuildBibleRequest(BaseModel):
    book_id: str

@app.post("/api/characters/rebuild_bible")
def rebuild_bible_endpoint(req: RebuildBibleRequest):
    return rebuild_character_bible_for_book(req.book_id)

class ToggleLockRequest(BaseModel):
    book_id: str
    character_id: str

@app.post("/api/characters/toggle_lock")
def toggle_character_lock_endpoint(req: ToggleLockRequest):
    db = load_db(req.book_id)
    chars = db.get("characters", [])
    found = None
    for c in chars:
        if c.get("id") == req.character_id:
            c["is_locked"] = not c.get("is_locked", False)
            found = c
            break
    if found:
        save_db(db, req.book_id)
        return {"status": "success", "is_locked": found["is_locked"]}
    raise HTTPException(status_code=404, detail="Character not found")

# --- KREA 2 TURBO 1:1 FACE PORTRAIT GENERATION (COMFYUI & GEMMA 4) ---

KREA2_WORKFLOW_PATH = r"C:\Users\MiroAI\Downloads\image_krea2_turbo_t2i_int8_Character_portrait.json"
COMFY_URL = "http://127.0.0.1:8188"
COMFY_PORTRAITS_DIR = r"C:\ComfyUI_windows_portable\ComfyUI\output\character_portraits"
os.makedirs(COMFY_PORTRAITS_DIR, exist_ok=True)

def generate_krea_prompt_gemma4(name: str, species: str, desc: str) -> str:
    sys_prompt = (
        "You are an elite prompt engineer specialized in Krea 2 Turbo (12B+ Vision-Language architecture) for high-speed concept character portraits.\n\n"
        "YOUR TASK:\n"
        "Generate a specialized, ultra-precise Krea 2 Turbo face-only prompt for a character portrait on a pure solid white background.\n\n"
        "CRITICAL CANONICAL RULES:\n"
        "1. BASELINE DEFAULT APPEARANCE ONLY (NO STATE-AVERAGING):\n"
        "   - Depict the character in their DEFAULT, everyday canonical form.\n"
        "   - NEVER average or blend temporary power-up forms, dragon transformations, battle stances, or combat shifts into the master baseline portrait!\n"
        "   - If a character is human with magical, shifting, or combat powers, depict them in their baseline human state: natural human skin, authentic facial features, NO combat transformations or lizard/creature parts on their face.\n"
        "2. VIBRANT, HEALTHY, CHARMING AESTHETICS (NO VAMPIRES / ZOMBIES):\n"
        "   - For fair or pale characters, NEVER use chalk-white, milk-white, or dead pallor that makes them look like a corpse or vampire.\n"
        "   - Describe them as extremely cute, youthful, and gorgeous: 'flawless creamy porcelain skin with soft warm peach undertones and healthy rosy blushed cheeks, soft natural pink lips, sweet charming smile, large expressive sparkling eyes'.\n"
        "3. NEVER use banned buzzwords: DO NOT use 'masterpiece', '8k', 'photorealistic', 'ultra detailed', 'hyper realistic', 'unreal engine', or 'trending on artstation'.\n"
        "4. Front-Load the Subject: Put camera framing and facial anatomy/eyes FIRST (highest visual weight).\n"
        "5. Frame Type: Tight close-up face portrait (1:1 square crop), centered, front-facing eye-level gaze directly looking into lens.\n"
        "6. Backdrop: Must explicitly specify 'isolated on a seamless flat solid pure white studio backdrop, #FFFFFF void, absolutely no shadows on background, no environmental scenery'.\n"
        "7. Lighting: 'High-key clean studio portrait lighting, soft neutral diffuse softbox illumination, zero background shadows'.\n"
        "8. Medium & Optics: '85mm prime portrait lens, sharp focus on facial features, natural skin pores and subsurface scattering, clean photographic clarity'.\n"
        "9. Format: Output ONLY the prompt paragraph. No conversational preamble, no markdown titles, no quotes around the prompt."
    )
    user_prompt = f"Character: {name}\nSpecies: {species}\nPhysical Description from novel: {desc}\n\nWrite the Krea 2 Turbo face-only prompt on a pure white background:"
    
    payload = {
        "model": "gemma4-26b-a4b-qat-uncensored-hauhaucs-balanced-mtp@q4_k_m",
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.4,
        "max_tokens": 400
    }
    try:
        r = requests.post(f"{LM_STUDIO_URL}/chat/completions", json=payload, timeout=25)
        if r.status_code == 200:
            prompt_text = r.json()["choices"][0]["message"]["content"].strip()
            if prompt_text.startswith('"') and prompt_text.endswith('"'):
                prompt_text = prompt_text[1:-1]
            return prompt_text
    except Exception as e:
        print(f"Error in Gemma 4 Krea prompt generation: {e}")
    
    # Fallback to deterministic Krea 2 formula if LLM is unavailable
    return (
        f"Tight close-up face portrait of {name}, centered, front-facing eye-level gaze directly looking into lens, "
        f"{desc}, 85mm prime portrait lens, sharp focus on facial features, natural skin pores and subsurface scattering, "
        "clean photographic clarity, high-key clean studio portrait lighting, soft neutral diffuse softbox illumination, "
        "zero background shadows, isolated on a seamless flat solid pure white studio backdrop, #FFFFFF void, absolutely no shadows on background, no environmental scenery."
    )

class KreaPromptRequest(BaseModel):
    book_id: str
    character_id: str

@app.post("/api/characters/generate_krea_prompt")
def generate_character_krea_prompt_endpoint(req: KreaPromptRequest):
    db = load_db(req.book_id)
    char = next((c for c in db.get("characters", []) if c.get("id") == req.character_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    
    prompt = generate_krea_prompt_gemma4(
        char.get("name", "Character"),
        char.get("species", "Demihuman"),
        char.get("physical_en", "")
    )
    char["krea2_face_prompt"] = prompt
    save_db(db, req.book_id)
    return {"status": "success", "prompt": prompt}

class GenerateFaceRequest(BaseModel):
    book_id: str
    character_id: str
    custom_prompt: Optional[str] = None

@app.post("/api/characters/generate_face")
def generate_character_face_endpoint(req: GenerateFaceRequest):
    if not os.path.exists(KREA2_WORKFLOW_PATH):
        raise HTTPException(status_code=500, detail=f"Workflow not found at {KREA2_WORKFLOW_PATH}")
    
    db = load_db(req.book_id)
    char = next((c for c in db.get("characters", []) if c.get("id") == req.character_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")
    
    # Determine prompt
    prompt = req.custom_prompt or char.get("krea2_face_prompt")
    if not prompt:
        prompt = generate_krea_prompt_gemma4(
            char.get("name", "Character"),
            char.get("species", "Demihuman"),
            char.get("physical_en", "")
        )
        char["krea2_face_prompt"] = prompt

    # Safe filename prefix
    safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', char.get("name", "character"))
    prefix = f"character_portraits/{safe_name}_face"

    with open(KREA2_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    # Strictly modify ONLY Node 53 and Node 29 per user requirements!
    wf["53"]["inputs"]["text"] = prompt
    wf["29"]["inputs"]["filename_prefix"] = prefix

    # Submit to ComfyUI
    try:
        res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf, "client_id": f"studio_{safe_name}"}, timeout=8)
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ComfyUI rejected prompt: {res.text}")
        prompt_id = res.json().get("prompt_id")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI offline or error: {str(e)}")

    # Poll ComfyUI for completion
    import time
    start_time = time.time()
    image_url = None
    filename_found = None

    while time.time() - start_time < 90:
        time.sleep(2)
        try:
            h_res = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
            if h_res.status_code == 200:
                h_data = h_res.json()
                if prompt_id in h_data:
                    item = h_data[prompt_id]
                    imgs = item.get("outputs", {}).get("29", {}).get("images", [])
                    if imgs:
                        filename_found = imgs[0].get("filename")
                        image_url = f"/portraits/{filename_found}"
                        break
        except Exception:
            pass

    if not image_url:
        raise HTTPException(status_code=504, detail="ComfyUI generation timed out after 90 seconds")

    # Update database
    char["portrait_image_url"] = image_url
    char["krea2_face_prompt"] = prompt
    save_db(db, req.book_id)

    return {
        "status": "success",
        "image_url": image_url,
        "filename": filename_found,
        "prompt": prompt
    }

# --- BATCH CHARACTER FACE GENERATION ENGINE (COMFYUI & GEMMA 4) ---

import copy
import time

active_face_batches: Dict[str, Dict[str, Any]] = {}

class FaceBatchRequest(BaseModel):
    book_id: str
    skip_existing: Optional[bool] = True

def run_face_batch_worker(book_id: str, skip_existing: bool = True):
    if not os.path.exists(KREA2_WORKFLOW_PATH):
        active_face_batches[book_id] = {
            "is_running": False,
            "error": f"Workflow not found at {KREA2_WORKFLOW_PATH}"
        }
        return

    db = load_db(book_id)
    characters = db.get("characters", [])
    if not characters:
        active_face_batches[book_id] = {"is_running": False, "error": "No characters found"}
        return

    # Filter targets
    targets = []
    for c in characters:
        if skip_existing and c.get("portrait_image_url"):
            continue
        targets.append(c)

    total_targets = len(targets)
    active_face_batches[book_id] = {
        "is_running": True,
        "total": total_targets,
        "completed": 0,
        "current_name": targets[0].get("name", "") if targets else "",
        "current_index": 0,
        "last_image_url": None,
        "cancelled": False,
        "error": None
    }

    if total_targets == 0:
        active_face_batches[book_id]["is_running"] = False
        return

    try:
        with open(KREA2_WORKFLOW_PATH, "r", encoding="utf-8") as f:
            base_wf = json.load(f)

        for idx, char in enumerate(targets):
            if active_face_batches.get(book_id, {}).get("cancelled"):
                print(f"[Face Batch] Cancelled by user at {idx}/{total_targets}")
                break

            char_name = char.get("name", f"Character_{idx+1}")
            active_face_batches[book_id]["current_name"] = char_name
            active_face_batches[book_id]["current_index"] = idx + 1

            # 1. Prompt formulation via Gemma 4 if missing
            prompt = char.get("krea2_face_prompt")
            if not prompt:
                print(f"[Face Batch] Formulating prompt for {char_name} via Gemma 4...")
                prompt = generate_krea_prompt_gemma4(
                    char_name,
                    char.get("species", "Demihuman"),
                    char.get("physical_en", "") or char.get("role", "")
                )
                char["krea2_face_prompt"] = prompt

            # 2. Workflow modification (ONLY Node 53 and Node 29 per user requirements!)
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', char_name)
            prefix = f"character_portraits/{safe_name}_face"

            wf = copy.deepcopy(base_wf)
            wf["53"]["inputs"]["text"] = prompt
            wf["29"]["inputs"]["filename_prefix"] = prefix

            # 3. Submit to ComfyUI
            print(f"[Face Batch {idx+1}/{total_targets}] Generating {char_name} in ComfyUI...")
            prompt_id = None
            try:
                res = requests.post(
                    f"{COMFY_URL}/prompt",
                    json={"prompt": wf, "client_id": f"batch_{safe_name}"},
                    timeout=10
                )
                if res.status_code == 200:
                    prompt_id = res.json().get("prompt_id")
                else:
                    print(f"[Face Batch Comfy Error] {res.text}")
                    continue
            except Exception as e:
                print(f"[Face Batch Submit Exception] {e}")
                continue

            if not prompt_id:
                continue

            # 4. Poll ComfyUI for completion
            image_url = None
            start_t = time.time()
            while time.time() - start_t < 120:
                if active_face_batches.get(book_id, {}).get("cancelled"):
                    break
                time.sleep(2)
                try:
                    h_res = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
                    if h_res.status_code == 200:
                        h_data = h_res.json()
                        if prompt_id in h_data:
                            imgs = h_data[prompt_id].get("outputs", {}).get("29", {}).get("images", [])
                            if imgs:
                                fn = imgs[0].get("filename")
                                image_url = f"/portraits/{fn}"
                                break
                except Exception:
                    pass

            if image_url:
                char["portrait_image_url"] = image_url
                active_face_batches[book_id]["completed"] += 1
                active_face_batches[book_id]["last_image_url"] = image_url
                print(f"[Face Batch] Finished {char_name} -> {image_url}")

                # Save database immediately after each character
                save_db(db, book_id)

    except Exception as e:
        print(f"[Face Batch Exception] {e}")
        active_face_batches[book_id]["error"] = str(e)
    finally:
        if book_id in active_face_batches:
            active_face_batches[book_id]["is_running"] = False

@app.post("/api/characters/generate_all_faces")
def start_face_batch_endpoint(req: FaceBatchRequest):
    if active_face_batches.get(req.book_id, {}).get("is_running"):
        return {
            "status": "already_running",
            **active_face_batches[req.book_id]
        }

    t = threading.Thread(
        target=run_face_batch_worker,
        args=(req.book_id, req.skip_existing),
        daemon=True
    )
    t.start()

    return {
        "status": "started",
        "book_id": req.book_id,
        "skip_existing": req.skip_existing
    }

@app.get("/api/characters/{book_id}/face_batch_status")
def get_face_batch_status_endpoint(book_id: str):
    status = active_face_batches.get(book_id, {
        "is_running": False,
        "total": 0,
        "completed": 0,
        "current_name": "",
        "current_index": 0,
        "last_image_url": None,
        "cancelled": False,
        "error": None
    })
    return status

# --- KREA 2 IDENTITY-EDIT OUTFIT GENERATION ENGINE (COMFYUI) ---

KREA2_OUTFIT_WORKFLOW_PATH = r"C:\Users\MiroAI\Downloads\krea2_identity_edit_Outfit.json"
COMFY_OUTFITS_DIR = r"C:\ComfyUI_windows_portable\ComfyUI\output\character_outfits"
COMFY_INPUT_DIR = r"C:\ComfyUI_windows_portable\ComfyUI\input"
os.makedirs(COMFY_OUTFITS_DIR, exist_ok=True)
os.makedirs(COMFY_INPUT_DIR, exist_ok=True)

class GenerateOutfitRequest(BaseModel):
    book_id: str
    character_id: str
    outfit_id: str

@app.post("/api/characters/generate_outfit")
def generate_character_outfit_endpoint(req: GenerateOutfitRequest):
    if not os.path.exists(KREA2_OUTFIT_WORKFLOW_PATH):
        raise HTTPException(status_code=404, detail="Outfit workflow file not found")

    db = load_db(req.book_id)
    char = next((c for c in db.get("characters", []) if c.get("id") == req.character_id), None)
    if not char:
        raise HTTPException(status_code=404, detail="Character not found")

    wardrobes = char.get("distinct_wardrobes", [])
    outfit = next((w for w in wardrobes if isinstance(w, dict) and w.get("id") == req.outfit_id), None)
    if not outfit:
        raise HTTPException(status_code=404, detail="Outfit not found")

    # Character portrait is required for identity editing
    portrait_url = char.get("portrait_image_url")
    if not portrait_url:
        raise HTTPException(status_code=400, detail="Character must have a generated face portrait first!")

    portrait_filename = os.path.basename(portrait_url)
    src_portrait = os.path.join(COMFY_PORTRAITS_DIR, portrait_filename)
    dst_portrait = os.path.join(COMFY_INPUT_DIR, portrait_filename)
    if os.path.exists(src_portrait) and not os.path.exists(dst_portrait):
        shutil.copy2(src_portrait, dst_portrait)

    outfit_prompt = outfit.get("prompt")
    if not outfit_prompt:
        pronoun = "his" if char.get("gender") == "Male" else "her"
        desc = outfit.get("description", "canonical outfit")
        dist_features = char.get("distinguishing_features") or char.get("physical_en") or "distinct facial features"
        outfit_prompt = (
            f"Change {pronoun} outfit to {desc}. "
            f"Full-body photograph from head to toe, standing upright on a seamless white studio floor. "
            f"Seamless flat solid pure white studio backdrop, #FFFFFF void, zero shadows on background. "
            f"Preserve the exact facial identity - distinguishing features: {dist_features}."
        )
        outfit["prompt"] = outfit_prompt

    with open(KREA2_OUTFIT_WORKFLOW_PATH, "r", encoding="utf-8") as f:
        wf = json.load(f)

    safe_char = re.sub(r'[^a-zA-Z0-9_]', '_', char.get("name", "character"))
    safe_outfit = re.sub(r'[^a-zA-Z0-9_]', '_', outfit.get("id", "outfit"))
    prefix = f"character_outfits/{safe_char}_{safe_outfit}"

    wf["72"]["inputs"]["image"] = portrait_filename
    wf["111"]["inputs"]["text"] = outfit_prompt
    if "84" in wf and "inputs" in wf["84"]:
        wf["84"]["inputs"]["prompt"] = ["111", 0]
    
    neg_prompt = char.get("negative_prompt", "")
    if not neg_prompt and not char.get("is_human"):
        neg_prompt = "human feet, human toes, bare human feet, barefoot with toes, toes, shoes, sneakers"
    if "85" in wf and "inputs" in wf["85"]:
        wf["85"]["inputs"]["prompt"] = neg_prompt

    if "79" in wf and "inputs" in wf["79"]:
        wf["79"]["inputs"]["ref_boost"] = 1.3
    wf["29"]["inputs"]["filename_prefix"] = prefix

    try:
        res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf, "client_id": f"studio_outfit_{safe_char}"}, timeout=8)
        if res.status_code != 200:
            raise HTTPException(status_code=502, detail=f"ComfyUI rejected prompt: {res.text}")
        prompt_id = res.json().get("prompt_id")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"ComfyUI offline or error: {str(e)}")

    start_time = time.time()
    image_url = None
    filename_found = None

    while time.time() - start_time < 90:
        time.sleep(2)
        try:
            h_res = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
            if h_res.status_code == 200:
                h_data = h_res.json()
                if prompt_id in h_data:
                    item = h_data[prompt_id]
                    imgs = item.get("outputs", {}).get("29", {}).get("images", [])
                    if imgs:
                        filename_found = imgs[0].get("filename")
                        image_url = f"/outfits/{filename_found}"
                        break
        except Exception:
            pass

    if not image_url:
        raise HTTPException(status_code=504, detail="ComfyUI outfit generation timed out after 90 seconds")

    outfit["image_url"] = image_url
    save_db(db, req.book_id)

    return {
        "status": "success",
        "image_url": image_url,
        "filename": filename_found,
        "outfit_id": req.outfit_id,
        "prompt": outfit_prompt
    }

active_outfit_batches: Dict[str, Dict[str, Any]] = {}

class OutfitBatchRequest(BaseModel):
    book_id: str
    skip_existing: Optional[bool] = True

def run_outfit_batch_worker(book_id: str, skip_existing: bool = True):
    if not os.path.exists(KREA2_OUTFIT_WORKFLOW_PATH):
        active_outfit_batches[book_id] = {"is_running": False, "error": "Outfit workflow not found"}
        return

    db = load_db(book_id)
    characters = db.get("characters", [])
    
    targets = []
    for c in characters:
        p_url = c.get("portrait_image_url")
        if not p_url:
            continue
        p_filename = os.path.basename(p_url)
        src_portrait = os.path.join(COMFY_PORTRAITS_DIR, p_filename)
        dst_portrait = os.path.join(COMFY_INPUT_DIR, p_filename)
        if os.path.exists(src_portrait) and not os.path.exists(dst_portrait):
            shutil.copy2(src_portrait, dst_portrait)

        for w in c.get("distinct_wardrobes", []):
            if isinstance(w, dict):
                if skip_existing and w.get("image_url"):
                    continue
                targets.append((c, w))

    total = len(targets)
    active_outfit_batches[book_id] = {
        "is_running": True,
        "total": total,
        "completed": 0,
        "current_name": "",
        "current_index": 0,
        "last_image_url": None,
        "cancelled": False,
        "error": None
    }

    if total == 0:
        active_outfit_batches[book_id]["is_running"] = False
        return

    try:
        with open(KREA2_OUTFIT_WORKFLOW_PATH, "r", encoding="utf-8") as f:
            base_wf = json.load(f)

        for idx, (char, outfit) in enumerate(targets):
            if active_outfit_batches.get(book_id, {}).get("cancelled"):
                break

            char_name = char.get("name", "Character")
            outfit_name = outfit.get("name", "Outfit")
            active_outfit_batches[book_id]["current_name"] = f"{char_name} - {outfit_name}"
            active_outfit_batches[book_id]["current_index"] = idx + 1

            p_filename = os.path.basename(char.get("portrait_image_url", ""))
            outfit_prompt = outfit.get("prompt")
            if not outfit_prompt:
                pronoun = "his" if char.get("gender") == "Male" else "her"
                desc = outfit.get("description", "canonical outfit")
                dist_features = char.get("distinguishing_features") or char.get("physical_en") or "distinct facial features"
                outfit_prompt = (
                    f"Change {pronoun} outfit to {desc}. "
                    f"Full-body photograph from head to toe, standing upright on a seamless white studio floor. "
                    f"Seamless flat solid pure white studio backdrop, #FFFFFF void, zero shadows on background. "
                    f"Preserve the exact facial identity - distinguishing features: {dist_features}."
                )
                outfit["prompt"] = outfit_prompt

            wf = copy.deepcopy(base_wf)
            safe_char = re.sub(r'[^a-zA-Z0-9_]', '_', char_name)
            safe_outfit = re.sub(r'[^a-zA-Z0-9_]', '_', outfit.get("id", "outfit"))
            prefix = f"character_outfits/{safe_char}_{safe_outfit}"

            wf["72"]["inputs"]["image"] = p_filename
            wf["111"]["inputs"]["text"] = outfit_prompt
            if "84" in wf and "inputs" in wf["84"]:
                wf["84"]["inputs"]["prompt"] = ["111", 0]
            
            neg_prompt = char.get("negative_prompt", "")
            if not neg_prompt and not char.get("is_human"):
                neg_prompt = "human feet, human toes, bare human feet, barefoot with toes, toes, shoes, sneakers"
            if "85" in wf and "inputs" in wf["85"]:
                wf["85"]["inputs"]["prompt"] = neg_prompt

            if "79" in wf and "inputs" in wf["79"]:
                wf["79"]["inputs"]["ref_boost"] = 1.3
            wf["29"]["inputs"]["filename_prefix"] = prefix

            res = requests.post(f"{COMFY_URL}/prompt", json={"prompt": wf, "client_id": f"batch_outfit_{safe_char}"}, timeout=8)
            if res.status_code != 200:
                continue
            prompt_id = res.json().get("prompt_id")

            start_wait = time.time()
            found_img = None
            while time.time() - start_wait < 75:
                time.sleep(2)
                try:
                    h_res = requests.get(f"{COMFY_URL}/history/{prompt_id}", timeout=5)
                    if h_res.status_code == 200:
                        h_data = h_res.json()
                        if prompt_id in h_data:
                            imgs = h_data[prompt_id].get("outputs", {}).get("29", {}).get("images", [])
                            if imgs:
                                found_img = imgs[0].get("filename")
                                break
                except Exception:
                    pass

            if found_img:
                outfit["image_url"] = f"/outfits/{found_img}"
                active_outfit_batches[book_id]["completed"] += 1
                active_outfit_batches[book_id]["last_image_url"] = f"/outfits/{found_img}"
                save_db(db, book_id)

    except Exception as e:
        active_outfit_batches[book_id]["error"] = str(e)
    finally:
        if book_id in active_outfit_batches:
            active_outfit_batches[book_id]["is_running"] = False

@app.post("/api/characters/generate_all_outfits")
def start_outfit_batch_endpoint(req: OutfitBatchRequest):
    if active_outfit_batches.get(req.book_id, {}).get("is_running"):
        return {"status": "already_running", **active_outfit_batches[req.book_id]}

    t = threading.Thread(
        target=run_outfit_batch_worker,
        args=(req.book_id, req.skip_existing),
        daemon=True
    )
    t.start()
    return {"status": "started", "book_id": req.book_id}

@app.get("/api/characters/{book_id}/outfit_batch_status")
def get_outfit_batch_status_endpoint(book_id: str):
    return active_outfit_batches.get(book_id, {
        "is_running": False,
        "total": 0,
        "completed": 0,
        "current_name": "",
        "current_index": 0,
        "last_image_url": None,
        "cancelled": False,
        "error": None
    })

@app.post("/api/characters/{book_id}/cancel_outfit_batch")
def cancel_outfit_batch_endpoint(book_id: str):
    if book_id in active_outfit_batches:
        active_outfit_batches[book_id]["cancelled"] = True
        active_outfit_batches[book_id]["is_running"] = False
    return {"status": "cancelled"}

# ==========================================
# UNIVERSAL PRODUCTION AGENT (GraphRAG + LangGraph)
# ==========================================

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.get("/agent", response_class=FileResponse)
def get_agent_ui():
    agent_html_path = os.path.join(STATIC_DIR, "agent.html")
    if os.path.exists(agent_html_path):
        return FileResponse(agent_html_path)
    raise HTTPException(status_code=404, detail="Agent UI not found")

@app.post("/api/agent/reset")
def reset_agent_studio():
    """Reset all character, environment, and production databases to zero (Clean Canvas)."""
    try:
        bible_path = os.path.join(DATA_DIR, "character_bible.json")
        with open(bible_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

        env_path = os.path.join(DATA_DIR, "environment_bible.json")
        with open(env_path, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2)

        db_path = os.path.join(DATA_DIR, "database.json")
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump({"characters": [], "scenes": [], "metadata": {"reset": True}}, f, indent=2)

        script_path = os.path.join(DATA_DIR, "script_bible.json")
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump({
                "dialogues": [],
                "voice_overs": [],
                "narrator_chunks": [],
                "narrator_voiceprint": {
                    "voiceprint_prompt": "A mature male cinematic narrator in his mid-40s with a deep, warm, resonant baritone. Measured, articulate, authoritative cadence with natural pauses between sentences. Calm, immersive storyteller demeanor. Rich studio microphone capture, zero room echo, warm low-end proximity.",
                    "timbre": "Deep resonant baritone with rich vocal chest resonance",
                    "cadence": "Measured cinematic rhythm, authoritative storytelling pace",
                    "acoustics": "Ultra-clean broadcast booth, zero reverb, dry proximity"
                }
            }, f, indent=2)

        scene_bible_path = os.path.join(DATA_DIR, "scene_bible.json")
        with open(scene_bible_path, "w", encoding="utf-8") as f:
            json.dump({"novel_title": "", "scenes": []}, f, indent=2)

        micro_path = os.path.join(DATA_DIR, "micro_scenes_bible.json")
        with open(micro_path, "w", encoding="utf-8") as f:
            json.dump({
                "novel_title": "",
                "pacing_standard": "2.0 words per second",
                "duration_constraint": "6.0s - 15.0s per take",
                "total_master_scenes": 0,
                "total_shots": 0,
                "total_runtime_seconds": 0.0,
                "total_runtime_formatted": "0m 00s",
                "shots": []
            }, f, indent=2)

        return {"status": "success", "message": "Studio reset to zero (Clean Canvas)"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_book_slug(book_id: Optional[str]) -> str:
    if not book_id:
        return ""
    clean = book_id.replace("__epub", "").strip()
    return re.sub(r'[^a-zA-Z0-9_-]', '_', clean).strip('_')

def get_book_bible_path(book_id: Optional[str], bible_name: str) -> str:
    slug = get_book_slug(book_id)
    if not slug:
        return os.path.join(DATA_DIR, bible_name)
    book_bible_dir = os.path.join(DATA_DIR, "bibles", slug)
    os.makedirs(book_bible_dir, exist_ok=True)
    return os.path.join(book_bible_dir, bible_name)

@app.get("/api/agent/characters")
def get_agent_characters(book_id: Optional[str] = Query(None)):
    """Retrieve all locked canonical characters with voiceprints for a specific book."""
    bible_path = get_book_bible_path(book_id, "character_bible.json")
    chars = []
    if os.path.exists(bible_path):
        try:
            with open(bible_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    ident = v.get("identity", {})
                    vp = v.get("voiceprint", {})
                    chars.append({
                        "name": ident.get("name", vp.get("character_name", k.capitalize())),
                        "species": ident.get("species", vp.get("species", "Unknown")),
                        "skin_tone_and_texture": ident.get("skin_tone_and_texture", "Canon verified"),
                        "eyes_description": ident.get("eyes_description", "Canon verified"),
                        "hair_description": ident.get("hair_description", "Canon verified"),
                        "unique_biology": ident.get("unique_biology", []),
                        "novel_quotes": ident.get("novel_quotes", []),
                        "face_portrait_prompt": ident.get("face_portrait_prompt", v.get("face_portrait_prompt", "")),
                        "portrait_image": ident.get("portrait_image", v.get("portrait_image", None)),
                        "wardrobes": ident.get("wardrobes", v.get("wardrobes", [])),
                        "voiceprint": vp if vp else None
                    })
        except Exception:
            pass
    return {"characters": chars}

@app.get("/api/agent/environments")
def get_agent_environments(book_id: Optional[str] = Query(None)):
    """Retrieve all locked canonical environments for a specific book."""
    env_path = get_book_bible_path(book_id, "environment_bible.json")
    envs = []
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    env = v.get("environment", {})
                    envs.append({
                        "name": env.get("name", k.replace("_", " ").title()),
                        "environment_type": env.get("environment_type", "Location"),
                        "architectural_style": env.get("architectural_style", "Canonical"),
                        "lighting_and_atmosphere": env.get("lighting_and_atmosphere", "Atmospheric"),
                        "key_props_and_spatial_features": env.get("key_props_and_spatial_features", []),
                        "sensory_details": env.get("sensory_details", ""),
                        "novel_quotes": env.get("novel_quotes", []),
                        "matte_painting_prompt": env.get("matte_painting_prompt", "")
                    })
        except Exception:
            pass
    return {"environments": envs}

class AgentRunRequest(BaseModel):
    book_id: str
    character: Optional[str] = ""

@app.post("/api/agent/run")
def run_agent_endpoint(req: AgentRunRequest):
    """Run the GraphRAG + LangGraph agent for the specified book and character."""
    import run_agent
    import novel_production_graph
    import voiceprint_production
    try:
        book_input = "Solar Dragons Need Love Too 1" if req.book_id and "solar" in req.book_id.lower() else req.book_id
        txt_path = run_agent.resolve_novel_text(book_input)
        bible_path = get_book_bible_path(req.book_id, "character_bible.json")
        app_graph = run_agent.build_production_graph()
        
        target_char = req.character.strip() if req.character else ""
        chars_to_run = []
        if target_char:
            chars_to_run = [target_char]
        else:
            discovered = novel_production_graph.discover_novel_characters(txt_path)
            chars_to_run = discovered[:4] if discovered else ["Protagonist"]
        
        logs = []
        for ch in chars_to_run:
            run_agent.run_for_character(app_graph, ch, txt_path, bible_path=bible_path)
            logs.append(f"[Success] Processed {ch} via LangGraph & GraphRAG")
            try:
                vp = voiceprint_production.extract_or_design_voiceprint(ch, txt_path)
                voiceprint_production.lock_voiceprint_into_bible(ch, vp, bible_path=bible_path)
                logs.append(f"[Voiceprint] Synthesized acoustic profile & original English dialogue for {ch}")
            except Exception as e_vp:
                logs.append(f"[Voiceprint Note] {ch}: {e_vp}")
        
        chars_res = get_agent_characters(book_id=req.book_id)
        return {
            "status": "success",
            "message": f"Successfully audited and locked {len(chars_to_run)} character(s) for {req.book_id}",
            "characters": chars_res.get("characters", []),
            "logs": logs
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "characters": get_agent_characters(book_id=req.book_id).get("characters", [])
        }

class VoiceprintRunRequest(BaseModel):
    book_id: str
    character: Optional[str] = ""

@app.post("/api/agent/run_voiceprint")
def run_voiceprint_endpoint(req: VoiceprintRunRequest):
    """Run dedicated voiceprint extraction for character."""
    import run_agent
    import voiceprint_production
    import novel_production_graph
    try:
        book_input = "Solar Dragons Need Love Too 1" if req.book_id and "solar" in req.book_id.lower() else req.book_id
        txt_path = run_agent.resolve_novel_text(book_input)
        bible_path = get_book_bible_path(req.book_id, "character_bible.json")
        target_char = req.character.strip() if req.character else ""
        if not target_char:
            discovered = novel_production_graph.discover_novel_characters(txt_path)
            target_char = discovered[0] if discovered else "Protagonist"
            
        vp = voiceprint_production.extract_or_design_voiceprint(target_char, txt_path)
        voiceprint_production.lock_voiceprint_into_bible(target_char, vp, bible_path=bible_path)
        chars_res = get_agent_characters(book_id=req.book_id)
        return {
            "status": "success",
            "message": f"Voiceprint generated for {target_char}",
            "characters": chars_res.get("characters", []),
            "logs": [
                f"[Voiceprint] 5-dimension acoustic profile created for {target_char}",
                f"[Original Dialogue] Original English lines and [Instruction:] tags extracted"
            ]
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "characters": get_agent_characters(book_id=req.book_id).get("characters", [])
        }

class EnvironmentRunRequest(BaseModel):
    book_id: str
    location: Optional[str] = ""

@app.post("/api/agent/run_environment")
def run_environment_endpoint(req: EnvironmentRunRequest):
    """Run GraphRAG + LangGraph Environment Agent for novel location."""
    import run_agent
    import environment_production_graph
    try:
        book_input = "Solar Dragons Need Love Too 1" if req.book_id and "solar" in req.book_id.lower() else req.book_id
        txt_path = run_agent.resolve_novel_text(book_input)
        bible_path = get_book_bible_path(req.book_id, "environment_bible.json")
        
        target_loc = req.location.strip() if req.location else ""
        locs_to_run = []
        if target_loc:
            locs_to_run = [target_loc]
        else:
            discovered_locs = environment_production_graph.discover_novel_locations(txt_path)
            locs_to_run = discovered_locs[:3] if discovered_locs else ["Primary Setting"]
            
        app_graph = environment_production_graph.build_environment_graph()
        logs = []
        for loc in locs_to_run:
            initial_state = {
                "location_name": loc,
                "novel_text_path": txt_path,
                "retrieved_passages": [],
                "canonical_environment": None,
                "conflicts": [],
                "user_resolutions": {},
                "is_locked": False,
                "matte_painting_prompt": None,
                "bible_path": bible_path
            }
            config = {"configurable": {"thread_id": f"env_run_{re.sub(r'[^a-zA-Z0-9]', '_', loc.lower())}"}}
            app_graph.invoke(initial_state, config=config)
            logs.append(f"[Success] Extracted environment: {loc}")
        
        envs_res = get_agent_environments(book_id=req.book_id)
        return {
            "status": "success",
            "message": f"Successfully extracted and locked {len(locs_to_run)} environment(s)",
            "environments": envs_res.get("environments", []),
            "logs": logs
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "environments": get_agent_environments(book_id=req.book_id).get("environments", [])
        }

@app.get("/api/agent/script")
def get_agent_script(book_id: Optional[str] = Query(None)):
    """Retrieve all extracted dialogues, voice-overs, and narrator lines for a book."""
    import dialogue_production_graph
    bible_path = get_book_bible_path(book_id, "script_bible.json")
    data = dialogue_production_graph.load_script_bible(bible_path=bible_path)
    return data

class ScriptRunRequest(BaseModel):
    book_id: str
    character: Optional[str] = ""
    scope: Optional[str] = "all"  # all | dialogue | voiceover | narrator

@app.post("/api/agent/run_script")
def run_script_endpoint(req: ScriptRunRequest):
    """Run GraphRAG + LangGraph Script Agent for novel dialogues, VO, and narrator."""
    import uuid
    import run_agent
    import dialogue_production_graph
    try:
        book_input = "Solar Dragons Need Love Too 1" if req.book_id and "solar" in req.book_id.lower() else req.book_id
        txt_path = run_agent.resolve_novel_text(book_input)
        bible_path = get_book_bible_path(req.book_id, "script_bible.json")
        
        target_char = req.character.strip() if req.character else None
        scope = req.scope.strip().lower() if req.scope else "all"
        
        app_graph = dialogue_production_graph.build_script_pipeline_graph()
        initial_state = {
            "novel_text_path": txt_path,
            "scope_filter": scope,
            "target_character": target_char,
            "target_scene": None,
            "raw_passages": [],
            "extracted_script": None,
            "is_locked": False,
            "bible_path": bible_path
        }
        config = {"configurable": {"thread_id": f"script_run_{uuid.uuid4().hex[:6]}"}}
        app_graph.invoke(initial_state, config=config)
        
        script_res = dialogue_production_graph.load_script_bible(bible_path=bible_path)
        return {
            "status": "success",
            "message": f"Successfully extracted script ({scope}) for {req.book_id}",
            "script": script_res,
            "logs": [
                f"[Script Agent] Ingested text from: {os.path.basename(txt_path)}",
                f"[Dialogues] Extracted {len(script_res.get('dialogues', []))} character exchanges in original English",
                f"[Voice-Over] Extracted {len(script_res.get('voice_overs', []))} internal monologues with [Instruction:] tags",
                f"[Narrator] Extracted {len(script_res.get('narrator_chunks', []))} cinematic exposition chunks",
                f"[Audio Locked] Saved to {bible_path}"
            ]
        }
    except Exception as e:
        import dialogue_production_graph
        bible_path = get_book_bible_path(req.book_id, "script_bible.json")
        return {
            "status": "error",
            "message": str(e),
            "script": dialogue_production_graph.load_script_bible(bible_path=bible_path)
        }

@app.get("/api/agent/scenes")
def get_agent_scenes(book_id: Optional[str] = Query(None)):
    """Retrieve all locked complete scenes for a specific book."""
    import scene_production_graph
    bible_path = get_book_bible_path(book_id, "scene_bible.json")
    data = scene_production_graph.load_scene_bible(bible_path=bible_path)
    return data

class SceneRunRequest(BaseModel):
    book_id: str
    max_scenes: Optional[int] = 12

@app.post("/api/agent/run_scenes")
def run_scenes_endpoint(req: SceneRunRequest):
    """Run GraphRAG + LangGraph Scene Agent for complete scenes."""
    import uuid
    import run_agent
    import scene_production_graph
    try:
        book_input = "Solar Dragons Need Love Too 1" if req.book_id and "solar" in req.book_id.lower() else req.book_id
        txt_path = run_agent.resolve_novel_text(book_input)
        bible_path = get_book_bible_path(req.book_id, "scene_bible.json")
        
        app_graph = scene_production_graph.build_scene_pipeline_graph()
        initial_state = {
            "novel_text_path": txt_path,
            "target_chapter_or_query": None,
            "raw_scene_blocks": [],
            "extracted_scenes": [],
            "is_locked": False,
            "bible_path": bible_path
        }
        config = {"configurable": {"thread_id": f"scene_run_{uuid.uuid4().hex[:6]}"}}
        app_graph.invoke(initial_state, config=config)
        
        bible_res = scene_production_graph.load_scene_bible(bible_path=bible_path)
        scenes_list = bible_res.get("scenes", [])
        return {
            "status": "success",
            "message": f"Successfully extracted and locked {len(scenes_list)} complete scenes for {req.book_id}",
            "scenes": scenes_list,
            "scene_bible": bible_res,
            "logs": [
                f"[Scene Agent] Ingested manuscript from: {os.path.basename(txt_path)}",
                f"[Segmentation] Segmented narrative into {len(scenes_list)} complete cinematic sequences",
                f"[Sluglines] Assigned INT./EXT. locations and time of day",
                f"[Keyframes] Generated ComfyUI prompts for all scenes",
                f"[Audio Integrated] Linked dialogues, voice-overs, and soundscape foley",
                f"[Locked] Complete Scene Bible saved to {bible_path}"
            ]
        }
    except Exception as e:
        import scene_production_graph
        bible_path = get_book_bible_path(req.book_id, "scene_bible.json")
        bible_res = scene_production_graph.load_scene_bible(bible_path=bible_path)
        return {
            "status": "error",
            "message": str(e),
            "scenes": bible_res.get("scenes", []),
            "scene_bible": bible_res
        }

@app.get("/api/agent/micro_scenes")
def get_agent_micro_scenes(book_id: Optional[str] = Query(None)):
    """Retrieve all locked 6s-15s micro-scenes / cinematic shots for a book."""
    import micro_scene_production_graph
    bible_path = get_book_bible_path(book_id, "micro_scenes_bible.json")
    data = micro_scene_production_graph.load_micro_scenes_bible(bible_path=bible_path)
    return data

class MicroSceneRunRequest(BaseModel):
    book_id: Optional[str] = ""
    min_duration: Optional[float] = 6.0
    max_duration: Optional[float] = 15.0
    words_per_sec: Optional[float] = 2.0

@app.post("/api/agent/run_micro_scenes")
def run_micro_scenes_endpoint(req: MicroSceneRunRequest):
    """Run GraphRAG + LangGraph Micro-Scene / Takes Breakdown Agent."""
    import micro_scene_production_graph
    try:
        scene_bible_path = get_book_bible_path(req.book_id, "scene_bible.json")
        micro_bible_path = get_book_bible_path(req.book_id, "micro_scenes_bible.json")
        graph = micro_scene_production_graph.build_micro_scene_graph()
        initial_state = {
            "scene_bible_path": scene_bible_path,
            "micro_bible_path": micro_bible_path,
            "novel_title": req.book_id or "Novel Adaptation",
            "raw_scenes": [],
            "micro_shots": [],
            "min_duration": req.min_duration or 6.0,
            "max_duration": req.max_duration or 15.0,
            "words_per_sec": req.words_per_sec or 2.0,
            "is_locked": False
        }
        config = {"configurable": {"thread_id": f"micro_run_{re.sub(r'[^a-zA-Z0-9]', '_', str(req.book_id).lower())}"}}
        graph.invoke(initial_state, config=config)
        bible = micro_scene_production_graph.load_micro_scenes_bible(bible_path=micro_bible_path)
        shots = bible.get("shots", [])
        return {
            "status": "success",
            "message": f"Successfully decomposed {bible.get('total_master_scenes')} master scenes into {bible.get('total_shots')} cinematic takes",
            "micro_scenes": bible,
            "shots": shots,
            "logs": [
                f"[MicroScene Agent] Loaded master scenes from {scene_bible_path}",
                f"[Pacing Calibration] Applied 2.0 words/second speech cadence formula",
                f"[Duration Boundaries] Calibrated all takes strictly between {req.min_duration or 6.0}s and {req.max_duration or 15.0}s",
                f"[AI Video Prompts] Generated MiniMax H3 / Kling / Runway Gen-3 prompts with lens specs",
                f"[Locked] Successfully locked {len(shots)} takes into {micro_bible_path}"
            ]
        }
    except Exception as e:
        import micro_scene_production_graph
        micro_bible_path = get_book_bible_path(req.book_id, "micro_scenes_bible.json")
        bible_res = micro_scene_production_graph.load_micro_scenes_bible(bible_path=micro_bible_path)
        return {
            "status": "error",
            "message": str(e),
            "micro_scenes": bible_res,
            "shots": bible_res.get("shots", [])
        }
    except Exception as e:
        import micro_scene_production_graph
        bible = micro_scene_production_graph.load_micro_scenes_bible()
        return {
            "status": "error",
            "message": str(e),
            "micro_scenes": bible,
            "shots": bible.get("shots", [])
        }


# ==========================================
# COMFYUI STUDIO GENERATION & AUTOMATION ENDPOINTS
# ==========================================

COMFY_AUDIO_DIR = r"C:\ComfyUI_windows_portable\ComfyUI\output\audio"
COMFY_ENV_DIR = r"C:\ComfyUI_windows_portable\ComfyUI\output\environments"
COMFY_VIDEO_DIR = r"C:\ComfyUI_windows_portable\ComfyUI\output\video"

os.makedirs(COMFY_AUDIO_DIR, exist_ok=True)
os.makedirs(COMFY_ENV_DIR, exist_ok=True)
os.makedirs(COMFY_VIDEO_DIR, exist_ok=True)

class ComfyVoiceRequest(BaseModel):
    character_name: str
    line_en: str
    voice_description: str
    shot_code: Optional[str] = None

@app.post("/api/comfy/generate_voice")
def comfy_generate_voice(req: ComfyVoiceRequest):
    import comfy_studio_engine
    if not comfy_studio_engine.is_comfy_online():
        raise HTTPException(status_code=503, detail="ComfyUI server is offline on 127.0.0.1:8188")
    try:
        res = comfy_studio_engine.generate_voice_qwen(
            character_name=req.character_name,
            line_en=req.line_en,
            voice_description=req.voice_description,
            shot_code=req.shot_code
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ComfyEnvRequest(BaseModel):
    location_name: str
    prompt_text: str
    aspect_ratio: Optional[str] = "16:9 (Widescreen)"

@app.post("/api/comfy/generate_environment")
def comfy_generate_environment(req: ComfyEnvRequest):
    import comfy_studio_engine
    if not comfy_studio_engine.is_comfy_online():
        raise HTTPException(status_code=503, detail="ComfyUI server is offline on 127.0.0.1:8188")
    try:
        res = comfy_studio_engine.generate_environment_krea(
            location_name=req.location_name,
            prompt_text=req.prompt_text,
            aspect_ratio=req.aspect_ratio or "16:9 (Widescreen)"
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ComfyPortraitRequest(BaseModel):
    character_name: str
    prompt_text: str
    book_id: Optional[str] = ""
    aspect_ratio: Optional[str] = "1:1 (Square)"

@app.post("/api/comfy/generate_portrait")
def comfy_generate_portrait(req: ComfyPortraitRequest):
    import comfy_studio_engine
    if not comfy_studio_engine.is_comfy_online():
        raise HTTPException(status_code=503, detail="ComfyUI server is offline on 127.0.0.1:8188")
    try:
        res = comfy_studio_engine.generate_portrait_krea(
            character_name=req.character_name,
            prompt_text=req.prompt_text,
            aspect_ratio=req.aspect_ratio or "1:1 (Square)"
        )
        if res.get("primary_image"):
            bible_path = get_book_bible_path(req.book_id, "character_bible.json")
            if os.path.exists(bible_path):
                try:
                    with open(bible_path, "r", encoding="utf-8") as f:
                        b_data = json.load(f)
                    c_key = req.character_name.lower().strip()
                    if c_key in b_data:
                        b_data[c_key].setdefault("identity", {})["portrait_image"] = res["primary_image"]
                        b_data[c_key]["portrait_image"] = res["primary_image"]
                        with open(bible_path, "w", encoding="utf-8") as f:
                            json.dump(b_data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ComfyActorOutfitRequest(BaseModel):
    character_name: str
    outfit_prompt: str
    portrait_filename: Optional[str] = ""
    outfit_id: Optional[str] = ""
    book_id: Optional[str] = ""
    aspect_ratio: Optional[str] = "9:16 (Portrait Widescreen)"

@app.post("/api/comfy/generate_actor_outfit")
def comfy_generate_actor_outfit(req: ComfyActorOutfitRequest):
    import comfy_studio_engine
    if not comfy_studio_engine.is_comfy_online():
        raise HTTPException(status_code=503, detail="ComfyUI server is offline on 127.0.0.1:8188")
    try:
        portrait_fn = (req.portrait_filename or "").strip()
        bible_path = get_book_bible_path(req.book_id, "character_bible.json")
        b_data = {}
        if os.path.exists(bible_path):
            try:
                with open(bible_path, "r", encoding="utf-8") as f:
                    b_data = json.load(f)
            except Exception:
                pass

        c_key = req.character_name.lower().strip()
        char_entry = b_data.get(c_key, {})
        
        # If portrait filename not explicitly supplied, find from character bible
        if not portrait_fn:
            p_img = char_entry.get("portrait_image") or char_entry.get("identity", {}).get("portrait_image")
            if p_img:
                portrait_fn = os.path.basename(p_img)

        # Fallback: check character_portraits directory for cine_{slug}
        if not portrait_fn:
            slug = comfy_studio_engine.clean_slug(req.character_name)
            portraits_dir = r"C:\ComfyUI_windows_portable\ComfyUI\output\character_portraits"
            if os.path.exists(portraits_dir):
                for f in os.listdir(portraits_dir):
                    if f"cine_{slug}" in f and f.endswith(('.png', '.jpg', '.webp')):
                        portrait_fn = f
                        break

        if not portrait_fn:
            raise HTTPException(status_code=400, detail=f"No facial portrait found for {req.character_name}. Please render the face portrait first via Krea 2 Turbo!")

        res = comfy_studio_engine.generate_actor_outfit_krea(
            character_name=req.character_name,
            portrait_filename=portrait_fn,
            outfit_prompt=req.outfit_prompt,
            aspect_ratio=req.aspect_ratio or "9:16 (Portrait Widescreen)"
        )

        # Update character wardrobe entry with generated outfit image
        if res.get("primary_image") and b_data and c_key in b_data:
            try:
                wardrobes = char_entry.get("identity", {}).get("wardrobes", []) or char_entry.get("wardrobes", [])
                for w in wardrobes:
                    if not req.outfit_id or w.get("id") == req.outfit_id:
                        w["outfit_image"] = res["primary_image"]
                        break
                with open(bible_path, "w", encoding="utf-8") as f:
                    json.dump(b_data, f, indent=2, ensure_ascii=False)
            except Exception:
                pass

        return res
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class ComfyVideoShotRequest(BaseModel):
    shot_code: str
    prompt_6part: str
    duration_seconds: Optional[float] = 8.0
    actor_image_filename: Optional[str] = None
    env_image_filename: Optional[str] = None
    audio_filename: Optional[str] = None

@app.post("/api/comfy/generate_video_shot")
def comfy_generate_video_shot(req: ComfyVideoShotRequest):
    import comfy_studio_engine
    if not comfy_studio_engine.is_comfy_online():
        raise HTTPException(status_code=503, detail="ComfyUI server is offline on 127.0.0.1:8188")
    try:
        res = comfy_studio_engine.generate_video_shot_minimax(
            shot_code=req.shot_code,
            prompt_6part=req.prompt_6part,
            duration_seconds=req.duration_seconds or 8.0,
            actor_image_filename=req.actor_image_filename,
            env_image_filename=req.env_image_filename,
            audio_filename=req.audio_filename
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/comfy/status")
def comfy_status():
    import comfy_studio_engine
    online = comfy_studio_engine.is_comfy_online()
    return {
        "online": online,
        "url": comfy_studio_engine.COMFY_URL,
        "workflows": {
            "voice_qwen3": os.path.exists(comfy_studio_engine.WF_VOICE_PATH),
            "env_krea2": os.path.exists(comfy_studio_engine.WF_ENV_PATH),
            "actor_outfit_krea2": os.path.exists(comfy_studio_engine.WF_ACTOR_OUTFIT_PATH),
            "portrait_krea2": os.path.exists(comfy_studio_engine.WF_PORTRAIT_PATH),
            "video_minimax_h3": os.path.exists(comfy_studio_engine.WF_VIDEO_PATH)
        }
    }


app.mount("/outfits", StaticFiles(directory=COMFY_OUTFITS_DIR), name="outfits")
app.mount("/character_outfits", StaticFiles(directory=COMFY_OUTFITS_DIR), name="character_outfits")
app.mount("/portraits", StaticFiles(directory=COMFY_PORTRAITS_DIR), name="portraits")
app.mount("/audio", StaticFiles(directory=COMFY_AUDIO_DIR), name="audio")
app.mount("/environments", StaticFiles(directory=COMFY_ENV_DIR), name="environments")
app.mount("/video", StaticFiles(directory=COMFY_VIDEO_DIR), name="video")
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8555)
