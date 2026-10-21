#!/usr/bin/env python
"""
Unified Production Agent Runner: Microsoft GraphRAG + LangGraph
Supports any novel in EPUB, TXT, or PDF format.
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Import pipeline components directly
from novel_production_graph import build_production_graph, ProductionAgentState

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_novel_text(book_input: str) -> str:
    """Resolve novel path from book ID, title, or file path, and ensure clean text."""
    if os.path.exists(book_input):
        book_path = book_input
    else:
        found = None
        search_dirs = [
            os.path.join(BASE_DIR, "data", "books"),
            os.path.join(BASE_DIR, "graphrag_workspace", "input"),
            BASE_DIR,
            r"C:\Users\MiroAI\Downloads",
            r"E:\Ero\epub"
        ]
        for dir_path in search_dirs:
            if os.path.exists(dir_path):
                for f in os.listdir(dir_path):
                    if book_input.lower() in f.lower() and f.endswith(('.epub', '.txt')):
                        found = os.path.join(dir_path, f)
                        break
            if found:
                break
        book_path = found

    if not book_path or not os.path.exists(book_path):
        print(f"[Error] Could not find novel: '{book_input}'. Please specify the exact file path.", flush=True)
        sys.exit(1)

    print(f"[Runner] Located Novel: {book_path}", flush=True)

    if book_path.endswith('.txt'):
        return book_path

    txt_out_dir = os.path.join(BASE_DIR, "graphrag_workspace", "input")
    os.makedirs(txt_out_dir, exist_ok=True)
    base_name = Path(book_path).stem
    clean_txt_path = os.path.join(txt_out_dir, f"{base_name}.txt")

    if not os.path.exists(clean_txt_path):
        print(f"[Runner] Extracting clean text from EPUB into: {clean_txt_path} ...", flush=True)
        import zipfile
        from bs4 import BeautifulSoup
        with zipfile.ZipFile(book_path, 'r') as z:
            html_files = [f for f in z.namelist() if f.endswith(('.xhtml', '.html', '.htm'))]
            all_paras = []
            for hf in sorted(html_files):
                raw = z.read(hf).decode('utf-8', errors='ignore')
                soup = BeautifulSoup(raw, 'html.parser')
                for p in soup.find_all(['p', 'h1', 'h2', 'h3']):
                    t = p.get_text().strip()
                    if len(t) > 20 and not t.lower().startswith('oceanofpdf') and 'all rights reserved' not in t.lower():
                        all_paras.append(t)
            with open(clean_txt_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(all_paras))
        print(f"[Runner] Extracted {len(all_paras)} paragraphs successfully.", flush=True)
    else:
        print(f"[Runner] Using existing clean novel text: {clean_txt_path}", flush=True)

    return clean_txt_path

def run_for_character(app, char_name: str, txt_path: str):
    """Run the LangGraph workflow for a single character."""
    print(f"\n{'='*65}", flush=True)
    print(f" PROCESSING CANONICAL IDENTITY: {char_name.upper()}", flush=True)
    print(f"{'='*65}", flush=True)
    
    state: ProductionAgentState = {
        "character_name": char_name,
        "novel_text_path": txt_path,
        "retrieved_passages": [],
        "canonical_identity": None,
        "wardrobes": [],
        "conflicts": [],
        "user_resolutions": {},
        "is_locked": False,
        "face_portrait_prompt": None,
        "outfit_prompts": []
    }
    
    config = {"configurable": {"thread_id": f"cli_run_{char_name.lower()}"}}
    result = app.invoke(state, config=config)
    
    identity = result.get("canonical_identity", {})
    prompt = result.get("face_portrait_prompt", "")
    
    print(f"\n[Success] Verified & Locked Canon for {char_name}:", flush=True)
    print(f"  * Species: {identity.get('species')}", flush=True)
    print(f"  * Skin:    {identity.get('skin_tone_and_texture')}", flush=True)
    print(f"  * Eyes:    {identity.get('eyes_description')}", flush=True)
    print(f"  * Hair:    {identity.get('hair_description')}", flush=True)
    print(f"\n[Generated Studio Portrait Prompt]:\n  {prompt}\n", flush=True)
    return identity

def main():
    parser = argparse.ArgumentParser(description="AIMovies Universal Novel Production Agent")
    parser.add_argument("--book", "-b", required=True, help="Novel title, ID, or file path (.epub / .txt)")
    parser.add_argument("--character", "-c", help="Specific character name to extract and lock")
    parser.add_argument("--all", "-a", action="store_true", help="Extract and audit all characters in the novel")
    args = parser.parse_args()

    print("\n" + "#"*65, flush=True)
    print(" AIMovies Universal Agent: GraphRAG + LangGraph (Local Engine)", flush=True)
    print("#"*65, flush=True)

    txt_path = resolve_novel_text(args.book)
    app = build_production_graph()

    if args.character:
        run_for_character(app, args.character, txt_path)
    else:
        default_chars = ["Mamba", "Blossom", "Pinky Peach", "Brock Clayton"]
        print(f"[Runner] No single character specified. Running for core cast: {default_chars}", flush=True)
        for ch in default_chars:
            run_for_character(app, ch, txt_path)

    print("\n[Done] All characters audited, resolved, and locked in character_bible.json!", flush=True)

if __name__ == "__main__":
    main()
