#!/usr/bin/env python3
"""
Gradio Demo: Arabic HRM-Grid Parser V2
Browse test examples and see predicted vs gold parse trees.
"""

import sys
import json
import torch
import numpy as np
import gradio as gr
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.v2.parser import ArabicHRMGridParserV2, ParserConfig

DATA_DIR = PROJECT_ROOT / "data" / "arabic_syntax_grid"
MODEL_DIR = PROJECT_ROOT / "models" / "v2_arabic_syntax"

# Reverse maps for display
DEP_RELS_INV = {
    0: '_', 1: 'root', 2: 'nsubj', 3: 'obj', 4: 'iobj', 5: 'obl',
    6: 'advmod', 7: 'amod', 8: 'nmod', 9: 'det', 10: 'case',
    11: 'conj', 12: 'cc', 13: 'punct', 14: 'flat', 15: 'compound',
    16: 'appos', 17: 'acl', 18: 'advcl', 19: 'cop', 20: 'mark',
    21: 'dep', 22: 'parataxis', 23: 'fixed', 24: 'vocative',
    25: 'nummod', 26: 'flat:foreign', 27: 'nsubj:pass', 28: 'csubj',
    29: 'xcomp', 30: 'ccomp', 31: 'orphan',
}
CASE_INV = {0: '—', 1: 'Nom', 2: 'Acc', 3: 'Gen'}

# Load model once
print("Loading model...")
ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location="cpu", weights_only=False)
config = ParserConfig(word_dim=384, n_heads=6, n_transformer_layers=3, n_gnn_rounds=3, n_relations=50, n_cases=4)
model = ArabicHRMGridParserV2(config)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"Model loaded (epoch={ckpt.get('epoch','?')}, UAS={ckpt.get('uas','?')}%)")

# Load test data
print("Loading test data...")
import importlib.util
spec = importlib.util.spec_from_file_location("train", PROJECT_ROOT / "scripts" / "10_train_v2.py")
train_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_mod)
RealArabicV2Dataset = train_mod.RealArabicV2Dataset

test_dataset = RealArabicV2Dataset(DATA_DIR, split="test", max_len=32, device="cpu")
test_texts = json.load(open(DATA_DIR / "test_texts.json", "r", encoding="utf-8"))
print(f"Loaded {len(test_texts)} test sentences")


def format_parse_table(words, heads, rels, cases, gold_heads, gold_rels, gold_cases, mask):
    """Format a comparison table of predicted vs gold parses."""
    n_words = int(mask.sum().item())
    
    rows = []
    rows.append("| # | Word (feature) | Pred Head→ | Pred Rel | Pred Case | Gold Head→ | Gold Rel | Gold Case | ✓ |")
    rows.append("|---|----------------|-----------|----------|-----------|-----------|----------|-----------|---|")
    
    correct_h = 0
    correct_las = 0
    total = 0
    
    for i in range(n_words):
        ph = int(heads[i].item())
        pr = DEP_RELS_INV.get(int(rels[i].item()), f"?{int(rels[i].item())}")
        pc = CASE_INV.get(int(cases[i].item()), '?')
        
        gh = int(gold_heads[i].item())
        gr_label = DEP_RELS_INV.get(int(gold_rels[i].item()), f"?{int(gold_rels[i].item())}")
        gc = CASE_INV.get(int(gold_cases[i].item()), '?')
        
        h_match = (ph == gh)
        las_match = h_match and (pr == gr_label)
        
        if h_match:
            correct_h += 1
        if las_match:
            correct_las += 1
        total += 1
        
        mark = "✅" if las_match else ("🟡" if h_match else "❌")
        rows.append(f"| {i+1} | token_{i} | →{ph} | {pr} | {pc} | →{gh} | {gr_label} | {gc} | {mark} |")
    
    uas = correct_h / max(total, 1) * 100
    las = correct_las / max(total, 1) * 100
    
    header = f"**Sentence UAS: {uas:.1f}% | LAS: {las:.1f}%** ({correct_h}/{total} heads, {correct_las}/{total} full)\n\n"
    return header + "\n".join(rows)


def predict_sentence(idx):
    """Run model on test sentence #idx."""
    idx = int(idx)
    if idx < 0 or idx >= len(test_texts):
        return "Invalid index", ""
    
    # Get batch (single sentence)
    batch = test_dataset[idx]
    # Add batch dimension
    batch_input = {k: v.unsqueeze(0) for k, v in batch.items()}
    
    with torch.no_grad():
        out = model(batch_input, epoch=0, training=False)
    
    pred_heads = out['pred_heads'][0]
    pred_rels = out['pred_rels'][0]
    pred_cases = out['pred_cases'][0]
    
    gold_heads = batch['heads']
    gold_rels = batch['relations']
    gold_cases = batch['cases']
    mask = batch['mask']
    
    text = test_texts[idx]
    table = format_parse_table(
        None, pred_heads, pred_rels, pred_cases,
        gold_heads, gold_rels, gold_cases, mask
    )
    
    return f"## 📝 {text}", table


def random_sentence():
    """Pick a random test sentence."""
    idx = np.random.randint(0, len(test_texts))
    return idx, *predict_sentence(idx)


# Build Gradio UI
with gr.Blocks(title="Arabic HRM Parser V2", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🕌 Arabic Dependency Parser (HRM-Grid V2)")
    gr.Markdown(f"**Model**: 32.78M params | **Test UAS**: 75.82% | **Test LAS**: 69.56% | **Test sentences**: {len(test_texts)}")
    
    with gr.Row():
        idx_slider = gr.Slider(0, len(test_texts)-1, value=0, step=1, label="Test Sentence #")
        random_btn = gr.Button("🎲 Random", variant="secondary")
    
    text_display = gr.Markdown("Click 'Random' or adjust slider to see a sentence")
    parse_table = gr.Markdown("")
    
    idx_slider.change(predict_sentence, inputs=idx_slider, outputs=[text_display, parse_table])
    random_btn.click(random_sentence, outputs=[idx_slider, text_display, parse_table])

demo.launch(share=False)
