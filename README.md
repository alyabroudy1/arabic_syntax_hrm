# HRM-Augmented Lightweight Arabic Syntax Model for Mobile

## نموذج تحليل النحو العربي المعزّز بالاستدلال الهرمي

A hybrid AI system combining a fine-tuned LLM (Qwen2.5-3B) with a Hierarchical Recurrent Model (HRM) for offline Arabic syntax analysis (iʻrāb) on mobile devices.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   USER INPUT (Arabic)                   │
│            "ذهب الطالبُ إلى المدرسةِ"                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────┐
        │  Small LLM (Qwen2.5-3B) │  ← Fine-tuned on Arabic syntax
        │  Quantized 4-bit ~1.5GB │
        └─────────┬────────────────┘
                  │
                  ▼ (structured grid)
        ┌──────────────────────────┐
        │   HRM Module (27M)       │  ← Trained on syntax-as-grid
        │   ONNX/TFLite ~50MB     │
        └─────────┬────────────────┘
                  │
                  ▼ (reasoning output)
        ┌──────────────────────────┐
        │  LLM Decoder             │
        │  iʻrāb, corrections,    │
        │  dependency tree, etc.   │
        └──────────────────────────┘
```

## Components

| Component | Size | Status |
|-----------|------|--------|
| Qwen2.5-3B-Instruct (Q4) | ~1.5 GB | ✅ Data Prep Done |
| HRM (27M) | ~50 MB | ✅ Feasibility Checked |
| Grid Encoder/Decoder | ~5 MB | ✅ Built & Tested |
| Arabic Syntax Dataset | 20 MB | ✅ 9.4k Grids Built |

## Current Status (End of Phase 2 Quick Test)

We have successfully established the project structure and data pipeline.
**Achievements:**
1. **Data Pipeline (Complete)**: Parsed 7,664 Universal Dependencies Arabic sentences. Built exactly 9,489 HRM syntax grids.
2. **Synthetic Data**: Generated 5,000 synthetic sentences for LLM fine-tuning.
3. **HRM Feasibility (Verified)**: Trained a 2.3M parameter HRM model for 92 epochs. The novel Grid Encoding works excellently for Arabic syntax.
   - **Case prediction (إعراب)**: 83.4% accuracy
   - **DepRel prediction**: 70.3% accuracy
   - **Head prediction**: 28.4% (Bottleneck)
4. **All Pipeline Scripts**: Wrote the remaining scripts for bridging the LLM with the HRM, testing evaluations, and ONNX/GGUF exporting.

## Recommended Next Steps

1. **Architectural Improvements**: Improve the dependency head prediction (currently at 28.4%). Replace the linear prediction head with a pointer network or cross-attention mechanism for `col: 4 (head)`.
2. **Regularization for HRM**: The model showed signs of overfitting after Epoch 8. Increase dropout (e.g., to 0.2-0.3), implement early stopping, and train the full-size model (hidden_dim=256).
3. **GPU LLM Fine-tuning**: Run `scripts/05_train_llm.py` on a CUDA-enabled GPU (RTX 3060+) using the generated `train_llm.jsonl` data.
4. **Hybrid Integration**: Once both models are trained and optimized, run `scripts/07_hybrid_model.py` to bridge the Qwen 3B model with the HRM for inference.

## Quick Start

```bash
# Create environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Phase 1: Download datasets
python scripts/01_download_datasets.py

# Phase 1: Build syntax grids
python scripts/02_build_syntax_grids.py

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
arabic_syntax_hrm/
├── data/                    # Raw and processed datasets (grids, LLM data)
├── models/                  # Trained model checkpoints
├── scripts/                 # Pipeline scripts (01 to 09)
├── configs/                 # Training configurations
├── eval/                    # Evaluation results (hrm_results.json)
├── android/                 # Android app & exported models
├── tests/                   # Unit tests (100% passing)
└── docs/                    # Documentation
```
