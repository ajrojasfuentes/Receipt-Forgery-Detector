# Receipt Forgery Detector: LLM-Judge Multi-Agent System

> **An AI-powered receipt forgery detection system using multiple LLM judges with majority voting to classify receipts as REAL or FAKE.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset: "Find it again!"](#2-dataset-find-it-again)
3. [Solution Design & Architecture](#3-solution-design--architecture)
4. [Brainstorming & Decision Log](#4-brainstorming--decision-log)
5. [Technology Stack](#5-technology-stack)
6. [Project Structure](#6-project-structure)
7. [8-Day Development Plan](#7-eight-day-development-plan)
8. [Expected Results](#8-expected-results)
9. [Setup & Installation](#9-setup--installation)
10. [Usage](#10-usage)
11. [References](#11-references)

---

## 1. Project Overview

### Problem Statement

Digital documents such as receipts are widely used as supporting evidence for expense claims, tax filings, and reimbursement processes. The ease of editing these documents with modern image editing software makes them vulnerable to forgery. This project builds a **multi-LLM judge system** that evaluates receipt images and votes on whether they are authentic or forged.

### Objective

Build a tool where **3 LLM "judges"** independently analyze a receipt image and cast a vote: **FAKE**, **REAL**, or **UNCERTAIN**, along with a confidence score and reasoning. A **majority voting mechanism** aggregates the individual decisions into a final verdict.

### Key Requirements (from project statement)

| Requirement | Description |
|---|---|
| **Dataset exploration** | Count REAL vs FAKE, distribution of totals, additional insights |
| **20-Receipt evaluation set** | 10 REAL + 10 FAKE randomly sampled (documented seed) |
| **3 LLM Judges** | Different models or same model with different personas/temperatures |
| **Structured output** | `{"label": "FAKE\|REAL\|UNCERTAIN", "confidence": 0.0-100.0, "reasons": [...]}` |
| **Majority voting** | Aggregation logic for the 3 judge decisions |
| **Demo notebook** | Complete walkthrough with analysis and results |
| **Docker deployment** | Containerized for portability and stability |
| **Optional UI** | Interactive interface (Streamlit chosen) |

---

## 2. Dataset: "Find it again!"

### Overview

The **"Find it again!"** dataset (Tornés et al., ICDAR 2023) is a benchmark for document forgery detection containing **988 scanned receipt images** from the SROIE dataset, of which **163 have been realistically forged**.

- **Paper**: [Receipt Dataset for Document Forgery Detection](https://link.springer.com/chapter/10.1007/978-3-031-41682-8_28) (ICDAR 2023)
- **Source**: [L3i Lab, University of La Rochelle](https://l3i-share.univ-lr.fr/2023Finditagain/index.html)
- **Download**: [findit2.zip](https://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip) (~658 MB)

### Dataset Statistics

| Metric | Value |
|---|---|
| **Total receipts** | 988 |
| **REAL (authentic)** | 825 (83.5%) |
| **FAKE (forged)** | 163 (16.5%) |
| **Class imbalance ratio** | 5.1:1 (REAL:FAKE) |
| **Train / Test / Val** | 577 / 218 / 193 |

**Per-split breakdown:**

| Split | Total | REAL | FAKE |
|---|---|---|---|
| Train | 577 | 483 | 94 |
| Test | 218 | 183 | 35 |
| Val | 193 | 159 | 34 |

### Forgery Characteristics

| Aspect | Details |
|---|---|
| **Forgery techniques** | CPI (Copy-Paste Imitation): 326, CUT: 42, IMI (Text Imitation): 35, PIX (Pixel Modification): 33, CPO: 10 |
| **Entity types targeted** | Total/payment: 289, Product: 138, Metadata: 129, Company: 37, Other: 29 |
| **Software used** | Paint: 63, GIMP: 62, Apercu: 15, Paint3D: 10, KolourPaint: 3 |
| **Avg forged regions/receipt** | 4.1 (range: 1-28) |
| **Annotation format** | VIA-style JSON with bounding boxes, entity types, technique labels |

### Image Properties

| Property | Min | Mean | Median | Max |
|---|---|---|---|---|
| Width (px) | 435 | 1,364 | 818 | 4,961 |
| Height (px) | 605 | 2,408 | 1,698 | 7,016 |
| File size (KB) | 32 | 1,048 | 568 | 68,105 |
| Aspect ratio | 0.26 | 0.52 | 0.50 | 0.97 |

### Receipt Content

- **Origin**: Malaysian receipts (RM currency) from SROIE
- **Text transcriptions**: Provided as `.txt` files alongside each `.png`
- **Avg text length**: 670 characters / 115 words per receipt
- **Monetary totals detected**: 574/988 receipts (~58%)
- **REAL mean total**: RM 60.77 | **FAKE mean total**: RM 74.63
- **Notable**: Some authentic receipts have handwritten/digital annotations that are NOT forgeries (50.6% have handwritten annotations)

### Key Insights for Approach Design

1. **Class imbalance** (5:1) must be considered in evaluation — accuracy alone is misleading
2. **Forgeries are subtle** — small bounding boxes targeting individual characters/digits
3. **CPI dominates** — most forgeries involve copying pixel patterns to alter amounts
4. **Total/payment is #1 target** — monetary fields are the primary forgery target
5. **Both visual and textual modalities** are available — enabling multi-modal analysis
6. **Image quality varies widely** — from 32KB to 68MB, grayscale to RGBA
7. **Handwritten annotations on authentic receipts** could confuse naive detectors

---

## 3. Solution Design & Architecture

### High-Level Architecture

```
                        ┌─────────────────────────┐
                        │     Receipt Image        │
                        │   + Text Transcription   │
                        └──────────┬──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
            ┌──────────┐  ┌──────────┐  ┌──────────┐
            │ Judge #1 │  │ Judge #2 │  │ Judge #3 │
            │ Visual   │  │ Content  │  │ Holistic │
            │ Forensics│  │ Integrity│  │ Auditor  │
            │ Expert   │  │ Analyst  │  │          │
            │(GPT-4o-m)│  │(Gemini-F)│  │(Claude-H)│
            └────┬─────┘  └────┬─────┘  └────┬─────┘
                 │              │              │
                 ▼              ▼              ▼
            ┌─────────┐  ┌─────────┐  ┌─────────┐
            │  JSON    │  │  JSON   │  │  JSON   │
            │ verdict  │  │ verdict │  │ verdict │
            └────┬─────┘  └────┬────┘  └────┬────┘
                 │              │             │
                 └──────────┬──┘─────────────┘
                            ▼
                   ┌────────────────┐
                   │ Majority Vote  │
                   │  Aggregator    │
                   └───────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │  Final Verdict  │
                  │ REAL/FAKE/UNCRT │
                  │ + confidence    │
                  │ + merged reasons│
                  └─────────────────┘
```

### The 3 LLM Judges

Each judge receives the **receipt image** (required) and optionally the **text transcription** as context. Each returns a structured JSON response.

#### Judge 1: Visual Forensics Expert

- **Default model**: OpenAI GPT-4o-mini
- **Persona**: Document forensics analyst specializing in visual tampering detection
- **Focus areas**:
  - Pixel-level inconsistencies and artifacts
  - Font mismatches within the receipt
  - Alignment and spacing anomalies
  - Copy-paste artifacts and irregular edges
  - Compression artifacts around suspected regions
  - Overall visual coherence

#### Judge 2: Content Integrity Analyst

- **Default model**: Google Gemini 2.5 Flash
- **Persona**: Financial auditor who checks receipts for mathematical and logical consistency
- **Focus areas**:
  - Mathematical consistency (items × price = subtotal, subtotals + tax = total)
  - Logical coherence of dates, times, and sequence numbers
  - Format consistency (currency, decimal places, alignment)
  - Tax calculation verification (GST rates, rounding)
  - Cross-field consistency (quantity × unit price checks)

#### Judge 3: Cross-Reference Auditor

- **Default model**: Anthropic Claude Haiku 4.5
- **Persona**: Fraud detection specialist evaluating overall document authenticity
- **Focus areas**:
  - Overall document presentation and layout quality
  - Consistency of business information
  - Receipt structure adherence to standard formats
  - Typography consistency throughout the document
  - Signs of digital editing or manipulation
  - Holistic trust assessment

### Structured Output Format

Each judge returns:
```json
{
  "label": "FAKE",
  "confidence": 78.5,
  "reasons": [
    "Font inconsistency detected in the total amount field",
    "Pixel artifacts visible around the modified digits",
    "Tax calculation does not match the stated total"
  ]
}
```

### Voting Logic

```python
# Majority voting algorithm:
# 1. Collect 3 verdicts
# 2. If 2+ judges agree on a label → that label wins
# 3. If all 3 disagree → UNCERTAIN
# 4. Final confidence = mean of agreeing judges' confidence
# 5. Reasons = merged from all judges (tagged by source)
```

**Edge cases handled:**
- All 3 agree → highest confidence verdict
- 2 agree, 1 disagrees → majority wins, confidence from agreeing judges only
- All 3 different → UNCERTAIN with mean confidence of all 3
- Any judge returns UNCERTAIN → treated as abstention in voting, falls back to remaining 2

### 20-Receipt Evaluation Set

- **Method**: Stratified random sampling from the **test split**
- **Composition**: 10 REAL + 10 FAKE receipts
- **Seed**: `random_state=42` (reproducible)
- **Selection**: Using `sklearn.model_selection.train_test_split` or `DataFrame.sample()` with documented seed

---

## 4. Brainstorming & Decision Log

### Approach Alternatives Considered

#### Option A: Multi-Provider LLM Judges (SELECTED)

Use 3 different LLM providers (OpenAI, Google, Anthropic) for maximum model diversity.

| Pros | Cons |
|---|---|
| True model diversity — different architectures, training data, and biases | Requires 3 separate API keys |
| Different strengths: GPT excels at vision, Gemini at speed/cost, Claude at reasoning | Higher configuration complexity |
| Reduces correlated failures — if one model has a blind spot, others may catch it | Dependency on 3 external services |
| Best representation of "wisdom of diverse crowds" | Slight latency variation between providers |

#### Option B: Single-Provider Multi-Persona

Use the same model (e.g., GPT-4o) with 3 different system prompts and temperatures.

| Pros | Cons |
|---|---|
| Single API key | Less true diversity (same model biases) |
| Simpler setup | Correlated failure modes |
| More predictable costs | Temperature variation alone may not diversify enough |
| Easier to debug | Doesn't satisfy the spirit of "different judges" |

#### Option C: Single-Provider Multi-Model

Use 2-3 models from one provider (e.g., GPT-4o + GPT-4o-mini + GPT-4.1).

| Pros | Cons |
|---|---|
| Single API key | Same provider biases |
| Model-level diversity within ecosystem | Less diverse than multi-provider |
| Consistent API interface | May have correlated weaknesses |

### Decision: Option A (Multi-Provider) with fallback to Option C

**Rationale**: The project specifically asks for judges that provide diverse perspectives. Using 3 different providers maximizes the diversity of analysis approaches and reduces correlated errors. However, the system is designed with a **configurable fallback**: if a user only has access to one provider, they can configure all 3 judges to use the same provider with different personas and temperatures.

### UI Technology Decision

| Option | Pros | Cons | Decision |
|---|---|---|---|
| **Streamlit** | Fastest to build, excellent for data apps, built-in caching, native image display, great Docker support | Limited customization | **SELECTED** |
| Flask | More flexible, RESTful | More boilerplate, need frontend separately | Rejected |
| Gradio | Good for ML demos | Less suitable for complex multi-page flows | Rejected |
| Next.js | Most customizable | Overkill, requires JS/TS, longer development | Rejected |

### Prompt Engineering Strategy

1. **Role-based prompting**: Each judge has a distinct expert persona
2. **Structured output enforcement**: Clear JSON schema in the prompt with examples
3. **Chain-of-thought**: Judges are asked to reason before concluding
4. **Grounding**: Judges receive both image and text data for multi-modal analysis
5. **Calibration instructions**: Explicit guidance on confidence score interpretation

---

## 5. Technology Stack

| Component | Technology | Justification |
|---|---|---|
| **Language** | Python 3.11+ | ML/AI ecosystem, LLM SDK support |
| **LLM - Judge 1** | OpenAI GPT-4o-mini | Strong vision capabilities, cost-effective |
| **LLM - Judge 2** | Google Gemini 2.5 Flash | Excellent price-performance, fast inference |
| **LLM - Judge 3** | Anthropic Claude Haiku 4.5 | Strong reasoning, safety-focused analysis |
| **UI Framework** | Streamlit | Rapid prototyping, native image/data display |
| **Data Processing** | pandas, NumPy | Standard data manipulation |
| **Visualization** | matplotlib, seaborn, Plotly | Charts and graphs |
| **Image Processing** | Pillow (PIL) | Image loading and manipulation |
| **Notebooks** | Jupyter | Dataset exploration and demo |
| **Containerization** | Docker + docker-compose | Portability and deployment |
| **Configuration** | python-dotenv | Environment-based config management |
| **Testing** | pytest | Unit and integration testing |

---

## 6. Project Structure

```
Receipt-Forgery-Detector/
│
├── README.md                          # This file — full project documentation
├── Statement and Instructions.pdf     # Original project requirements
├── docker-compose.yml                 # Docker Compose configuration
├── Dockerfile                         # Docker container definition
├── requirements.txt                   # Python dependencies
├── .env.example                       # Environment variables template
├── .gitignore                         # Git ignore rules
├── .dockerignore                      # Docker ignore rules
│
├── config/
│   └── settings.py                    # Centralized configuration management
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset loading, parsing, and sampling
│   ├── prompts.py                     # Judge prompt templates (3 personas)
│   ├── judges/
│   │   ├── __init__.py
│   │   ├── base_judge.py              # Abstract judge interface
│   │   ├── openai_judge.py            # OpenAI GPT judge implementation
│   │   ├── gemini_judge.py            # Google Gemini judge implementation
│   │   └── claude_judge.py            # Anthropic Claude judge implementation
│   ├── voting.py                      # Majority voting aggregation logic
│   └── evaluation.py                  # Metrics: accuracy, precision, recall, F1
│
├── app/
│   └── streamlit_app.py               # Streamlit interactive UI
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb    # Complete dataset analysis (EDA)
│   └── 02_demo_and_evaluation.ipynb   # Full pipeline demo + evaluation results
│
├── data/
│   └── findit2/findit2/               # Dataset (git-ignored, downloaded separately)
│       ├── train/                     # 577 receipts (483 REAL, 94 FAKE)
│       ├── test/                      # 218 receipts (183 REAL, 35 FAKE)
│       ├── val/                       # 193 receipts (159 REAL, 34 FAKE)
│       ├── train.txt                  # Train ground truth labels
│       ├── test.txt                   # Test ground truth labels
│       └── val.txt                    # Validation ground truth labels
│
├── outputs/                           # Generated visualizations and results
│   ├── class_distribution.png
│   ├── forgery_analysis.png
│   ├── image_properties.png
│   └── ...
│
└── tests/
    ├── test_data_loader.py
    ├── test_judges.py
    ├── test_voting.py
    └── test_evaluation.py
```

---

## 7. Eight-Day Development Plan

### Day 1: Foundation & Dataset Exploration (COMPLETED)

| Task | Status | Details |
|---|---|---|
| Read & analyze project statement | Done | Full understanding of requirements |
| Research dataset | Done | ICDAR 2023 paper, structure, annotations |
| Download dataset | Done | 658 MB, 988 receipts |
| Create exploration notebook | Done | `01_dataset_exploration.ipynb` with 8 analysis sections |
| Generate all EDA visualizations | Done | 8 PNG files in `/outputs/` |
| Design architecture | Done | Multi-judge + voting system |
| Create README with plan | Done | This document |

### Day 2: Core Infrastructure & Data Pipeline

| Task | Details |
|---|---|
| Set up `config/settings.py` | Environment-based config for API keys, model selection, paths |
| Implement `src/data_loader.py` | Load split CSVs, parse annotations, image loading utilities |
| Implement 20-receipt sampling | Stratified random sampling (10 REAL + 10 FAKE) from test set, seed=42 |
| Create `src/prompts.py` | 3 distinct judge prompts with persona descriptions |
| Set up `.env.example` | Template for all required environment variables |
| Create `requirements.txt` | Pin all dependencies |
| Create `.gitignore` | Exclude data/, .env, __pycache__, etc. |

### Day 3: LLM Judge Implementations

| Task | Details |
|---|---|
| Implement `src/judges/base_judge.py` | Abstract base class with `judge(image, text) -> JudgeVerdict` |
| Implement `src/judges/openai_judge.py` | GPT-4o-mini integration, vision API, structured output |
| Implement `src/judges/gemini_judge.py` | Gemini 2.5 Flash integration, multimodal API |
| Implement `src/judges/claude_judge.py` | Claude Haiku 4.5 integration, vision messages |
| JSON response parsing | Robust parsing with validation and fallback handling |
| Test individual judges | Verify each judge produces valid structured output |

### Day 4: Voting Engine & Evaluation Pipeline

| Task | Details |
|---|---|
| Implement `src/voting.py` | Majority voting with confidence aggregation |
| Implement `src/evaluation.py` | Accuracy, precision, recall, F1, confusion matrix |
| End-to-end pipeline | Wire judges → voting → evaluation |
| Test with 3-5 sample receipts | Verify full pipeline works correctly |
| Error handling & retry logic | Handle API failures gracefully, rate limiting |

### Day 5: Full Evaluation & Demo Notebook

| Task | Details |
|---|---|
| Run all 20 receipts through pipeline | Execute judges on full evaluation set |
| Create `02_demo_and_evaluation.ipynb` | Complete demo with step-by-step walkthrough |
| Per-judge performance analysis | Individual accuracy, agreement rates |
| Voting system performance | Final aggregated metrics |
| Results visualization | Confusion matrices, per-receipt breakdowns |
| Comparison: expected vs actual | Analysis of where the system succeeds/fails |

### Day 6: Streamlit Interactive UI

| Task | Details |
|---|---|
| Build `app/streamlit_app.py` | Multi-page Streamlit application |
| Page 1: Dataset Overview | Summary stats, class distribution, sample images |
| Page 2: Receipt Inspector | Select receipt → view image + text + metadata |
| Page 3: Judge Evaluation | Run 3 judges → display JSON verdicts → voting result |
| Page 4: Batch Evaluation | Run all 20 receipts, show metrics dashboard |
| Caching & performance | Use `@st.cache_resource` for heavy operations |

### Day 7: Docker Containerization

| Task | Details |
|---|---|
| Create `Dockerfile` | Python 3.11-slim base, multi-stage build |
| Create `docker-compose.yml` | Service definition with env vars and volumes |
| Create `.dockerignore` | Exclude unnecessary files from image |
| Test container build | `docker build -t receipt-forgery-detector .` |
| Test container run | `docker-compose up` → verify Streamlit UI works |
| Dataset volume mount | Mount data/ as volume for flexibility |
| Documentation | Docker setup instructions in README |

### Day 8: Review, Polish & Final Documentation

| Task | Details |
|---|---|
| Full end-to-end testing | Run complete pipeline in Docker |
| Polish demo notebook | Ensure all cells execute cleanly with outputs |
| Code cleanup | Remove debug code, add docstrings where needed |
| Final README update | Add actual results, setup instructions |
| Review evaluation metrics | Verify all metrics are correctly computed |
| Edge case handling | Test with edge cases (very large/small images) |
| Final commit & push | Tag release version |

---

## 8. Expected Results

### Baseline Expectations

Based on dataset characteristics and LLM capabilities for document analysis:

| Metric | Expected Range | Reasoning |
|---|---|---|
| **Overall accuracy** | 60-75% | LLMs are not specifically trained for forgery detection; forgeries are subtle pixel-level changes |
| **REAL precision** | 70-85% | LLMs should identify most authentic receipts correctly |
| **FAKE recall** | 45-65% | Detecting subtle forgeries via visual inspection alone is challenging |
| **FAKE precision** | 50-70% | Some false positives expected (authentic annotated receipts flagged as fake) |
| **Judge agreement rate** | 55-75% | Different models may focus on different aspects |
| **Voting improvement** | +5-10% over individual judges | Ensemble benefit from diverse perspectives |

### Why Moderate Performance is Expected

1. **Forgeries are extremely subtle**: Most modifications are single-digit pixel changes (bounding boxes of 8-15px)
2. **LLMs are not forensics tools**: They analyze visual patterns, not pixel-level anomalies
3. **No fine-tuning**: We use general-purpose LLMs, not forgery-specialized models
4. **Handwritten annotations on REAL receipts**: These may confuse LLMs into flagging authentic receipts
5. **Image quality variance**: From 32KB grayscale to 68MB RGBA images

### What Makes the System Valuable

Despite moderate accuracy on subtle forensic forgeries, the system demonstrates:
- **Multi-agent decision making** with structured output
- **Diverse reasoning approaches** (visual, content, holistic)
- **Transparent decision process** with explanations
- **Ensemble voting** that improves over individual judges
- **Practical architecture** that could be enhanced with fine-tuned models

---

## 9. Setup & Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- API keys for at least one of: OpenAI, Google AI (Gemini), Anthropic

### Local Development

```bash
# Clone the repository
git clone https://github.com/ajrojasfuentes/Receipt-Forgery-Detector.git
cd Receipt-Forgery-Detector

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys

# Download the dataset
mkdir -p data && cd data
wget https://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip
unzip findit2.zip
cd ..

# Run Streamlit app
streamlit run app/streamlit_app.py

# Or run the demo notebook
jupyter notebook notebooks/02_demo_and_evaluation.ipynb
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the app at http://localhost:8501
```

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-...          # For Judge 1 (GPT-4o-mini)
GOOGLE_API_KEY=AI...           # For Judge 2 (Gemini 2.5 Flash)
ANTHROPIC_API_KEY=sk-ant-...   # For Judge 3 (Claude Haiku 4.5)

# Optional: Override default model selections
JUDGE1_MODEL=gpt-4o-mini
JUDGE2_MODEL=gemini-2.5-flash
JUDGE3_MODEL=claude-haiku-4-5-20251001

# Optional: Override default temperatures
JUDGE1_TEMPERATURE=0.2
JUDGE2_TEMPERATURE=0.3
JUDGE3_TEMPERATURE=0.2
```

---

## 10. Usage

### Jupyter Notebooks

1. **`01_dataset_exploration.ipynb`**: Complete EDA with visualizations
   - Class distribution (REAL vs FAKE)
   - Forgery technique analysis
   - Image property analysis (sizes, aspect ratios, file sizes)
   - Receipt text content analysis (totals, word counts)
   - Sample receipt visualization with forgery bounding boxes

2. **`02_demo_and_evaluation.ipynb`**: Full pipeline demo (to be completed)
   - Load and display 20-receipt evaluation set
   - Run 3 judges on each receipt
   - Display individual judge decisions
   - Aggregated voting results
   - Evaluation metrics and analysis

### Streamlit App

Interactive web interface with pages for dataset overview, individual receipt inspection, judge evaluation, and batch evaluation with metrics dashboard.

### CLI

```bash
# Run evaluation on the 20-receipt sample
python -m src.evaluation --sample-size 20 --seed 42

# Run judges on a single receipt
python -m src.judges --image path/to/receipt.png
```

---

## 11. References

- Tornés, B.M. et al. (2023). *Receipt Dataset for Document Forgery Detection.* ICDAR 2023, LNCS vol 14189, Springer. [Paper](https://link.springer.com/chapter/10.1007/978-3-031-41682-8_28)
- [Dataset homepage](https://l3i-share.univ-lr.fr/2023Finditagain/index.html) — L3i Lab, University of La Rochelle
- [Dataset on Kaggle](https://www.kaggle.com/datasets/nikita2998/find-it-again-dataset)
- [Full thesis: Document Forgery Detection in Receipts](https://theses.hal.science/tel-05088756v1/file/2024MARTINEZTORNES223267.pdf) — Martínez Tornés (2024)
- [SROIE Dataset](https://rrc.cvc.uab.es/?ch=13) — Original receipt dataset (ICDAR 2019)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [Google Gemini API](https://ai.google.dev/docs)
- [Anthropic Claude API](https://docs.anthropic.com/en/docs/build-with-claude/vision)
- [Streamlit Docker Deployment](https://docs.streamlit.io/deploy/tutorials/docker)

---

*Project developed as part of an LLM-based document analysis challenge. Last updated: February 2026.*
