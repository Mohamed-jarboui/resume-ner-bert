# Resume NER System - Smart Keyword Extraction Using BERT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project implements an **end-to-end Named Entity Recognition (NER) system** for automatically extracting key information from resumes using state-of-the-art BERT models. The system identifies and classifies three main entity types:

- **Skills** (technical, soft skills, tools)
- **Degrees** (education qualifications)
- **Experience** (work experience, roles)
- **Personal Info** (Names, Emails, Locations, Dates)

### Key Features

✅ **CRF Integration**: Conditional Random Field layer for sequence dependency logic  
✅ **Hybrid Data**: 9,000 samples blending synthetic data with real-world "Hard Cases"  
✅ **Section Intelligence**: Automatic segmentation of resumes into Experience, Education, etc.  
✅ **Human-in-the-Loop**: UI feedback mechanism to save and retrain on user corrections  
✅ **BERT Fine-Tuning**: bert-base-multilingual-cased for cross-language robustness  
✅ **Interactive UI**: Advanced Streamlit dashboard with entity highlighting and tabular views  
✅ **MLflow Tracking**: Complete experiment history and metric visualization

## 🏗️ Project Structure

```
resume-ner-system/
├── data/                          # Dataset files
│   ├── raw/                       # Generated annotated data (JSON, CSV, CoNLL)
│   ├── processed/                 # Tokenized data
│   └── statistics/                # Dataset analysis
├── models/checkpoints/            # Saved model weights
│   └── bert/
│       ├── best_model.pt
│       └── label_mappings.json
├── src/                           # Source code
│   ├── data_generation/           # Dataset generation
│   │   ├── templates.py
│   │   ├── generator.py
│   │   └── augmentation.py
│   ├── preprocessing/             # Data preprocessing
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── models/                    # Model architectures
│   │   ├── bert_model.py
│   │   └── bilstm_model.py
│   ├── training/                  # Training infrastructure
│   │   ├── config.py
│   │   └── trainer.py
│   ├── evaluation/                # Metrics and analysis
│   │   ├── metrics.py
│   │   └── analysis.py
│   └── inference/                 # Prediction and visualization
│       ├── predictor.py
│       └── visualizer.py
├── app/                           # Streamlit application
│   └── streamlit_app.py
├── scripts/                       # Executable scripts
│   ├── generate_dataset.py
│   ├── train_bert.py
│   └── train_baseline.py
├── notebooks/                     # Jupyter notebooks
├── config.yaml                    # Configuration file
├── requirements.txt               # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/resume-ner-system.git
cd resume-ner-system

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python scripts/generate_dataset.py
```

This will create 4,000 annotated resume sentences and save them in multiple formats.

### 3. Train BERT Model

```bash
python scripts/train_bert.py
```

Training takes approximately 15-20 minutes on GPU (or 1-2 hours on CPU). The best model will be saved to `models/checkpoints/bert/best_model.pt`.

### 4. Launch Interactive UI

```bash
streamlit run app/streamlit_app.py
```

The web interface will open at `http://localhost:8501`.

## 📊 Dataset

### Statistics

- **Total Sentences**: 4,000 (Train: 2,800 | Val: 600 | Test: 600)
- **Entity Distribution**: 
  - Skills: ~40%
  - Degrees: ~30%
  - Experience: ~30%
- **Format**: BIO tagging (B-SKILL, I-SKILL, B-DEGREE, I-DEGREE, B-EXPERIENCE, I-EXPERIENCE, O)
- **Exports**: JSON, CSV, CoNLL

### Sample Annotation

```
Tokens: ['Proficient', 'in', 'Python', ',', 'Java', ',', 'and', 'JavaScript', '.']
Tags:   ['O', 'O', 'B-SKILL', 'O', 'B-SKILL', 'O', 'O', 'B-SKILL', 'O']
```

## 🧠 Model Architecture

### BERT NER Model

- **Base Model**: `bert-base-uncased` (110M parameters)
- **Architecture**: BERT encoder + Token classification head
- **Training**: Fine-tuned with AdamW optimizer
- **Hyperparameters**:
  - Learning rate: 5e-5
  - Batch size: 16
  - Max epochs: 10
  - Early stopping patience: 3

### BiLSTM Baseline

- **Architecture**: Embedding layer + Bidirectional LSTM + Linear classifier
- **Embeddings**: 100-dimensional (can use GloVe pretrained)
- **Hidden size**: 256
- **Layers**: 2

## 📈 Results

### BERT Model Performance

| Metric     | Precision | Recall | F1 Score |
|------------|-----------|--------|----------|
| **SKILL**     | 0.88      | 0.85   | **0.86** |
| **DEGREE**    | 0.90      | 0.87   | **0.88** |
| **EXPERIENCE**| 0.85      | 0.83   | **0.84** |
| **Overall**   | 0.88      | 0.85   | **0.86** |

### Model Comparison

| Model  | F1 Score | Training Time | Parameters |
|--------|----------|---------------|------------|
| BiLSTM | 0.73     | ~5 min        | ~2M        |
| BERT   | **0.86** | ~15 min       | ~110M      |

**BERT achieves 18% improvement over BiLSTM baseline.**

## 🔬 Evaluation

### Entity-Level Metrics

The system uses **seqeval** for proper NER evaluation, which considers:
- Exact entity boundary matching
- Entity type correctness
- Partial matches as errors

### Error Analysis

Common errors include:
- **False Positives**: Generic words misclassified as skills
- **False Negatives**: Rare or abbreviated entities missed
- **Boundary Errors**: Incorrect entity span detection

## 💻 Usage Examples

### Python API

```python
from src.inference.predictor import ResumeNERPredictor

# Load model
predictor = ResumeNERPredictor(
    model_path="models/checkpoints/bert/best_model.pt",
    label_mappings_path="models/checkpoints/bert/label_mappings.json"
)

# Extract entities
text = "Experienced Python developer with Master's degree in Computer Science."
entities = predictor.predict(text)

for entity in entities:
    print(f"{entity['type']}: {entity['entity']}")
```

### Streamlit UI

The interactive interface provides:
- Resume text input
- Real-time entity extraction
- Color-coded visualization
- Entity summary table
- Example resumes

## 🛠️ Configuration

Edit `config.yaml` to customize:
- Dataset size and split ratios
- Model hyperparameters
- Training settings
- File paths

## 📚 Academic References

1. **BERT**: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
2. **NER Evaluation**: Tjong Kim Sang & De Meulder, "Introduction to the CoNLL-2003 Shared Task" (2003)
3. **BiLSTM-CRF**: Lample et al., "Neural Architectures for Named Entity Recognition" (2016)


---

**Developed for Advanced NLP Academic Evaluation.**
