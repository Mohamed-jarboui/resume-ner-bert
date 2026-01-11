# Project Usage Guide

## Quick Start Commands

### 1. Generate Dataset
```bash
python scripts/generate_dataset.py
```

### 2. Train BERT Model
```bash
python scripts/train_bert.py
```

### 3. Launch UI
```bash
streamlit run app/streamlit_app.py
```

## Training Tips

- **GPU Recommended**: Training on GPU takes ~15 min vs 1-2 hours on CPU
- **Early Stopping**: Model will stop if no improvement for 3 epochs
- **MLflow**: Track experiments with `mlflow ui` to view training logs
- **Checkpoints**: Best model saved to `models/checkpoints/bert/best_model.pt`

## Customization

### Modify Dataset Size
Edit `config.yaml`:
```yaml
dataset:
  total_sentences: 4000  # Change this
```

### Adjust Training Parameters
Edit `config.yaml`:
```yaml
training:
  batch_size: 16
  learning_rate: 5e-5
  num_epochs: 10
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in `config.yaml` or `src/training/config.py`

### Model Not Loading in UI
Ensure you've trained the model first with `python scripts/train_bert.py`

### Import Errors
Make sure you're running from the project root directory
