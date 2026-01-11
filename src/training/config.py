"""
Training configuration.
"""

import torch
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""
    
    # Model
    model_name: str = "bert-base-multilingual-cased"
    max_seq_length: int = 128
    use_crf: bool = False
    dropout: float = 0.1
    
    # Training
    batch_size: int = 16
    learning_rate: float = 5e-5
    num_epochs: int = 10
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_metric: str = "f1"
    
    # Checkpointing
    save_total_limit: int = 3
    save_strategy: str = "epoch"  # 'epoch' or 'steps'
    
    # Logging
    logging_steps: int = 50
    eval_steps: int = 100
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed
    seed: int = 42
    
    # MLflow
    experiment_name: str = "Resume_NER_BERT"
    run_name: Optional[str] = None


@dataclass
class BiLSTMConfig:
    """BiLSTM baseline configuration."""
    
    # Model
    embedding_dim: int = 100
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    use_glove: bool = False
    
    # Training
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 20
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 5.0
    
    # Early stopping
    early_stopping_patience: int = 5
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Random seed
    seed: int = 42
    
    # MLflow
    experiment_name: str = "Resume_NER_BiLSTM"
    run_name: Optional[str] = None


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
