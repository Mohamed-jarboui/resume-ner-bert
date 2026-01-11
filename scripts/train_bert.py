"""
Train BERT NER model on Resume dataset.
"""

import sys
import json
import yaml
from pathlib import Path
import torch
import mlflow
from transformers import BertConfig

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.preprocessing.tokenizer import NERTokenizer
from src.preprocessing.dataset import NERDataset, get_label_mappings
from src.models.bert_model import BertForNER, BertCRFForNER
from src.training.config import TrainingConfig, set_seed
from src.training.trainer import NERTrainer
from src.evaluation.metrics import NERMetrics
import argparse


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train BERT NER model")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_crf", action="store_true", help="Use CRF layer")
    parser.add_argument("--train_data", type=str, default="train.json", help="Training data filename")
    args = parser.parse_args()
    
    # Load configuration
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        yaml_config = yaml.safe_load(f)
    
    # Set up training config
    training_config = TrainingConfig()
    if args.num_epochs: training_config.num_epochs = args.num_epochs
    if args.batch_size: training_config.batch_size = args.batch_size
    if args.learning_rate: training_config.learning_rate = args.learning_rate
    if args.device: training_config.device = args.device
    if args.use_crf: training_config.use_crf = True
    
    set_seed(training_config.seed)
    
    print("=" * 60)
    print("BERT NER Training - Resume Keyword Extraction")
    print("=" * 60)
    print(f"Model: {training_config.model_name}")
    print(f"Use CRF: {training_config.use_crf}")
    print(f"Device: {training_config.device}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Epochs: {training_config.num_epochs}")
    print("=" * 60)
    
    # Get label mappings
    label2id, id2label = get_label_mappings(yaml_config)
    num_labels = len(label2id)
    
    # Initialize tokenizer
    ner_tokenizer = NERTokenizer(
        model_name=training_config.model_name,
        max_length=training_config.max_seq_length
    )
    
    # Load datasets
    data_dir = project_root / yaml_config['paths']['raw_data_dir']
    train_dataset = NERDataset(
        data_path=data_dir / args.train_data,
        tokenizer=ner_tokenizer,
        label2id=label2id,
        max_length=training_config.max_seq_length
    )
    val_dataset = NERDataset(
        data_path=data_dir / "val.json",
        tokenizer=ner_tokenizer,
        label2id=label2id,
        max_length=training_config.max_seq_length
    )
    
    # Initialize model
    print(f"\nInitializing {'BERT+CRF' if training_config.use_crf else 'BERT'} model...")
    bert_config = BertConfig.from_pretrained(training_config.model_name)
    bert_config.num_labels = num_labels # Set this for HF compatibility
    
    model_class = BertCRFForNER if training_config.use_crf else BertForNER
    model = model_class(
        config=bert_config,
        num_labels=num_labels,
        dropout=training_config.dropout
    )
    
    # Load pretrained BERT weights
    from transformers import BertModel
    pretrained_bert = BertModel.from_pretrained(training_config.model_name)
    model.bert = pretrained_bert
    
    # Initialize metrics
    metrics_calculator = NERMetrics(id2label=id2label)
    
    # Initialize trainer
    # Save mappings immediately
    checkpoint_dir = project_root / "models" / "checkpoints" / ("bert_crf" if training_config.use_crf else "bert")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    mappings_path = checkpoint_dir / "label_mappings.json"
    with open(mappings_path, 'w') as f:
        json.dump({'label2id': label2id, 'id2label': id2label}, f, indent=2)
    
    trainer = NERTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=training_config,
        metrics_calculator=metrics_calculator,
        save_dir=str(checkpoint_dir)
    )
    
    # Set up MLflow
    mlflow.set_experiment(training_config.experiment_name)
    
    with mlflow.start_run(run_name=f"{training_config.run_name or 'bert'}_crf={training_config.use_crf}"):
        mlflow.log_params({
            'use_crf': training_config.use_crf,
            'num_epochs': training_config.num_epochs,
            'batch_size': training_config.batch_size
        })
        
        history = trainer.train()
        
        final_model_path = checkpoint_dir / "final_model.pt"
        torch.save(model.state_dict(), final_model_path)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()
