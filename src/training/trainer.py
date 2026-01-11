"""
Unified trainer for NER models with MLflow tracking.
"""

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from pathlib import Path
import mlflow
from typing import Optional, Dict


class NERTrainer:
    """Trainer for NER models with early stopping and MLflow logging."""
    
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config,
        metrics_calculator,
        save_dir: str = "models/checkpoints"
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
            metrics_calculator: NERMetrics instance
            save_dir: Directory to save checkpoints
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.metrics = metrics_calculator
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        total_steps = len(self.train_loader) * config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Early stopping
        self.best_metric = 0.0
        self.patience_counter = 0
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]
                total_loss += loss.item()
                
                # Get predictions
                if isinstance(outputs, dict) and 'tags' in outputs:
                    predictions = outputs['tags']
                else:
                    logits = outputs['logits'] if isinstance(outputs, dict) else outputs[1]
                    predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        metrics = self.metrics.compute_metrics(all_predictions, all_labels)
        metrics['val_loss'] = total_loss / len(self.val_loader)
        
        return metrics
    
    def train(self) -> Dict:
        """
        Full training loop with early stopping and MLflow logging.
        
        Returns:
            Training history
        """
        print(f"\nTraining on device: {self.device}")
        print(f"Number of training examples: {len(self.train_dataset)}")
        print(f"Number of validation examples: {len(self.val_dataset)}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Number of epochs: {self.config.num_epochs}\n")
        
        history = {'train_loss': [], 'val_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
        
        for epoch in range(self.config.num_epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Training loss: {train_loss:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate()
            print(f"Validation loss: {val_metrics['val_loss']:.4f}")
            print(f"Precision: {val_metrics['precision']:.4f}")
            print(f"Recall: {val_metrics['recall']:.4f}")
            print(f"F1 Score: {val_metrics['f1']:.4f}")
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_metrics['val_loss'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'val_f1': val_metrics['f1']
            }, step=epoch)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['val_loss'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_precision'].append(val_metrics['precision'])
            history['val_recall'].append(val_metrics['recall'])
            
            # Early stopping check
            current_metric = val_metrics['f1']
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                
                # Save best model
                model_path = self.save_dir / "best_model.pt"
                torch.save(self.model.state_dict(), model_path)
                print(f"✓ Saved best model (F1: {current_metric:.4f})")
            else:
                self.patience_counter += 1
                print(f"No improvement for {self.patience_counter} epoch(s)")
                
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best F1 Score: {self.best_metric:.4f}")
        print(f"{'='*60}\n")
        
        return history
