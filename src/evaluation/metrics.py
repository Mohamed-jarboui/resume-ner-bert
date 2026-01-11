"""
Evaluation metrics for NER.
Entity-level precision, recall, and F1 score using seqeval.
"""

from typing import List, Dict, Tuple
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from seqeval.scheme import IOB2
from sklearn.metrics import confusion_matrix
import numpy as np


class NERMetrics:
    """Calculate entity-level metrics for NER."""
    
    def __init__(self, id2label: Dict[int, str]):
        """
        Initialize metrics calculator.
        
        Args:
            id2label: Mapping from label ID to label string
        """
        self.id2label = id2label
    
    def align_predictions(
        self,
        predictions: np.ndarray,
        label_ids: np.ndarray
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Align predictions with labels, removing padding and special tokens.
        
        Args:
            predictions: Predicted label IDs (batch_size, seq_len)
            label_ids: True label IDs (batch_size, seq_len)
            
        Returns:
            Tuple of (true_labels, pred_labels) as lists of lists
        """
        true_labels = []
        pred_labels = []
        
        for prediction, label in zip(predictions, label_ids):
            true_labels_seq = []
            pred_labels_seq = []
            
            for pred_id, label_id in zip(prediction, label):
                # Skip padding and special tokens (label_id == -100)
                if label_id != -100:
                    true_labels_seq.append(self.id2label[label_id])
                    pred_labels_seq.append(self.id2label[pred_id])
            
            true_labels.append(true_labels_seq)
            pred_labels.append(pred_labels_seq)
        
        return true_labels, pred_labels
    
    def compute_metrics(
        self,
        predictions: np.ndarray,
        label_ids: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute entity-level metrics.
        
        Args:
            predictions: Predicted label IDs
            label_ids: True label IDs
            
        Returns:
            Dictionary with metrics
        """
        true_labels, pred_labels = self.align_predictions(predictions, label_ids)
        
        # Compute seqeval metrics (entity-level)
        precision = precision_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
        recall = recall_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
        f1 = f1_score(true_labels, pred_labels, mode='strict', scheme=IOB2)
        
        # Get classification report
        report = classification_report(
            true_labels, 
            pred_labels, 
            mode='strict', 
            scheme=IOB2,
            output_dict=True
        )
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }
        
        return metrics
    
    def get_confusion_matrix(
        self,
        predictions: np.ndarray,
        label_ids: np.ndarray
    ) -> np.ndarray:
        """
        Get confusion matrix for entity types.
        
        Args:
            predictions: Predicted label IDs
            label_ids: True label IDs
            
        Returns:
            Confusion matrix
        """
        true_labels, pred_labels = self.align_predictions(predictions, label_ids)
        
        # Flatten lists
        true_flat = [label for seq in true_labels for label in seq]
        pred_flat = [label for seq in pred_labels for label in seq]
        
        # Get unique labels
        labels = sorted(list(set(true_flat + pred_flat)))
        
        # Compute confusion matrix
        cm = confusion_matrix(true_flat, pred_flat, labels=labels)
        
        return cm, labels
    
    def print_classification_report(
        self,
        predictions: np.ndarray,
        label_ids: np.ndarray
    ):
        """
        Print detailed classification report.
        
        Args:
            predictions: Predicted label IDs
            label_ids: True label IDs
        """
        true_labels, pred_labels = self.align_predictions(predictions, label_ids)
        
        report = classification_report(
            true_labels,
            pred_labels,
            mode='strict',
            scheme=IOB2,
            digits=4
        )
        
        print("\n" + "="*60)
        print("Classification Report (Entity-Level)")
        print("="*60)
        print(report)
