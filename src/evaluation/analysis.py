"""
Error analysis utilities for NER predictions.
"""

from typing import List, Dict, Tuple
import pandas as pd


class ErrorAnalyzer:
    """Analyze errors in NER predictions."""
    
    def __init__(self, id2label: Dict[int, str]):
        """
        Initialize error analyzer.
        
        Args:
            id2label: Mapping from label ID to label string
        """
        self.id2label = id2label
    
    def analyze_errors(
        self,
        dataset,
        predictions: List[List[int]],
        limit: int = 20
    ) -> Dict:
        """
        Analyze prediction errors.
        
        Args:
            dataset: NERDataset instance
            predictions: List of predicted label sequences
            limit: Number of examples to analyze
            
        Returns:
            Dictionary with error analysis
        """
        errors = {
            'false_positives': [],
            'false_negatives': [],
            'correct_predictions': []
        }
        
        for idx in range(min(len(dataset), limit)):
            raw_item = dataset.get_raw_item(idx)
            pred_labels = [self.id2label[p] for p in predictions[idx]]
            true_labels = raw_item['tags']
            tokens = raw_item['tokens']
            
            # Find mismatches
            for i, (token, true_label, pred_label) in enumerate(zip(tokens, true_labels, pred_labels)):
                if true_label != pred_label:
                    if true_label == 'O' and pred_label != 'O':
                        # False positive
                        errors['false_positives'].append({
                            'token': token,
                            'context': ' '.join(tokens[max(0, i-3):min(len(tokens), i+4)]),
                            'predicted': pred_label,
                            'true': true_label
                        })
                    elif true_label != 'O' and pred_label == 'O':
                        # False negative
                        errors['false_negatives'].append({
                            'token': token,
                            'context': ' '.join(tokens[max(0, i-3):min(len(tokens), i+4)]),
                            'predicted': pred_label,
                            'true': true_label
                        })
                else:
                    if true_label != 'O':
                        # Correct entity prediction
                        errors['correct_predictions'].append({
                            'token': token,
                            'label': true_label
                        })
        
        return errors
    
    def print_error_summary(self, errors: Dict):
        """
        Print error analysis summary.
        
        Args:
            errors: Error dictionary from analyze_errors
        """
        print("\n" + "="*60)
        print("Error Analysis Summary")
        print("="*60)
        
        print(f"\nFalse Positives: {len(errors['false_positives'])}")
        if errors['false_positives']:
            print("\nSample False Positives:")
            for fp in errors['false_positives'][:5]:
                print(f"  Token: '{fp['token']}' | Predicted: {fp['predicted']} | True: {fp['true']}")
                print(f"  Context: {fp['context']}")
                print()
        
        print(f"\nFalse Negatives: {len(errors['false_negatives'])}")
        if errors['false_negatives']:
            print("\nSample False Negatives:")
            for fn in errors['false_negatives'][:5]:
                print(f"  Token: '{fn['token']}' | Predicted: {fn['predicted']} | True: {fn['true']}")
                print(f"  Context: {fn['context']}")
                print()
        
        print(f"\nCorrect Entity Predictions: {len(errors['correct_predictions'])}")
