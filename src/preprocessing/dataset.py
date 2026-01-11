"""
PyTorch Dataset classes for NER.
"""

import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple
from pathlib import Path


class NERDataset(Dataset):
    """PyTorch Dataset for Resume NER."""
    
    def __init__(
        self, 
        data_path: str, 
        tokenizer,
        label2id: Dict[str, int],
        max_length: int = 128
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to JSON data file
            tokenizer: NERTokenizer instance
            label2id: Label to ID mapping
            max_length: Maximum sequence length
        """
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.raw_data = json.load(f)
        
        # Pre-tokenize all data
        self.tokenized_data = []
        for item in self.raw_data:
            tokenized = self.tokenizer.tokenize_and_align(
                item['tokens'],
                item['tags'],
                self.label2id
            )
            self.tokenized_data.append(tokenized)
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        """Get a single tokenized example."""
        item = self.tokenized_data[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
            'labels': torch.tensor(item['labels'], dtype=torch.long)
        }
    
    def get_raw_item(self, idx):
        """Get the original raw data item."""
        return self.raw_data[idx]


def get_label_mappings(config: Dict) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Create label to ID and ID to label mappings.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (label2id, id2label) dictionaries
    """
    label2id = config['labels']['label2id']
    id2label = {v: k for k, v in label2id.items()}
    
    return label2id, id2label
