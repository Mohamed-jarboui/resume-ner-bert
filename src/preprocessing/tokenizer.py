"""
Tokenization utilities for BERT NER.
Handles subword tokenization and label alignment.
"""

from typing import List, Tuple, Dict
from transformers import AutoTokenizer


class NERTokenizer:
    """Tokenizer for NER with BERT that handles subword alignment."""
    
    def __init__(self, model_name: str = "bert-base-multilingual-cased", max_length: int = 128):
        """
        Initialize the tokenizer.
        
        Args:
            model_name: BERT model name for tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def align_labels_with_tokens(
        self, 
        labels: List[str], 
        word_ids: List[int]
    ) -> List[int]:
        """
        Align labels with BERT subword tokens.
        
        Strategy:
        - First subword of a word gets the label
        - Subsequent subwords get -100 (ignored by PyTorch loss)
        - Special tokens get -100
        
        Args:
            labels: List of BIO tags
            word_ids: Word IDs from tokenizer
            
        Returns:
            List of aligned label IDs
        """
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            # Special tokens have word_id None
            if word_idx is None:
                aligned_labels.append(-100)
            # First subword of a word
            elif word_idx != previous_word_idx:
                aligned_labels.append(labels[word_idx])
            # Subsequent subwords of the same word
            else:
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        return aligned_labels
    
    def tokenize_and_align(
        self, 
        tokens: List[str], 
        tags: List[str],
        label2id: Dict[str, int]
    ) -> Dict:
        """
        Tokenize tokens and align labels.
        
        Args:
            tokens: List of tokens (words)
            tags: List of BIO tags
            label2id: Mapping from tag to ID
            
        Returns:
            Dictionary with tokenized inputs and aligned labels
        """
        # Convert tags to IDs
        label_ids = [label2id[tag] for tag in tags]
        
        # Tokenize with is_split_into_words=True since we have pre-tokenized data
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors=None  # Return lists, not tensors
        )
        
        # Get word IDs for alignment
        word_ids = tokenized_inputs.word_ids()
        
        # Align labels
        aligned_labels = self.align_labels_with_tokens(label_ids, word_ids)
        
        # Pad labels to max_length
        padding_length = self.max_length - len(aligned_labels)
        aligned_labels = aligned_labels + [-100] * padding_length
        
        tokenized_inputs['labels'] = aligned_labels
        
        return tokenized_inputs
    
    def decode_predictions(
        self, 
        predictions: List[int], 
        id2label: Dict[int, str],
        input_ids: List[int]
    ) -> List[str]:
        """
        Decode predictions back to labels, handling subwords.
        
        Args:
            predictions: List of predicted label IDs
            id2label: Mapping from ID to label
            input_ids: Original input token IDs
            
        Returns:
            List of predicted labels
        """
        # Get word IDs for this sequence
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        decoded_labels = []
        for pred_id, token in zip(predictions, tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            if pred_id != -100:
                decoded_labels.append(id2label[pred_id])
        
        return decoded_labels
