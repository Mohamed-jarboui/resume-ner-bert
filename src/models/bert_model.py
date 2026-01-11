"""
BERT-based model for Named Entity Recognition.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertConfig
from typing import Optional, Tuple, List
from torchcrf import CRF


class BertForNER(BertPreTrainedModel):
    """BERT model for token classification (NER)."""
    
    def __init__(self, config: BertConfig, **kwargs):
        """
        Initialize BERT NER model.
        
        Args:
            config: BERT configuration
            **kwargs: Backward compatibility for num_labels and dropout
        """
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", kwargs.get("num_labels"))
        
        # BERT encoder
        self.bert = BertModel(config)
        
        # Dropout and classifier
        dropout_prob = getattr(config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = kwargs.get("dropout", 0.1)
            
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Initialize weights
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Tuple:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Gold labels for training
            return_dict: Whether to return as dict
            
        Returns:
            Loss and logits
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get sequence output (last hidden state)
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout and classifier
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Calculate cross-entropy loss
            # Only calculate loss on non-padded tokens (labels != -100)
            loss_fct = nn.CrossEntropyLoss()
            
            # Flatten tensors
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            
            loss = loss_fct(active_logits, active_labels)
        
        if return_dict:
            return {
                'loss': loss,
                'logits': logits,
                'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
                'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
            }
        
        return (loss, logits) if loss is not None else logits


class BertCRFForNER(BertPreTrainedModel):
    """BERT model with CRF layer for NER."""
    
    def __init__(self, config: BertConfig, **kwargs):
        """
        Initialize BERT + CRF NER model.
        
        Args:
            config: BERT configuration
            **kwargs: Backward compatibility
        """
        super().__init__(config)
        self.num_labels = getattr(config, "num_labels", kwargs.get("num_labels"))
        
        # BERT encoder
        self.bert = BertModel(config)
        
        # Dropout
        dropout_prob = getattr(config, "classifier_dropout", None)
        if dropout_prob is None:
            dropout_prob = kwargs.get("dropout", 0.1)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Hidden to tag projection
        self.hidden2tag = nn.Linear(config.hidden_size, self.num_labels)
        
        # CRF layer
        self.crf = CRF(self.num_labels, batch_first=True)
        
        self.init_weights()
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Tuple:
        """Forward pass with CRF."""
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Project to tag space (emissions)
        emissions = self.hidden2tag(sequence_output)
        
        loss = None
        if labels is not None:
            # Mask preparation for CRF (1 for valid tokens, 0 for padding)
            crf_mask = attention_mask.bool()
            
            # Clean labels (set -100 to 0 temporarily as CRF uses mask anyway)
            labels_clean = labels.clone()
            labels_clean[labels == -100] = 0
            
            # Calculate negative log likelihood
            loss = -self.crf(emissions, labels_clean, mask=crf_mask, reduction='mean')
            
        # Decode best sequence
        tags_list = self.crf.decode(emissions, mask=attention_mask.bool())
        
        # Convert tags_list to a padded tensor for back-compatibility with argmax logic
        max_len = input_ids.shape[1]
        padded_tags = []
        for tag_seq in tags_list:
            padded_tags.append(tag_seq + [0] * (max_len - len(tag_seq)))
        
        # In CRF models, 'logits' will hold the decoded tag IDs for the trainer to use
        # but the trainer usually calls argmax(logits). To support this without changing
        # the trainer too much, we can return a one-hot-like tensor or just update trainer.
        # Let's return emissions as 'logits' so trainer can still use them for some things,
        # but we add 'tags' for the true CRF output.
        
        if return_dict:
            return {
                'loss': loss,
                'logits': emissions,
                'tags': torch.tensor(padded_tags).to(input_ids.device)
            }
        
        return (loss, emissions) if loss is not None else emissions
