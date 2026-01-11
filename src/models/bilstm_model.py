"""
BiLSTM baseline model for NER.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class BiLSTMForNER(nn.Module):
    """BiLSTM with embeddings for NER baseline."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_labels: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings: Optional[torch.Tensor] = None
    ):
        """
        Initialize BiLSTM NER model.
        
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Embedding dimension
            hidden_dim: Hidden state dimension
            num_labels: Number of NER labels
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embeddings (e.g., GloVe)
        """
        super().__init__()
        
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 for bidirectional
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Hidden to tag
        self.hidden2tag = nn.Linear(hidden_dim, num_labels)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple:
        """
        Forward pass.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Gold labels for training
            
        Returns:
            Loss and logits
        """
        # Embedding lookup
        embeddings = self.embedding(input_ids)  # (batch, seq_len, emb_dim)
        
        # Pack padded sequence for efficiency (if using attention mask)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                embeddings, 
                lengths, 
                batch_first=True, 
                enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embeddings)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Project to tag space
        logits = self.hidden2tag(lstm_out)  # (batch, seq_len, num_labels)
        
        loss = None
        if labels is not None:
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                logits.view(-1, self.num_labels),
                labels.view(-1)
            )
        
        return {'loss': loss, 'logits': logits} if loss is not None else {'logits': logits}
    
    def predict(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Generate predictions.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Predicted label IDs
        """
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            logits = outputs['logits']
            predictions = torch.argmax(logits, dim=-1)
        
        return predictions
