"""
Resume NER Predictor supporting the upgraded 7-entity schema and post-processing.
"""

import torch
from pathlib import Path
import json
from transformers import AutoTokenizer
import sys
from typing import Dict, List, Optional, Tuple

# Standard cloud pathing
import os
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.bert_model import BertForNER, BertCRFForNER
from src.inference.post_processing import SkillFilter

class ResumeNERPredictor:
    """Predictor for Resume NER supporting both BERT and BERT+CRF."""
    
    def __init__(
        self,
        model_path: str,
        label_mappings_path: str,
        model_name: str = "bert-base-multilingual-cased",
        max_length: int = 128,
        use_crf: bool = False
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_crf = "crf" in model_path.lower() or use_crf
        
        # Load mappings
        with open(label_mappings_path, 'r') as f:
            mappings = json.load(f)
            self.id2label = {int(k): v for k, v in mappings['id2label'].items()}
            self.label2id = mappings['label2id']
            
        # Initialize model
        model_class = BertCRFForNER if self.use_crf else BertForNER
        self.model = model_class.from_pretrained(
            model_name,
            num_labels=len(self.id2label)
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        if hasattr(self.model, 'load_state_dict'):
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Post-processor
        self.skill_filter = SkillFilter()

    def predict(self, text: str):
        """Extract entities from raw text."""
        if not text or not text.strip():
            return []
            
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        offset_mapping = inputs["offset_mapping"][0]
        
        # Predict
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict) and 'tags' in outputs:
                # CRF returns decoded tags
                predictions = outputs['tags']
                if isinstance(predictions, torch.Tensor):
                    predictions = predictions[0]
                else: # list
                    predictions = torch.tensor(predictions[0])
            else:
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=2)[0]
            
        # Reconstruct entities
        entities = []
        current_entity = None
        
        # Convert tensors to list for easier processing
        input_ids_list = input_ids[0].tolist()
        preds_list = predictions.tolist()
        
        for i, (pred_idx, offsets) in enumerate(zip(preds_list, offset_mapping)):
            start, end = offsets
            if start == end:  # Special token
                continue
                
            label = self.id2label[pred_idx]
            
            if label.startswith("B-"):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    "type": label[2:],
                    "entity": text[start:end],
                    "start": int(start),
                    "end": int(end)
                }
            elif label.startswith("I-") and current_entity:
                # Continuous entity
                if label[2:] == current_entity["type"]:
                    current_entity["entity"] = text[current_entity["start"]:end]
                    current_entity["end"] = int(end)
                else:
                    # Type mismatch, end current
                    entities.append(current_entity)
                    current_entity = None
            else:
                if current_entity:
                    entities.append(current_entity)
                current_entity = None
                
        if current_entity:
            entities.append(current_entity)
            
        # Apply strict filtering
        final_entities = self.skill_filter.filter_entities(entities)
        
        return final_entities

    def predict_with_sections(self, text: str) -> Dict[str, List[Dict]]:
        """Extract entities grouped by resume sections."""
        from src.preprocessing.sections import segment_resume
        segments = segment_resume(text)
        results = {}
        
        for section, section_text in segments.items():
            if section_text.strip():
                entities = self.predict(section_text)
                if entities:
                    results[section] = entities
                    
        return results

    def get_structured_json(self, text: str):
        """Get highly structured JSON output."""
        entities = self.predict(text)
        return self.skill_filter.normalize_results(entities)
