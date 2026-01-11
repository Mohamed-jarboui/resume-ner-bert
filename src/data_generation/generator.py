import random
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from .templates import TEMPLATES, VARIABLE_MAP, VARIABLE_TAGS

class ResumeNERDataGenerator:
    """Enhanced synthetic data generator for Resume NER (7 entities)."""
    
    def __init__(self, templates=TEMPLATES, variable_map=VARIABLE_MAP, variable_tags=VARIABLE_TAGS):
        self.templates = templates
        self.variable_map = variable_map
        self.variable_tags = variable_tags

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization preserving punctuation."""
        return re.findall(r"[\w@\.-]+|[^\w\s]", text)

    def generate_sentence(self) -> Tuple[List[str], List[str]]:
        """Generate a single annotated sentence."""
        category = random.choice(list(self.templates.keys()))
        template = random.choice(self.templates[category])
        
        # Split template by placeholders like {name}, {skill1}
        parts = re.split(r"(\{[\w]+\})", template)
        
        final_tokens = []
        final_tags = []
        
        for part in parts:
            if not part:
                continue
            
            match = re.match(r"\{(\w+)\}", part)
            if match:
                var_name = match.group(1)
                val = random.choice(self.variable_map[var_name])
                val_tokens = self._tokenize(val)
                tag_base = self.variable_tags[var_name]
                
                if tag_base == "O":
                    for t in val_tokens:
                        final_tokens.append(t)
                        final_tags.append("O")
                else:
                    for i, t in enumerate(val_tokens):
                        final_tokens.append(t)
                        if i == 0:
                            final_tags.append(f"B-{tag_base}")
                        else:
                            final_tags.append(f"I-{tag_base}")
            else:
                # Regular text
                text_tokens = self._tokenize(part)
                for t in text_tokens:
                    final_tokens.append(t)
                    final_tags.append("O")
                    
        return final_tokens, final_tags

    def generate_samples(self, n: int) -> List[Dict]:
        """Generate n samples."""
        samples = []
        for _ in tqdm(range(n), desc="Generating samples"):
            tokens, tags = self.generate_sentence()
            samples.append({
                "tokens": tokens,
                "tags": tags
            })
        return samples

    def save_to_json(self, samples: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

    def save_to_csv(self, samples: List[Dict], output_path: str):
        import pandas as pd
        rows = []
        for i, sample in enumerate(samples):
            for token, tag in zip(sample['tokens'], sample['tags']):
                rows.append({"sentence_id": i, "token": token, "tag": tag})
        pd.DataFrame(rows).to_csv(output_path, index=False)

    def save_to_conll(self, samples: List[Dict], output_path: str):
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                for token, tag in zip(sample['tokens'], sample['tags']):
                    f.write(f"{token}\t{tag}\n")
                f.write("\n")
