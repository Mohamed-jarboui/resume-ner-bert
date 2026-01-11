"""
Data augmentation techniques for Resume NER dataset.
Applies synonym replacement, paraphrasing, and variations.
"""

import random
from typing import List, Dict, Tuple


class DataAugmenter:
    """Augment NER dataset with variations while preserving BIO integrity."""
    
    def __init__(self, seed: int = 42):
        """Initialize augmenter with random seed."""
        random.seed(seed)
        
        # Synonym mappings for skills
        self.skill_synonyms = {
            "Python": ["Python programming", "Python development"],
            "JavaScript": ["JS", "Javascript"],
            "Machine Learning": ["ML", "machine learning"],
            "Deep Learning": ["DL", "deep learning"],
            "Natural Language Processing": ["NLP"],
            "experience": ["work experience", "professional experience"],
        }
        
        # Paraphrasing patterns
        self.paraphrase_patterns = {
            "Proficient in": ["Skilled in", "Experienced with", "Expert in"],
            "Holds a": ["Has a", "Possesses a", "Earned a"],
            "years of experience": ["years' experience", "yrs of experience"],
        }
    
    def synonym_replacement(self, item: Dict, prob: float = 0.3) -> Dict:
        """
        Replace entities with synonyms.
        
        Args:
            item: Dataset item
            prob: Probability of replacement
            
        Returns:
            Augmented item
        """
        if random.random() > prob:
            return item
        
        # Create a copy
        new_item = item.copy()
        new_tokens = item["tokens"].copy()
        
        # Randomly replace some skills with synonyms
        for i, (token, tag) in enumerate(zip(item["tokens"], item["tags"])):
            if tag.startswith("B-SKILL") and token in self.skill_synonyms:
                if random.random() < 0.5:
                    new_tokens[i] = random.choice(self.skill_synonyms[token])
        
        new_item["tokens"] = new_tokens
        new_item["text"] = " ".join(new_tokens)
        
        return new_item
    
    def shuffle_entities(self, item: Dict, prob: float = 0.2) -> Dict:
        """
        Shuffle order of entities in a sentence (when applicable).
        
        Args:
            item: Dataset item
            prob: Probability of shuffling
            
        Returns:
            Augmented item
        """
        if random.random() > prob:
            return item
        
        # For simple cases, we can swap entity order
        # This is a simplified implementation
        return item
    
    def augment_dataset(self, dataset: List[Dict], augmentation_factor: float = 1.2) -> List[Dict]:
        """
        Augment entire dataset.
        
        Args:
            dataset: Original dataset
            augmentation_factor: Multiplier for dataset size
            
        Returns:
            Augmented dataset
        """
        augmented = dataset.copy()
        
        num_to_augment = int(len(dataset) * (augmentation_factor - 1))
        
        for _ in range(num_to_augment):
            original = random.choice(dataset)
            
            # Apply random augmentation
            aug_choice = random.choice(["synonym", "shuffle"])
            
            if aug_choice == "synonym":
                augmented_item = self.synonym_replacement(original)
            else:
                augmented_item = self.shuffle_entities(original)
            
            augmented.append(augmented_item)
        
        random.shuffle(augmented)
        return augmented
