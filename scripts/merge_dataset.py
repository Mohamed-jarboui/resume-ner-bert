import json
from pathlib import Path
import random

def merge_datasets(synthetic_path: str, manual_path: str, output_path: str, augment_factor: int = 50):
    """Merge synthetic and manual datasets with oversampling of manual cases."""
    
    with open(synthetic_path, 'r', encoding='utf-8') as f:
        synthetic = json.load(f)
        
    with open(manual_path, 'r', encoding='utf-8') as f:
        manual = json.load(f)
        
    print(f"Loaded {len(synthetic)} synthetic samples")
    print(f"Loaded {len(manual)} manual samples")
    
    # Oversample manual samples to ensure they have visibility
    augmented_manual = manual * augment_factor
    random.shuffle(augmented_manual)
    
    combined = synthetic + augmented_manual
    random.shuffle(combined)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
        
    print(f"Combined dataset saved to {output_path} ({len(combined)} total samples)")

if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    merge_datasets(
        str(project_root / "data/raw/train.json"),
        str(project_root / "data/raw/manual_examples.json"),
        str(project_root / "data/raw/train_hybrid.json"),
        augment_factor=200
    )
