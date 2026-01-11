"""
Generate the complete Resume NER dataset.
Creates train, validation, and test splits with statistics.
"""

import os
import sys
import json
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data_generation.generator import ResumeNERDataGenerator
from src.data_generation.augmentation import DataAugmenter


def main():
    """Generate and save the complete NER dataset."""
    
    # Load configuration
    config_path = project_root / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Configuration
    dataset_cfg = config["dataset"]
    total_sentences = dataset_cfg["total_sentences"]
    ratios = dataset_cfg["split_ratios"]
    train_ratio = ratios["train"]
    val_ratio = ratios["val"]
    test_ratio = ratios["test"]
    seed = dataset_cfg["random_seed"]
    
    # Create output directories
    raw_dir = project_root / config["paths"]["raw_data_dir"]
    stats_dir = project_root / "data/statistics" # Using default or config
    raw_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Resume NER Dataset Generation")
    print("=" * 60)
    print(f"Total sentences: {total_sentences}")
    print(f"Random seed: {seed}")
    print()
    
    # Initialize generator
    generator = ResumeNERDataGenerator()
    
    # Generate dataset
    print("Generating dataset...")
    dataset = generator.generate_samples(total_sentences)
    print(f"✓ Generated {len(dataset)} sentences")
    
    # Split dataset
    print("\nSplitting dataset...")
    train_size = int(len(dataset) * train_ratio)
    val_size = int(len(dataset) * val_ratio)
    
    train_data = dataset[:train_size]
    val_data = dataset[train_size:train_size + val_size]
    test_data = dataset[train_size + val_size:]
    
    print(f"✓ Train: {len(train_data)} sentences")
    print(f"✓ Validation: {len(val_data)} sentences")
    print(f"✓ Test: {len(test_data)} sentences")
    
    # Save datasets in multiple formats
    print("\nSaving datasets...")
    
    # JSON format
    generator.save_to_json(train_data, raw_dir / "train.json")
    generator.save_to_json(val_data, raw_dir / "val.json")
    generator.save_to_json(test_data, raw_dir / "test.json")
    print("✓ Saved JSON files")
    
    # CSV format
    generator.save_to_csv(train_data, raw_dir / "train.csv")
    generator.save_to_csv(val_data, raw_dir / "val.csv")
    generator.save_to_csv(test_data, raw_dir / "test.csv")
    print("✓ Saved CSV files")
    
    # CoNLL format
    generator.save_to_conll(train_data, raw_dir / "train.conll")
    generator.save_to_conll(val_data, raw_dir / "val.conll")
    generator.save_to_conll(test_data, raw_dir / "test.conll")
    print("✓ Saved CoNLL files")
    
    print("\n" + "=" * 60)
    print("Dataset generation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
