import sys
import json
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.predictor import ResumeNERPredictor

def test_hard_cases(model_path: str, mappings_path: str, data_path: str):
    print(f"Testing model: {Path(model_path).parent.name}")
    predictor = ResumeNERPredictor(
        model_path=model_path,
        label_mappings_path=mappings_path
    )
    
    with open(data_path, 'r', encoding='utf-8') as f:
        cases = json.load(f)
        
    print(f"\nEvaluating on {len(cases)} hard cases...\n")
    print("-" * 80)
    
    for i, case in enumerate(cases):
        text = " ".join(case['tokens'])
        print(f"CASE {i+1}: {text[:100]}...")
        
        # Predict
        entities = predictor.predict(text)
        
        # Simple comparison
        pred_types = [ent['type'] for ent in entities]
        predicted_unique = list(set(pred_types))
        
        print(f"PREDICTED ENTITIES: {predicted_unique}")
        for ent in entities:
            print(f"  - {ent['type']}: {ent['entity']}")
        print("-" * 40)

if __name__ == "__main__":
    baseline_path = project_root / "models/checkpoints/bert/final_model.pt"
    baseline_mappings = project_root / "models/checkpoints/bert/label_mappings.json"
    crf_path = project_root / "models/checkpoints/bert_crf/final_model.pt"
    crf_mappings = project_root / "models/checkpoints/bert_crf/label_mappings.json"
    data_path = project_root / "data/raw/manual_examples.json"
    
    if baseline_path.exists():
        print("\n=== BASELINE EVALUATION ===")
        test_hard_cases(str(baseline_path), str(baseline_mappings), str(data_path))
        
    if crf_path.exists():
        print("\n=== CRF EVALUATION ===")
        test_hard_cases(str(crf_path), str(crf_mappings), str(data_path))
    else:
        print("\n[INFO] CRF model not found. Still training?")
