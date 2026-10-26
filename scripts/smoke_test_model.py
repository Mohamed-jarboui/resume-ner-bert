
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.predictor import ResumeNERPredictor

def run_smoke_test():
    model_path = "models/checkpoints/bert_crf/best_model.pt"
    mappings_path = "models/checkpoints/bert_crf/label_mappings.json"
    
    print("🚀 Initializing Predictor...")
    predictor = ResumeNERPredictor(
        model_path=model_path,
        label_mappings_path=mappings_path,
        model_name="bert-base-multilingual-cased"
    )
    
    test_text = """
    Karim Haddad
    karim.haddad@example.com
    Sousse, Tunisie
    
    Étudiant en Génie Logiciel à l’École Polytechnique.
    Data Scientist chez The Bridge (Août 2025).
    
    Compétences: Python, Machine Learning, TF-IDF, SQLite.
    """
    
    print("\n🔍 Running Prediction on Test Case...")
    entities = predictor.predict(test_text)
    structured = predictor.get_structured_json(test_text)
    
    print("\n✅ Extracted Entities:")
    for ent in entities:
        print(f"  [{ent['type']}] {ent['entity']} ({ent['start']}-{ent['end']})")
        
    print("\n📊 Structured JSON Output:")
    print(json.dumps(structured, indent=2, ensure_ascii=False))
    
    # Validation check for the user's specific request
    name_found = any(ent['type'] == 'NAME' and 'Karim' in ent['entity'] for ent in entities)
    email_found = any(ent['type'] == 'EMAIL' and 'karim.haddad' in ent['entity'] for ent in entities)
    skill_as_location = any(ent['type'] == 'SKILL' and ('Sousse' in ent['entity'] or 'Tunisie' in ent['entity']) for ent in entities)
    
    print("\n🎯 Validation Analysis:")
    print(f"  - Name Correctly Identified: {'✅' if name_found else '❌'}")
    print(f"  - Email Correctly Identified: {'✅' if email_found else '❌'}")
    print(f"  - Sousse/Tunisie NOT as skill: {'✅' if not skill_as_location else '❌'}")

if __name__ == "__main__":
    run_smoke_test()
