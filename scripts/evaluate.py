"""
Evaluate the trained NER model with entity-level metrics (seqeval, strict IOB2).

By default it generates a fresh synthetic test set from the templates with a fixed
seed, so the numbers are reproducible. You can also pass your own annotated file.

Run from the repository root:
    python scripts/evaluate.py                       # fresh synthetic test set
    python scripts/evaluate.py --data data/raw/test.json
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data_generation.generator import ResumeNERDataGenerator
from src.evaluation.metrics import NERMetrics
from src.models.bert_model import BertCRFForNER, BertForNER
from src.preprocessing.dataset import NERDataset
from src.preprocessing.tokenizer import NERTokenizer


def build_test_file(n: int, seed: int, out_path: Path) -> Path:
    random.seed(seed)
    samples = ResumeNERDataGenerator().generate_samples(n)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Resume NER model")
    parser.add_argument("--model", default="models/checkpoints/bert_crf/best_model.pt")
    parser.add_argument("--mappings", default="models/checkpoints/bert_crf/label_mappings.json")
    parser.add_argument("--data", default=None, help="Annotated JSON file (tokens + tags)")
    parser.add_argument("--n", type=int, default=1000, help="Sentences to generate if --data is not given")
    parser.add_argument("--seed", type=int, default=2026, help="Seed for the generated test set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", default="results/test_metrics.json")
    args = parser.parse_args()

    with open(project_root / "config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model_name = config["model"]["model_name"]
    max_length = config["model"]["max_seq_length"]

    with open(project_root / args.mappings, encoding="utf-8") as f:
        mappings = json.load(f)
    label2id = mappings["label2id"]
    id2label = {int(k): v for k, v in mappings["id2label"].items()}

    # Test data
    if args.data:
        data_path = Path(args.data)
        source = str(data_path)
    else:
        data_path = build_test_file(args.n, args.seed, project_root / "results" / "generated_test.json")
        source = f"{args.n} generated sentences (seed {args.seed})"

    with open(data_path, encoding="utf-8") as f:
        raw = json.load(f)
    unique = len({" ".join(s["tokens"]) for s in raw})

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_crf = "crf" in args.model.lower()
    model_class = BertCRFForNER if use_crf else BertForNER
    model = model_class.from_pretrained(model_name, num_labels=len(id2label))
    state = torch.load(project_root / args.model, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device).eval()

    tokenizer = NERTokenizer(model_name=model_name, max_length=max_length)
    dataset = NERDataset(data_path, tokenizer, label2id, max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            if isinstance(outputs, dict) and "tags" in outputs:
                preds = outputs["tags"]
            else:
                preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].numpy())

    metrics = NERMetrics(id2label=id2label).compute_metrics(np.array(all_preds), np.array(all_labels))
    report = metrics["report"]
    entity_types = sorted(k for k in report if k.isupper())

    result = {
        "model": args.model,
        "test_data": source,
        "sentences": len(raw),
        "unique_sentences": unique,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "per_entity": {
            t: {
                "precision": report[t]["precision"],
                "recall": report[t]["recall"],
                "f1": report[t]["f1-score"],
                "support": int(report[t]["support"]),
            }
            for t in entity_types
        },
    }

    out = project_root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"\nTest data: {source} ({unique} unique)")
    print("\n| Entity | Precision | Recall | F1 | Support |")
    print("|---|---|---|---|---|")
    for t, m in result["per_entity"].items():
        print(f"| {t} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |")
    print(f"| **Overall** | {result['precision']:.3f} | {result['recall']:.3f} | {result['f1']:.3f} | |")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
