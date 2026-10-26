"""
Command-line script for processing PDF resumes.
Usage: python scripts/process_pdf.py resume.pdf
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference.pdf_processor import PDFResumeProcessor


def main():
    """Main function for PDF processing."""
    
    parser = argparse.ArgumentParser(
        description="Extract entities from PDF resumes using trained BERT NER model"
    )
    
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF resume file or directory containing PDFs"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=str(project_root / "models/checkpoints/bert_crf/best_model.pt"),
        help="Path to trained model (default: models/checkpoints/bert_crf/best_model.pt)"
    )
    
    parser.add_argument(
        "--mappings",
        type=str,
        default=str(project_root / "models/checkpoints/bert_crf/label_mappings.json"),
        help="Path to label mappings (default: models/checkpoints/bert_crf/label_mappings.json)"
    )
    
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for scanned PDFs"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for results (default: output)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "txt", "html", "all"],
        default="all",
        help="Output format (default: all)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all PDFs in directory (if pdf_path is a directory)"
    )
    
    args = parser.parse_args()
    
    # Check if model exists
    if not Path(args.model).exists():
        print("❌ Error: Model not found!")
        print(f"   Expected: {args.model}")
        print("\n💡 Train the model first:")
        print("   python scripts/train_bert.py")
        sys.exit(1)
    
    print("=" * 60)
    print("📄 PDF Resume Entity Extraction")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"OCR Enabled: {args.ocr}")
    print(f"Output: {args.output}")
    print("=" * 60)
    
    # Initialize processor
    try:
        processor = PDFResumeProcessor(
            model_path=args.model,
            label_mappings_path=args.mappings,
            use_ocr=args.ocr
        )
    except Exception as e:
        print(f"❌ Error initializing processor: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Determine if processing single file or batch
    pdf_path = Path(args.pdf_path)
    
    if not pdf_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    # Batch processing
    if pdf_path.is_dir() or args.batch:
        if pdf_path.is_file():
            print("❌ Error: --batch flag requires a directory path")
            sys.exit(1)
        
        pdf_files = list(pdf_path.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ No PDF files found in: {pdf_path}")
            sys.exit(1)
        
        print(f"\n📁 Found {len(pdf_files)} PDF files\n")
        
        results = processor.process_batch(
            pdf_paths=[str(p) for p in pdf_files],
            output_dir=str(output_dir)
        )
        
        # Summary
        successful = sum(1 for r in results if 'error' not in r)
        print(f"\n{'='*60}")
        print(f"✓ Processed {successful}/{len(pdf_files)} files successfully")
        print(f"📁 Results saved to: {output_dir}")
        print(f"{'='*60}")
    
    # Single file processing
    else:
        try:
            print(f"\n📄 Processing: {pdf_path.name}\n")
            
            result = processor.process_pdf(str(pdf_path))
            
            # Export based on format
            base_name = pdf_path.stem
            
            if args.format in ["json", "all"]:
                processor.export_to_json(result, output_dir / f"{base_name}_entities.json")
            
            if args.format in ["csv", "all"]:
                processor.export_to_csv(result, output_dir / f"{base_name}_entities.csv")
            
            if args.format in ["txt", "all"]:
                processor.export_summary(result, output_dir / f"{base_name}_summary.txt")
            
            if args.format in ["html", "all"]:
                html = processor.create_highlighted_resume(result)
                html_path = output_dir / f"{base_name}_highlighted.html"
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                print(f"✓ Saved HTML to: {html_path}")
            
            # Print summary
            print(f"\n{'='*60}")
            print("📊 Extraction Summary")
            print(f"{'='*60}")
            
            for entity_type in ['SKILL', 'DEGREE', 'EXPERIENCE']:
                entities = result['grouped_entities'].get(entity_type, [])
                if entities:
                    print(f"\n{entity_type}S ({len(entities)}):")
                    for entity in entities[:10]:  # Show first 10
                        print(f"  • {entity}")
                    if len(entities) > 10:
                        print(f"  ... and {len(entities) - 10} more")
            
            print(f"\n{'='*60}")
            print(f"✓ Results saved to: {output_dir}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"\n❌ Error processing PDF: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
