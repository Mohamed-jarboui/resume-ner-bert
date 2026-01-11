"""
Complete PDF Resume Processing Pipeline for 7-Entity Multilingual Schema.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from src.inference.pdf_parser import PDFTextExtractor, segment_resume
from src.inference.predictor import ResumeNERPredictor
from src.inference.visualizer import EntityVisualizer
from src.inference.post_processing import SkillFilter

class PDFResumeProcessor:
    """End-to-end PDF resume processing with entity extraction."""
    
    def __init__(
        self,
        model_path: str,
        label_mappings_path: str,
        use_ocr: bool = False,
        model_name: str = "bert-base-multilingual-cased"
    ):
        # Initialize PDF extractor
        self.pdf_extractor = PDFTextExtractor(use_ocr=use_ocr)
        
        # Initialize NER predictor
        self.ner_predictor = ResumeNERPredictor(
            model_path=model_path,
            label_mappings_path=label_mappings_path,
            model_name=model_name
        )
        
        # Initialize post-processor
        self.skill_filter = SkillFilter()
        
        # Initialize visualizer
        self.visualizer = EntityVisualizer()
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Process PDF and extract 7 types of entities."""
        text = self.pdf_extractor.extract_text(pdf_path)
        
        if not text or len(text.strip()) < 50:
            raise ValueError("Insufficient text extracted from PDF")
            
        entities = self.ner_predictor.predict(text)
        sections = segment_resume(text)
        
        # Normalized and filtered results for structured JSON
        final_struct = self.skill_filter.normalize_results(entities)
        
        result = {
            'filename': Path(pdf_path).name,
            'extracted_text': text,
            'entities': entities,
            'structured_data': final_struct,
            'sections': sections
        }
        
        return result

    def create_highlighted_resume(self, result: Dict) -> str:
        """Create HTML with 7-entity highlighting."""
        viz_html = self.visualizer.get_html(result['extracted_text'], result['entities'])
        
        # Simple HTML wrapper
        html = f"""
        <html>
        <head>
            <title>Analysis - {result['filename']}</title>
            <style>
                body {{ font-family: sans-serif; padding: 40px; background: #f9f9f9; }}
                .container {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analysis Result: {result['filename']}</h1>
                {viz_html}
            </div>
        </body>
        </html>
        """
        return html
