"""
PDF text extraction utilities for resume parsing.
Supports both digital and scanned PDFs.
"""

import pdfplumber
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import re


class PDFTextExtractor:
    """Extract text from PDF resumes with cleanup."""
    
    def __init__(self, use_ocr: bool = False):
        """
        Initialize PDF extractor.
        
        Args:
            use_ocr: Whether to use OCR for scanned PDFs
        """
        self.use_ocr = use_ocr
        
        if use_ocr:
            try:
                import pytesseract
                from pdf2image import convert_from_path
                self.pytesseract = pytesseract
                self.convert_from_path = convert_from_path
            except ImportError:
                print("Warning: OCR libraries not available. Install pytesseract and pdf2image.")
                self.use_ocr = False
    
    def extract_with_pdfplumber(self, pdf_path: str) -> str:
        """
        Extract text using pdfplumber (best for digital PDFs).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text_parts = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        
        return "\n\n".join(text_parts)
    
    def extract_with_pymupdf(self, pdf_path: str) -> str:
        """
        Extract text using PyMuPDF (alternative method).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text_parts = []
        
        doc = fitz.open(pdf_path)
        for page in doc:
            page_text = page.get_text()
            if page_text:
                text_parts.append(page_text)
        doc.close()
        
        return "\n\n".join(text_parts)
    
    def extract_with_ocr(self, pdf_path: str) -> str:
        """
        Extract text using OCR (for scanned PDFs).
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        if not self.use_ocr:
            raise ValueError("OCR not enabled. Initialize with use_ocr=True")
        
        # Convert PDF to images
        images = self.convert_from_path(pdf_path)
        
        # OCR each page
        text_parts = []
        for image in images:
            text = self.pytesseract.image_to_string(image)
            if text.strip():
                text_parts.append(text)
        
        return "\n\n".join(text_parts)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove excessive whitespace
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'(?m)^\s*Page \d+\s*$', '', text)
        text = re.sub(r'(?m)^\s*\d+\s*$', '', text)
        
        # Normalize encoding
        text = text.encode('utf-8', 'ignore').decode('utf-8')
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def extract_text(self, pdf_path: str, method: str = "auto") -> str:
        """
        Extract text from PDF with automatic method selection.
        
        Args:
            pdf_path: Path to PDF file
            method: Extraction method ('auto', 'pdfplumber', 'pymupdf', 'ocr')
            
        Returns:
            Extracted and cleaned text
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Try pdfplumber first
        if method in ["auto", "pdfplumber"]:
            try:
                text = self.extract_with_pdfplumber(str(pdf_path))
                # Check if we got meaningful text
                if len(text.strip()) > 100:
                    return self.clean_text(text)
            except Exception as e:
                print(f"pdfplumber failed: {e}")
        
        # Try PyMuPDF as fallback
        if method in ["auto", "pymupdf"]:
            try:
                text = self.extract_with_pymupdf(str(pdf_path))
                if len(text.strip()) > 100:
                    return self.clean_text(text)
            except Exception as e:
                print(f"PyMuPDF failed: {e}")
        
        # Use OCR if enabled and other methods failed
        if method in ["auto", "ocr"] and self.use_ocr:
            try:
                text = self.extract_with_ocr(str(pdf_path))
                return self.clean_text(text)
            except Exception as e:
                print(f"OCR failed: {e}")
                raise
        
        raise ValueError("Could not extract text from PDF. Try enabling OCR.")
    
    def extract_metadata(self, pdf_path: str) -> Dict:
        """
        Extract PDF metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with metadata
        """
        doc = fitz.open(pdf_path)
        metadata = doc.metadata
        
        info = {
            'title': metadata.get('title', ''),
            'author': metadata.get('author', ''),
            'subject': metadata.get('subject', ''),
            'pages': doc.page_count,
            'producer': metadata.get('producer', ''),
            'creator': metadata.get('creator', '')
        }
        
        doc.close()
        return info


def segment_resume(text: str) -> Dict[str, str]:
    """
    Segment resume text into sections.
    
    Args:
        text: Full resume text
        
    Returns:
        Dictionary with sections
    """
    sections = {
        'summary': '',
        'experience': '',
        'education': '',
        'skills': '',
        'other': ''
    }
    
    # Common section headers
    patterns = {
        'summary': r'(?i)(summary|profile|about|objective)',
        'experience': r'(?i)(experience|employment|work history|professional experience)',
        'education': r'(?i)(education|academic|qualifications)',
        'skills': r'(?i)(skills|technical skills|competencies|expertise)'
    }
    
    # Simple section detection (can be enhanced)
    lines = text.split('\n')
    current_section = 'other'
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check if this is a section header
        for section, pattern in patterns.items():
            if re.search(pattern, line_lower) and len(line.strip()) < 50:
                current_section = section
                break
        else:
            # Add to current section
            sections[current_section] += line + '\n'
    
    return sections
