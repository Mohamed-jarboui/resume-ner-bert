# PDF Resume Processing Guide

## 📄 Overview

The system now supports **direct PDF resume processing** with automatic text extraction and entity recognition. Process resumes from PDF files using multiple methods:

1. **Command-Line Interface** - Batch processing
2. **Streamlit Web UI** - Interactive upload
3. **Python API** - Programmatic access

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Install OCR (Optional - for scanned PDFs)

**Windows:**
```bash
# Download and install Tesseract OCR
# https://github.com/UB-Mannheim/tesseract/wiki
```

**macOS:**
```bash
brew install tesseract
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

---

## 💻 Usage Methods

### Method 1: Command-Line Interface

**Single PDF:**
```bash
python scripts/process_pdf.py resume.pdf
```

**With custom output directory:**
```bash
python scripts/process_pdf.py resume.pdf --output results
```

**Batch processing (all PDFs in directory):**
```bash
python scripts/process_pdf.py resumes/ --batch
```

**Enable OCR for scanned PDFs:**
```bash
python scripts/process_pdf.py scanned_resume.pdf --ocr
```

**Select specific output format:**
```bash
python scripts/process_pdf.py resume.pdf --format json
# Options: json, csv, txt, html, all
```

**Full example:**
```bash
python scripts/process_pdf.py candidate_resume.pdf \
    --output results/candidate1 \
    --format html \
    --ocr
```

---

### Method 2: Streamlit Web UI

**Launch the UI:**
```bash
streamlit run app/streamlit_app.py
```

**Steps:**
1. Click "Load Model" in sidebar
2. Select "PDF Upload" input mode
3. Upload your PDF resume
4. Click "Extract from PDF"
5. View results and download in JSON/CSV/HTML

**Features:**
- 📤 PDF upload widget
- 📄 Extracted text preview
- 🔍 Real-time entity extraction
- 💾 Download results in multiple formats
- 🎨 Highlighted resume visualization

---

### Method 3: Python API

```python
from src.inference.pdf_processor import PDFResumeProcessor

# Initialize processor
processor = PDFResumeProcessor(
    model_path="models/checkpoints/bert/best_model.pt",
    label_mappings_path="models/checkpoints/bert/label_mappings.json",
    use_ocr=False  # Set to True for scanned PDFs
)

# Process a PDF
result = processor.process_pdf("resume.pdf")

# Access results
print(f"Found {len(result['entities'])} entities")
print("Skills:", result['grouped_entities'].get('SKILL', []))
print("Degrees:", result['grouped_entities'].get('DEGREE', []))
print("Experience:", result['grouped_entities'].get('EXPERIENCE', []))

# Export results
processor.export_to_json(result, "output/entities.json")
processor.export_to_csv(result, "output/entities.csv")
processor.export_summary(result, "output/summary.txt")

# Create highlighted HTML
html = processor.create_highlighted_resume(result)
with open("output/highlighted.html", "w") as f:
    f.write(html)

# Batch processing
results = processor.process_batch(
    pdf_paths=["resume1.pdf", "resume2.pdf", "resume3.pdf"],
    output_dir="batch_results"
)
```

---

## 📊 Output Formats

### 1. JSON Format
```json
{
  "filename": "john_doe_resume.pdf",
  "num_entities": 15,
  "skills": [
    "Python",
    "Machine Learning",
    "Docker",
    "AWS"
  ],
  "degrees": [
    "Master of Science in Computer Science",
    "Stanford University"
  ],
  "experience": [
    "5 years",
    "Software Engineer",
    "Data Scientist"
  ],
  "metadata": {
    "pages": 2,
    "author": "John Doe"
  }
}
```

### 2. CSV Format
```csv
Entity Type,Entity Text,Position
SKILL,Python,1
SKILL,Machine Learning,2
DEGREE,Master of Science,3
EXPERIENCE,5 years,4
```

### 3. Text Summary
```
============================================================
Resume Analysis: john_doe_resume.pdf
============================================================

Total Entities Found: 15

SKILLS (8):
  • Python
  • Machine Learning
  • Docker
  • AWS
  ...

DEGREES (3):
  • Master of Science in Computer Science
  • Stanford University
  
EXPERIENCES (4):
  • 5 years
  • Software Engineer
  ...
```

### 4. HTML Visualization
- Color-coded entity highlighting
- Interactive entity table
- Professional styling
- Ready for web display

---

## 🔧 Advanced Features

### Text Extraction Methods

The system tries multiple extraction methods automatically:

1. **pdfplumber** (default) - Best for digital PDFs
2. **PyMuPDF** (fallback) - Alternative for complex PDFs
3. **OCR** (optional) - For scanned/image-based PDFs

**Manually specify method:**
```python
result = processor.process_pdf("resume.pdf", extract_method="ocr")
# Options: "auto", "pdfplumber", "pymupdf", "ocr"
```

### Entity Deduplication

The system automatically:
- Removes duplicate entities (case-insensitive)
- Preserves original casing
- Groups entities by type

### Resume Section Detection

Optional resume segmentation:
```python
result = processor.process_pdf("resume.pdf", segment=True)
sections = result['sections']
# Returns: {'summary': '...', 'experience': '...', 'education': '...', 'skills': '...'}
```

---

## ⚡ Performance Tips

### For Large Batches

```python
# Process in parallel (future enhancement)
from concurrent.futures import ThreadPoolExecutor

pdf_files = ["resume1.pdf", "resume2.pdf", ...]

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(processor.process_pdf, pdf_files))
```

### Memory Optimization

- Process PDFs one at a time for large files
- Enable text extraction only (skip visualization for batch)
- Use lightweight output formats (JSON/CSV vs HTML)

---

## 🐛 Troubleshooting

### Issue: No text extracted from PDF

**Solution:**
```bash
# Enable OCR
python scripts/process_pdf.py resume.pdf --ocr
```

### Issue: OCR not working

**Solution:**
```bash
# Check Tesseract installation
tesseract --version

# Install required packages
pip install pytesseract pdf2image

# Windows: Add Tesseract to PATH or configure in code
```

### Issue: ModuleNotFoundError for PDF libraries

**Solution:**
```bash
# Reinstall PDF dependencies
pip install pdfplumber PyMuPDF pytesseract Pillow pdf2image
```

### Issue: Model not found error

**Solution:**
```bash
# Train the model first
python scripts/train_bert.py

# Or specify custom model path
python scripts/process_pdf.py resume.pdf --model path/to/model.pt
```

---

## 📋 Example Workflow

### Processing a Candidate's Resume

```bash
# 1. Upload PDF to project directory
cp ~/Downloads/candidate_resume.pdf resumes/

# 2. Process with OCR (if scanned)
python scripts/process_pdf.py resumes/candidate_resume.pdf \
    --ocr \
    --output results/candidate1 \
    --format all

# 3. Results are saved:
# - results/candidate1/candidate_resume_entities.json
# - results/candidate1/candidate_resume_entities.csv
# - results/candidate1/candidate_resume_summary.txt
# - results/candidate1/candidate_resume_highlighted.html

# 4. Open HTML for visual review
start results/candidate1/candidate_resume_highlighted.html  # Windows
# open results/candidate1/candidate_resume_highlighted.html  # macOS
```

---

## 🔮 Future Enhancements

- [ ] Parallel batch processing
- [ ] Confidence scores per entity
- [ ] Support for DOCX/DOC formats
- [ ] Advanced section detection (projects, certifications)
- [ ] Entity normalization (e.g., "ML" → "Machine Learning")
- [ ] Integration with ATS systems
- [ ] REST API endpoint

---

## 📚 API Reference

### PDFTextExtractor

```python
from src.inference.pdf_parser import PDFTextExtractor

extractor = PDFTextExtractor(use_ocr=False)
text = extractor.extract_text("resume.pdf", method="auto")
metadata = extractor.extract_metadata("resume.pdf")
```

### PDFResumeProcessor

```python
from src.inference.pdf_processor import PDFResumeProcessor

processor = PDFResumeProcessor(
    model_path="models/checkpoints/bert/best_model.pt",
    label_mappings_path="models/checkpoints/bert/label_mappings.json",
    use_ocr=False
)

result = processor.process_pdf("resume.pdf")
processor.export_to_json(result, "output.json")
processor.export_to_csv(result, "output.csv")
processor.export_summary(result, "summary.txt")
html = processor.create_highlighted_resume(result)
```

---

## ✅ Checklist for Production Use

- [x] Install required dependencies
- [x] Train BERT model
- [ ] Set up OCR (if needed for scanned PDFs)
- [ ] Test with sample PDFs
- [ ] Configure output directories
- [ ] Set up batch processing scripts
- [ ] Create backup of trained models
- [ ] Document API usage for team

---

**The PDF processing system is now fully integrated and ready for use!** 🎉
