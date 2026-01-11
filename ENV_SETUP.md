# Virtual Environment Setup

## Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

## Step 2: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Install PDF processing libraries
pip install pdfplumber pymupdf
pip install pytesseract pillow pdf2image

# Install additional utilities
pip install python-multipart  # For file uploads in Streamlit
```

## Step 3: OCR Setup (Optional - for scanned PDFs)

### Windows:
1. Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki
2. Install and note the installation path
3. Add to PATH or configure in code

### macOS:
```bash
brew install tesseract
```

### Linux:
```bash
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils  # For pdf2image
```

## Step 4: Verify Installation

```bash
python -c "import pdfplumber; print('pdfplumber:', pdfplumber.__version__)"
python -c "import fitz; print('PyMuPDF: OK')"
python -c "import pytesseract; print('pytesseract: OK')"
```

## Step 5: Ready to Use

Once environment is activated:
- Generate dataset: `python scripts/generate_dataset.py`
- Train model: `python scripts/train_bert.py`
- Process PDFs: `python scripts/process_pdf.py resume.pdf`
- Launch UI: `streamlit run app/streamlit_app.py`
