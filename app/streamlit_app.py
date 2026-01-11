"""
Streamlit Web Application for Resume NER (7-Entity Multilingual).
"""

import streamlit as st
import sys
from pathlib import Path
import json
import tempfile
import pandas as pd

# Add project root to path
import os
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference.predictor import ResumeNERPredictor
from src.inference.visualizer import EntityVisualizer
from src.inference.pdf_processor import PDFResumeProcessor

# Page configuration
st.set_page_config(
    page_title="Resume NER - Advanced Parser",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1e3799;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4a69bd;
        text-align: center;
        margin-bottom: 2rem;
    }
    .entity-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #eee;
        background: white;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = EntityVisualizer()

def load_system():
    # Attempt to find the hybrid CRF model first, then fall back to the base model
    crf_path = project_root / "models/checkpoints/bert_crf/best_model.pt"
    base_path = project_root / "models/checkpoints/bert/best_model.pt"
    
    checkpoint_path = crf_path if crf_path.exists() else base_path
    mappings_path = project_root / "models/checkpoints/bert_crf/label_mappings.json"
    
    if not mappings_path.exists():
        mappings_path = project_root / "models/checkpoints/bert/label_mappings.json"
    
    # Check if weights exist (usually not on Streamlit Cloud due to size)
    if not checkpoint_path.exists():
        return None, None, False
    
    try:
        # Load Predictor
        predictor = ResumeNERPredictor(
            model_path=str(checkpoint_path),
            label_mappings_path=str(mappings_path),
            model_name="bert-base-multilingual-cased"
        )
        # Load PDF Processor
        pdf_processor = PDFResumeProcessor()
        return predictor, pdf_processor, True
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None, None, False

# Interface Header
st.markdown('<div class="main-header">Resume NER - Advanced Parser 🚀</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Phase 6: Enterprise-Grade BERT+CRF Structural Analysis</div>', unsafe_allow_html=True)

# Load the system
with st.spinner("Initializing NLP Engine..."):
    predictor, pdf_processor, system_active = load_system()

if not system_active:
    st.error("❌ NLP Engine failed to initialize. Please check model weights.")
    st.stop()

# Examples
EXAMPLES = {
    "Academic Profile": """Karim Haddad
karim.haddad@example.com
Sousse, Tunisie

Génie Logiciel à l’École Polytechnique.
Data Scientist chez The Bridge (Août 2025).
Expertise: Python, Machine Learning, TF-IDF, SQLite.""",
    
    "Standard Resume": """Sarah Johnson | New York, USA | sarah.j@tech.com
Senior Software Engineer at Google (2020-2024). 
Bachelor of Computer Science from MIT.
Skills: Java, React, Docker, Kubernetes, AWS."""
}

# Sidebar
with st.sidebar:
    st.title("🚀 Control Panel")
    if st.button("🔄 Initialize System", type="primary", use_container_width=True):
        with st.spinner("Loading Multilingual BERT..."):
            try:
                p, pdf = load_system()
                st.session_state.predictor = p
                st.session_state.pdf_processor = pdf
                st.success("System Ready!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    st.divider()
    input_mode = st.radio("Select Input Method", ["Text Input", "PDF Upload"])
    
    st.divider()
    st.info("💡 **Tip**: This model is trained to distinguish between Skills and Names/Locations.")

# Main UI
st.markdown('<div class="main-header">Resume NER Advanced</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multilingual Entity Extraction with 7-Label Schema</div>', unsafe_allow_html=True)

col1, col2 = st.columns([3, 2])

with col1:
    if input_mode == "Text Input":
        st.subheader("📝 Input Text")
        selected_example = st.selectbox("Choose an example", [""] + list(EXAMPLES.keys()))
        input_text = st.text_area("Paste Resume Text", value=EXAMPLES.get(selected_example, ""), height=250)
        deep_analysis = st.checkbox("🔍 Deep Analysis (Section Grouping)", value=True)
        process_btn = st.button("Run Extraction", type="primary")
    else:
        st.subheader("📄 Upload PDF")
        uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])
        deep_analysis = st.checkbox("🔍 Deep Analysis (Section Grouping)", value=True, key="pdf_deep")
        process_btn = st.button("Process PDF", type="primary")

with col2:
    st.subheader("📋 Parsed Results")
    if process_btn:
        if 'predictor' not in st.session_state or st.session_state.predictor is None:
            st.warning("Please load the system first via the sidebar.")
        else:
            with st.spinner("Analyzing..."):
                if input_mode == "Text Input":
                    if deep_analysis:
                        data = st.session_state.predictor.predict_with_sections(input_text)
                    else:
                        data = st.session_state.predictor.get_structured_json(input_text)
                    
                    entities = st.session_state.predictor.predict(input_text)
                    viz_html = st.session_state.visualizer.get_html(input_text, entities)
                else:
                    if uploaded_file:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(uploaded_file.getvalue())
                            res = st.session_state.pdf_processor.process_pdf(tmp.name)
                        
                        entities = res['entities']
                        if deep_analysis:
                            # Group by section logic for PDF
                            data = st.session_state.predictor.predict_with_sections(res['extracted_text'])
                        else:
                            data = st.session_state.pdf_processor.skill_filter.normalize_results(entities)
                        
                        viz_html = st.session_state.visualizer.get_html(res['extracted_text'], entities)
                    else:
                        st.error("No file uploaded")
                        st.stop()
                
                # Display Results
                st.json(data)
                
                # Filter and display Skills Table
                skills = [e['entity'] for e in entities if e['type'] == 'SKILL']
                if skills:
                    st.divider()
                    st.subheader("🛠️ Extracted Skills")
                    df_skills = pd.DataFrame(skills, columns=["Skill Name"])
                    st.dataframe(df_skills, use_container_width=True, hide_index=True)
                
                st.session_state.current_viz = viz_html
                st.session_state.current_data = data
                st.session_state.current_text = input_text if input_mode == "Text Input" else res['extracted_text']

# Correction & Feedback Section
if 'current_data' in st.session_state:
    with st.expander("🛠️ Correction & Feedback Loop"):
        st.write("Help improve the model! Edit the results below if anything is wrong.")
        corrected_json = st.text_area("Corrected JSON", value=json.dumps(st.session_state.current_data, indent=2), height=200)
        
        if st.button("Submit Correction"):
            try:
                feedback_dir = project_root / "data/feedback"
                feedback_dir.mkdir(exist_ok=True)
                feedback_path = feedback_dir / "corrections.jsonl"
                
                feedback_item = {
                    "text": st.session_state.current_text,
                    "correction": json.loads(corrected_json)
                }
                
                with open(feedback_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(feedback_item) + "\n")
                
                st.success("Thank you! Correction saved to data/feedback/corrections.jsonl")
            except Exception as e:
                st.error(f"Error saving feedback: {e}")

# Visualization Footer
if 'current_viz' in st.session_state:
    st.divider()
    st.subheader("🎨 Entity Visualization")
    st.session_state.visualizer.display_legend()
    st.markdown(st.session_state.current_viz, unsafe_allow_html=True)
