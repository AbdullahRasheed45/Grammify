Grammify - Intelligent Grammar Correction System
AI-Powered Grammar Error Detection Using Transformer Models
<div align="center">
[
[
[
[

NLP Application Project | Deployed on Hugging Face Spaces

</div>
An intelligent grammar correction application leveraging state-of-the-art Seq2Seq transformer models to detect and correct grammatical errors with real-time visual feedback and detailed linguistic error analysis.

Overview
Grammify implements an advanced grammar correction system designed to enhance written communication across professional, academic, and personal contexts. Built on the Gramformer library and powered by the T5-based prithivida/grammar_error_correcter_v1 model, the system processes natural language input through a transformer architecture to identify and correct diverse grammatical errors with high accuracy and contextual awareness.

Technical Context: Full-stack NLP application integrating FastAPI microservices, Streamlit frontend, and Hugging Face Transformers for production-grade grammar correction.

Key Features
Transformer-Based Architecture
Seq2Seq Deep Learning: T5-based encoder-decoder architecture processes grammatical correction as sequence-to-sequence translation

Production Deployment: FastAPI inference server with uvicorn workers for concurrent request handling

Real-time Processing: ~2-3 second inference latency per sentence

Grammar Error Coverage
The system corrects 15+ grammatical error types with high linguistic precision:

Error Type	Description	Example Correction
Subject-Verb Agreement	Verb conjugation matching subject	"Matt like fish" → "Matt likes fish"
Verb Tense Consistency	Temporal coherence in narratives	"I walk to the store and I bought milk" → "I walked to the store and bought milk"
Article Usage	Determiner selection (a/an/the)	Missing or incorrect articles
Pronoun Errors	Possessive vs. contraction	"They're house" → "Their house"
Preposition Selection	Contextual preposition choice	"Feel free reach out" → "Feel free to reach out"
Word Form	Part-of-speech corrections	"Life is shortest" → "Life is short"
Auxiliary Verbs	Modal and helping verb errors	"what be the reason" → "what is the reason"
Gerund/Infinitive	Verb form following verbs	"everyone leave" → "everyone leaving"
Pronoun Case	Subject/object pronoun usage	"How is you?" → "How are you?"
Punctuation	Apostrophes, commas, periods	"Its going to rain" → "It's going to rain"
Interactive Visualization
Color-Coded Annotations: Visual highlighting system distinguishes error types

Red (Deletion): Words/characters to remove

Green (Addition): Missing words/characters

Yellow (Change): Word replacements or modifications

Detailed Edit Tables: Structured breakdown of each grammatical correction with token positions

Linguistic Error Classification: ERRANT-based error type identification (morphology, syntax, orthography)

System Performance
Model Specifications
text
Model Architecture:     T5-based Seq2Seq Transformer
Model Tag:             prithivida/grammar_error_correcter_v1
Tokenizer:             AutoTokenizer (SentencePiece)
Maximum Sequence:      128 tokens
Sampling Strategy:     Top-k (50) + Top-p (0.95)
Temperature:           1.0 (diverse generation)
Device:                CPU (GPU compatible)
Inference Latency:     ~2-3 seconds per sentence
Model Size:            ~220MB (full precision)
Generation Parameters
python
Generation Configuration:
├── do_sample: True          # Stochastic sampling enabled
├── max_length: 128          # Maximum output tokens
├── top_k: 50                # Top-k sampling threshold
├── top_p: 0.95             # Nucleus sampling probability
├── early_stopping: True     # Stop at first EOS token
└── num_return_sequences: 1  # Single best candidate
System Architecture Performance
Component	Performance Metric
FastAPI Server	Multi-worker uvicorn deployment
Startup Time	~15-20 seconds (model loading)
Concurrent Requests	Handles 2+ simultaneous corrections
Port Configuration	8080 (inference server)
Health Check	Socket-based port availability monitoring
Technical Architecture
Seq2Seq Transformer Pipeline
python
Input Text: "what be the reason for everyone leave the company"
    ↓
Preprocessing: Add task prefix → "gec: what be the reason..."
    ↓
Tokenization: SentencePiece encoding → Token IDs
    ↓
T5 Encoder: Contextualized embeddings (512 dimensions)
    ↓
T5 Decoder: Autoregressive generation with beam search
    ↓
Sampling: Top-k (50) + Top-p (0.95) filtering
    ↓
Detokenization: Token IDs → "what is the reason for everyone leaving the company"
    ↓
Post-processing: Remove special tokens, strip whitespace
    ↓
Output: Corrected sentence + confidence score
Key Technical Design:

Task Prefix: "gec: " signals grammar error correction task to T5 model

Encoder-Decoder: Bidirectional attention in encoder, causal attention in decoder

Sampling Strategy: Balances diversity (top-p) and quality (top-k) for natural corrections

Early Stopping: Terminates generation at first end-of-sequence token for efficiency

Error Analysis Pipeline
python
Original Sentence → spaCy Tokenization
                          ↓
Corrected Sentence → spaCy Tokenization
                          ↓
              ERRANT Alignment
                          ↓
        Edit Extraction & Classification
                          ↓
    ┌──────────────┬──────────────┐
    │  Highlights  │  Edit Table  │
    │  (Visual)    │  (Tabular)   │
    └──────────────┴──────────────┘
ERRANT Framework Integration:

Parse Trees: spaCy dependency parsing for syntactic structure

Token Alignment: Levenshtein-based sequence alignment

Edit Operations: Insertions, deletions, substitutions, and transpositions

Linguistic Classification: Maps edits to error taxonomy (VERB:TENSE, DET, PREP, etc.)

System Architecture
text
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                       │
│         • Interactive text input interface                  │
│         • Pre-loaded example selector                       │
│         • Visual error highlighting display                 │
│         • Expandable edit table components                  │
└─────────────────┬───────────────────────────────────────────┘
                  │ HTTP POST
┌─────────────────▼───────────────────────────────────────────┐
│              FastAPI Inference Server                       │
│         • uvicorn ASGI server (port 8080)                   │
│         • Multi-worker request handling                     │
│         • Health check and monitoring                       │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────────────────┐
│           Grammar Correction Engine                         │
│    ┌─────────────────────┬─────────────────────┐            │
│    │   T5 Transformer    │   ERRANT Analyzer   │            │
│    │   • Tokenization    │   • spaCy NLP       │            │
│    │   • Seq2Seq Gen     │   • Error taxonomy  │            │
│    └─────────────────────┴─────────────────────┘            │
└─────────────────────────────────────────────────────────────┘
Microservices Design:

Frontend Layer (Streamlit): User interaction and visualization

API Layer (FastAPI): Stateless request processing

Model Layer (Transformers): Core correction logic

Analysis Layer (ERRANT): Linguistic error identification

Installation
Prerequisites
bash
Python 3.8+
4GB RAM minimum
Internet connection (initial model download)
Webcam optional (for future multimodal features)
Backend Setup (FastAPI + Transformers)
bash
# Clone repository
git clone https://huggingface.co/spaces/Abdullahrasheed45/Grammify
cd Grammify

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm

# Start FastAPI inference server (automatic on first run)
# Server launches at http://0.0.0.0:8080

# Start Streamlit application
streamlit run app.py
# Application available at http://localhost:8501
Docker Deployment (Optional)
bash
# Build Docker image
docker build -t grammify:latest .

# Run container
docker run -p 8501:8501 -p 8080:8080 grammify:latest
Hugging Face Spaces Deployment
bash
# Configure space metadata in README.md
---
title: Grammify
emoji: ⚡
colorFrom: gray
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
license: apache-2.0
sdk_version: 1.51.0
---

# Push to Hugging Face Hub
git push https://huggingface.co/spaces/YOUR_USERNAME/Grammify main
Usage
Interactive Web Application
The system provides a Streamlit-based interface with the following workflow:

Basic Correction:

Choose Example - Select from 14 pre-loaded grammatical error examples

Custom Input - Enter your own sentence in the text input field

Automatic Processing - Correction triggers on non-empty input

View Results - Corrected text displayed in success banner

Analyze Errors - Expand "Show highlights" for color-coded annotations

Inspect Edits - Expand "Show edits" for detailed error breakdown

Example Workflow:

python
# Input
"Matt like fish"

# Output (Success Banner)
"Matt likes fish"

# Highlights (Expandable)
Matt [like → likes (VERB:SVA)] fish

# Edit Table (Expandable)
| Type | Original | Pos | Corrected | Pos |
|------|----------|-----|-----------|-----|
| VERB:SVA | like | 1-2 | likes | 1-2 |
API Integration
For programmatic access, use the FastAPI endpoint:

python
import requests

# Make correction request
response = requests.get(
    "http://0.0.0.0:8080/correct",
    params={"input_sentence": "They're house is on fire"}
)

# Parse response
result = response.json()
corrected_text = result["scored_corrected_sentence"][0]
confidence = result["scored_corrected_sentence"][1]

print(f"Corrected: {corrected_text}")
# Output: "Their house is on fire"
Python Library Integration
python
# Direct model usage (without server)
from gramformer import Gramformer

# Initialize model
gf = Gramformer(models=1, use_gpu=False)

# Correct sentence
corrections = gf.correct(
    "Feel free reach out to me",
    max_candidates=1
)

for corrected in corrections:
    print(corrected)
# Output: "Feel free to reach out to me"
Technical Implementation
File Structure
text
Grammify/
├── app.py                    # Main Streamlit application
├── InferenceServer.py        # FastAPI inference server
├── requirements.txt          # Python dependencies
├── .gitattributes           # Git LFS configuration
└── README.md                # This documentation
Core Dependencies
requirements.txt Analysis:

python
# NLP & Deep Learning
transformers           # Hugging Face model hub
torch                 # PyTorch backend
sentencepiece         # Tokenization

# Web Frameworks
streamlit             # Interactive frontend
fastapi              # API server
uvicorn              # ASGI server

# Grammar Analysis
spacy                # Linguistic processing
errant               # Error annotation toolkit
nltk (>=3.6)         # Natural language toolkit

# Utilities
st-annotated-text    # Visual highlighting
bs4                  # HTML parsing for annotations
pandas               # Edit table generation
protobuf (>=3.19.0)  # Model serialization
requests             # HTTP client
Key Code Components
1. InferenceServer.py - Core Correction Logic

python
# Model initialization
correction_model_tag = "prithivida/grammar_error_correcter_v1"
correction_tokenizer = AutoTokenizer.from_pretrained(correction_model_tag)
correction_model = AutoModelForSeq2SeqLM.from_pretrained(correction_model_tag)

# Correction function
def correct(input_sentence, max_candidates=1):
    correction_prefix = "gec: "
    input_sentence = correction_prefix + input_sentence
    input_ids = correction_tokenizer.encode(input_sentence, return_tensors='pt')
    
    preds = correction_model.generate(
        input_ids,
        do_sample=True,
        max_length=128,
        top_k=50,
        top_p=0.95,
        early_stopping=True,
        num_return_sequences=max_candidates
    )
    
    corrected = set()
    for pred in preds:
        corrected.add(correction_tokenizer.decode(pred, skip_special_tokens=True).strip())
    
    return (corrected[0], 0)  # Corrected sentence, dummy confidence
2. app.py - Error Analysis Pipeline

python
# ERRANT-based edit extraction
