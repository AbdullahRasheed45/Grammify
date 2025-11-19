import uvicorn
from fastapi import File
from fastapi import FastAPI
from fastapi import UploadFile
import torch
import os
import sys
import glob
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM


print("Loading models...")
app = FastAPI()

device = "cpu"
correction_model_tag = "prithivida/grammar_error_correcter_v1"
correction_tokenizer = AutoTokenizer.from_pretrained(correction_model_tag)
correction_model     = AutoModelForSeq2SeqLM.from_pretrained(correction_model_tag)

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

print("Models loaded !")


@app.get("/")
def read_root():
    return {"Gramformer !"}

@app.get("/{correct}")
def get_correction(input_sentence):
    set_seed(1212)
    scored_corrected_sentence = correct(input_sentence)
    return {"scored_corrected_sentence": scored_corrected_sentence}

def correct(input_sentence, max_candidates=1):
    correction_prefix = "gec: "
    input_sentence = correction_prefix + input_sentence
    input_ids = correction_tokenizer.encode(input_sentence, return_tensors='pt')
    input_ids = input_ids.to(device)

    preds = correction_model.generate(
        input_ids,
        do_sample=True, 
        max_length=128, 
        top_k=50, 
        top_p=0.95, 
#        num_beams=7,
        early_stopping=True,
        num_return_sequences=max_candidates)

    corrected = set()
    for pred in preds:  
        corrected.add(correction_tokenizer.decode(pred, skip_special_tokens=True).strip())

    corrected = list(corrected)
    return (corrected[0], 0)  #Corrected Sentence, Dummy score
