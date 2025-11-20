import streamlit as st
from multiprocessing import Process
from annotated_text import annotated_text
from bs4 import BeautifulSoup
import pandas as pd
import torch
import math
import re
import json
import requests
import spacy
import errant
import time
import os


def start_server():   
    os.system("python3 -m spacy download en_core_web_sm")
    os.system("uvicorn InferenceServer:app --port 8080 --host 0.0.0.0 --workers 2")


def load_models():
    if not is_port_in_use(8080):
        with st.spinner(text="Loading models, please wait..."):
            proc = Process(target=start_server, args=(), daemon=True)
            proc.start()
            while not is_port_in_use(8080):
                time.sleep(1)
            st.success("Model server started.")
    else:
        st.success("Model server already running...")
    st.session_state['models_loaded'] = True


def is_port_in_use(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) == 0


if 'models_loaded' not in st.session_state:
    st.session_state['models_loaded'] = False


def show_highlights(input_text, corrected_sentence):
    try:
        strikeout = lambda x: '\u0336'.join(x) + '\u0336'
        highlight_text = highlight(input_text, corrected_sentence)
        
        # Updated colors for better visibility on both themes
        color_map = {
            'd': '#ff6b6b',  # Soft red for deletions (works on both themes)
            'a': '#51cf66',  # Soft green for additions (works on both themes)
            'c': '#ffd43b'   # Soft yellow for changes (works on both themes)
        }
        
        tokens = re.split(r'(<[dac]\s.*?<\/[dac]>)', highlight_text)
        annotations = []
        for token in tokens:
            soup = BeautifulSoup(token, 'html.parser')
            tags = soup.findAll()
            if tags:
                _tag = tags[0].name
                _type = tags[0]['type']
                _text = tags[0]['edit']
                _color = color_map[_tag]

                if _tag == 'd':
                    _text = strikeout(tags[0].text)

                annotations.append((_text, _type, _color))
            else:
                annotations.append(token)
        annotated_text(*annotations)
    except Exception as e:
        st.error('Some error occured!' + str(e))
        st.stop()


def show_edits(input_text, corrected_sentence):
    try:
        edits = get_edits(input_text, corrected_sentence)
        df = pd.DataFrame(edits, columns=['type','original word', 'original start', 'original end', 'correct word', 'correct start', 'correct end'])
        df = df.set_index('type')
        st.table(df)
    except Exception as e:
        st.error('Some error occured!')
        st.stop()


def highlight(orig, cor):
      edits = _get_edits(orig, cor)
      orig_tokens = orig.split()

      ignore_indexes = []

      for edit in edits:
          edit_type = edit[0]
          edit_str_start = edit[1]
          edit_spos = edit[2]
          edit_epos = edit[3]
          edit_str_end = edit[4]

          # if no_of_tokens(edit_str_start) > 1 ==> excluding the first token, mark all other tokens for deletion
          for i in range(edit_spos+1, edit_epos):
            ignore_indexes.append(i)

          if edit_str_start == "":
              if edit_spos - 1 >= 0:
                  new_edit_str = orig_tokens[edit_spos - 1]
                  edit_spos -= 1
              else:
                  new_edit_str = orig_tokens[edit_spos + 1]
                  edit_spos += 1
              if edit_type == "PUNCT":
                st = "<a type='" + edit_type + "' edit='" + \
                    edit_str_end + "'>" + new_edit_str + "</a>"
              else:
                st = "<a type='" + edit_type + "' edit='" + new_edit_str + \
                    " " + edit_str_end + "'>" + new_edit_str + "</a>"
              orig_tokens[edit_spos] = st
          elif edit_str_end == "":
            st = "<d type='" + edit_type + "' edit=''>" + edit_str_start + "</d>"
            orig_tokens[edit_spos] = st
          else:
            st = "<c type='" + edit_type + "' edit='" + \
                edit_str_end + "'>" + edit_str_start + "</c>"
            orig_tokens[edit_spos] = st

      for i in sorted(ignore_indexes, reverse=True):
        del(orig_tokens[i])

      return(" ".join(orig_tokens))


def _get_edits(orig, cor):
    orig = annotator.parse(orig)
    cor = annotator.parse(cor)
    alignment = annotator.align(orig, cor)
    edits = annotator.merge(alignment)

    if len(edits) == 0:  
        return []

    edit_annotations = []
    for e in edits:
        e = annotator.classify(e)
        edit_annotations.append((e.type[2:], e.o_str, e.o_start, e.o_end,  e.c_str, e.c_start, e.c_end))
            
    if len(edit_annotations) > 0:
        return edit_annotations
    else:    
        return []


def get_edits(orig, cor):
    return _get_edits(orig, cor)        


def get_correction(input_text):
    correct_request = "http://0.0.0.0:8080/correct?input_sentence="+input_text
    correct_response = requests.get(correct_request)
    correct_json = json.loads(correct_response.text)
    scored_corrected_sentence = correct_json["scored_corrected_sentence"]
    
    corrected_sentence, score = scored_corrected_sentence
    st.markdown(f'##### Corrected text:')
    st.write('')
    st.success(corrected_sentence)
    exp1 = st.expander(label='Show highlights', expanded=True)
    with exp1:
        # Add legend for color coding
        st.markdown("""
        <div style='margin-bottom: 10px; font-size: 0.9em;'>
            <span style='background-color: #ff6b6b; padding: 2px 6px; border-radius: 3px; margin-right: 10px;'>🗑️ Deletion</span>
            <span style='background-color: #51cf66; padding: 2px 6px; border-radius: 3px; margin-right: 10px;'>➕ Addition</span>
            <span style='background-color: #ffd43b; padding: 2px 6px; border-radius: 3px;'>🔄 Change</span>
        </div>
        """, unsafe_allow_html=True)
        show_highlights(input_text, corrected_sentence)
    exp2 = st.expander(label='Show edits')
    with exp2:
        show_edits(input_text, corrected_sentence)
          
        
if __name__ == "__main__":
    
        st.title('Grammify - AI-Powered Grammar Correction')
        st.subheader('Intelligent grammar error detection and correction using Transformer models')
        st.markdown("Built by **Muhammad Abdullah Rasheed** | [![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blue)](https://techvibes360.com) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/abdullahrasheed-/) [![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/Abdullahrasheed45)", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("🎯 **High Accuracy**")
            st.caption("T5-based Seq2Seq model")
        with col2:
            st.markdown("⚡ **Real-time**")
            st.caption("Fast inference engine")
        with col3:
            st.markdown("🔍 **Detailed Analysis**")
            st.caption("Visual error breakdown")
        
        st.markdown("---")

        examples = [
                     "what be the reason for everyone leave the company",
                    "They're house is on fire",
                    "Look if their is fire on the top",
                    "Where is you're car?",
                    "Its going to rain",
                    "Feel free reach out to me",
                    "Life is shortest so live freely",
                    "We do the boy actually stole the books",
                    "I am doing fine. How is you?",
                    "Each of you all should run fast", 
                    "Matt like fish",
                    "We enjoys horror movies",
                    "I walk to the store and I bought milk",
                    "We all eat the fish and then made dessert",
                    ]

        if not st.session_state['models_loaded']:
            load_models()                     

        import en_core_web_sm
        nlp = en_core_web_sm.load()
        annotator = errant.load('en', nlp)

        st.markdown(f'### Try it now:')
        input_text = st.selectbox(
        label="Choose an example",
        options=examples
        )
        st.write("(or)")
        input_text = st.text_input(
            label="Bring your own sentence",
            value=input_text
        )

        if input_text.strip(): 
            get_correction(input_text)
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: gray; font-size: 0.9em;'>
            <p>Powered by Transformers & FastAPI | Deployed on Hugging Face Spaces</p>
            <p>© 2024 Muhammad Abdullah Rasheed</p>
        </div>
        """, unsafe_allow_html=True)
