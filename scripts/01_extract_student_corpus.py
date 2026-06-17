import os
import glob
import pickle
import json
import re
import pandas as pd
from pathlib import Path

def extract_student_corpus(txt_dir):
    files = sorted(glob.glob(os.path.join(txt_dir, "meeting_*.txt")))
    student_texts = []
    
    def is_header(l):
        if "原文" in l:
            return True
        if re.match(r"^\d{4}年\d{2}月\d{2}日", l):
            return True
        return False
        
    def get_speaker_type(speaker_name):
        cleaned = re.sub(r'[\s_]+', '', speaker_name).lower()
        if cleaned.startswith('student') or cleaned.startswith('sudden') or 'student' in cleaned:
            return "student"
        if cleaned in ['researcher', 'yuxingao', '明明', '刘明明', '老师', 'host'] or cleaned.startswith('发言人'):
            return "researcher"
        return None

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_speaker_type = None
        current_text_buffer = []
        
        is_meeting_01 = "meeting_01" in os.path.basename(filepath)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if is_header(line):
                continue
            
            if is_meeting_01:
                # Same-line speaker with colon
                same_line_match = re.match(r"^([^:：()（）\s]+(?:\s+[^:：()（）\s]+)*)(?:\s*[(（][^）)][)）])?\s*[:：]\s*(.*)", line)
                if same_line_match:
                    speaker = same_line_match.group(1).strip()
                    text = same_line_match.group(2).strip()
                    spk_type = get_speaker_type(speaker)
                    if spk_type:
                        if current_speaker_type == "student":
                            joined = " ".join(current_text_buffer).strip()
                            if joined:
                                student_texts.append(joined)
                        current_speaker_type = spk_type
                        current_text_buffer = [text]
            else:
                # Own-line speaker format
                cleaned = re.sub(r'[\s_]+', '', line).lower()
                spk_type = get_speaker_type(cleaned)
                if spk_type:
                    if current_speaker_type == "student":
                        joined = " ".join(current_text_buffer).strip()
                        if joined:
                            student_texts.append(joined)
                    current_speaker_type = spk_type
                    current_text_buffer = []
                else:
                    if current_speaker_type == "student":
                        current_text_buffer.append(line)
        
        # Flush last speaker of the file
        if current_speaker_type == "student":
            joined = " ".join(current_text_buffer).strip()
            if joined:
                student_texts.append(joined)
                
    return student_texts

def extract_ai_corpus(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return []
        
    print("Loading pickle file using pandas...")
    try:
        data = pd.read_pickle(pkl_path)
    except Exception as e:
        print(f"Error reading pickle: {e}")
        return []
        
    ai_texts = []
    print(f"Pickle loaded. Type: {type(data)}")
    
    if isinstance(data, dict):
        print(f"Keys in model_texts: {list(data.keys())}")
        target_model = "32B_Det"
        if target_model in data:
            df = data[target_model]
            print(f"Extracting texts from dataframe of shape {df.shape}")
            # Filter to prompt == "P1_Child" and lang == "zh" (Chinese output, child role)
            df_filtered = df[(df["lang"] == "zh") & (df["prompt"] == "P1_Child")]
            print(f"Filtered to P1_Child & lang=zh: {df_filtered.shape[0]} rows.")
            if "reason" in df_filtered.columns:
                for t in df_filtered["reason"].dropna().astype(str).tolist():
                    ai_texts.append({"text": t, "type": "reason"})
            if "suggestion" in df_filtered.columns:
                for t in df_filtered["suggestion"].dropna().astype(str).tolist():
                    ai_texts.append({"text": t, "type": "suggestion"})
    
    ai_texts = [{"text": t["text"].strip(), "type": t["type"]} for t in ai_texts if t["text"].strip() and t["text"].strip() != "nan"]
    return ai_texts

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    txt_dir = base_dir / "txt_cleaned"
    out_dir = base_dir / "data"
    os.makedirs(out_dir, exist_ok=True)
    
    print("Extracting student corpus...")
    student_texts = extract_student_corpus(txt_dir)
    print(f"Extracted {len(student_texts)} student speech segments.")
    for i in range(min(5, len(student_texts))):
        print(f"  [Sample {i+1}]: {student_texts[i][:150]}...")
        
    with open(out_dir / 'corpus_children.json', 'w', encoding='utf-8') as f:
        json.dump(student_texts, f, ensure_ascii=False, indent=2)
        
    ai_pkl_path = out_dir / "model_texts.pkl"
    print(f"\nExtracting AI corpus from {ai_pkl_path}...")
    ai_texts = extract_ai_corpus(ai_pkl_path)
    print(f"Extracted {len(ai_texts)} AI text segments.")
    if len(ai_texts) > 0:
        for i in range(min(5, len(ai_texts))):
            print(f"  [Sample {i+1}]: {ai_texts[i]['text'][:150]}...")
            
    with open(out_dir / 'corpus_ai.json', 'w', encoding='utf-8') as f:
        json.dump(ai_texts, f, ensure_ascii=False, indent=2)

    print("\nExtraction complete.")
