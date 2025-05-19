# some texts suffer from glued words after extracting texts from PDFs. 
# this script handels this by using spacy and wordninja
import os
from pathlib import Path
import spacy
import wordninja

# === Config ===
INPUT_FOLDER = Path("...")
OUTPUT_FOLDER = Path("...")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def split_text_into_chunks(text, max_length=900000):
   # Split text into smaller chunks to avoid hitting spaCy's max_length limit.
   
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def smart_unglue_text(text):
    # Detect and correct glued words using spaCy and wordninja.
    # Handles very long texts safely by chunking.
    
    corrected_tokens = []

    chunks = split_text_into_chunks(text)
    for chunk_idx, chunk in enumerate(chunks):
        doc = nlp(chunk)
        for token in doc:
            if token.is_oov and token.is_alpha and len(token.text) > 10:
                split = wordninja.split(token.text)
                if len(split) > 1 and all(len(w) > 1 for w in split):
                    corrected_tokens.extend(split)
                else:
                    corrected_tokens.append(token.text)
            else:
                corrected_tokens.append(token.text)

    return ' '.join(corrected_tokens)


# === Process all .txt files ===
txt_files = list(INPUT_FOLDER.glob("*.txt"))
print(f"Found {len(txt_files)} text files.\n")

for i, txt_file in enumerate(txt_files, 1):
    print(f"[{i}/{len(txt_files)}] Processing: {txt_file.name}")
    with open(txt_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    cleaned_text = smart_unglue_text(text)

    output_path = OUTPUT_FOLDER / txt_file.name
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

print(f"\n All cleaned texts saved to: {OUTPUT_FOLDER}")
