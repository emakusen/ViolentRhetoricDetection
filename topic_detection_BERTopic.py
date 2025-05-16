"""
BERTopic Topic Modeling Pipeline with Chunked Texts and LLM Labeling Prep

This script performs topic modeling using BERTopic on a folder of `.txt` documents. It handles variable-length
documents by chunking large texts, applies sentence embeddings, clusters them into topics using HDBSCAN,
and exports both:

1. A document-topic probability matrix (`bertopic_document_topic_matrix.csv`)
2. A CSV file with topic numbers and top 10 keywords (`bertopic_llm_labeling_keywords.csv`)
   for use with a local LLM (e.g., Mistral via Ollama) for automatic topic labeling.

Steps:
- Texts are chunked if they exceed `MAX_TOKENS`, using token-aware overlapping chunks
- Sentence embeddings are created with `sentence-transformers` (MiniLM)
- Topics are modeled via BERTopic and HDBSCAN
- Topic probabilities across chunks are aggregated back to the original document level
- Top topic keywords are saved for further use (e.g., prompting an LLM for topic labels)

Notes:
- Robust to documents ranging from a few words to hundreds of thousands of words
- Designed to support follow-up RAG/LLM pipelines

Dependencies:
    pip install bertopic sentence-transformers tiktoken hdbscan pandas
"""


from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import tiktoken
from hdbscan import HDBSCAN

FOLDER = "..."
MODEL_NAME = "all-MiniLM-L6-v2"
MAX_TOKENS = 512
OVERLAP = 50  # tokens

tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def count_tokens(text):
    return len(tokenizer.encode(text))

def chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        chunk_words = []
        token_count = 0
        for word in words[start:]:
            word_tokens = count_tokens(word)
            if token_count + word_tokens > max_tokens:
                break
            chunk_words.append(word)
            token_count += word_tokens
        if not chunk_words:
            chunk_words = [words[start]]
        chunk = ' '.join(chunk_words)
        chunks.append(chunk)
        if start + len(chunk_words) >= len(words):
            break
        start += len(chunk_words) - overlap if (len(chunk_words) - overlap) > 0 else 1
    return chunks

# Read and Chunk Files
all_chunks = []
chunk_to_file = []
file_names = []

folder = Path(FOLDER)
txt_files = list(folder.glob("*.txt"))

for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8", errors='replace') as f:
        text = f.read()
    if count_tokens(text) > MAX_TOKENS:
        chunks = chunk_text(text, max_tokens=MAX_TOKENS, overlap=OVERLAP)
    else:
        chunks = [text]
    all_chunks.extend(chunks)
    chunk_to_file.extend([file_path.name] * len(chunks))
    file_names.append(file_path.name)

# BERTopic
embedding_model = SentenceTransformer(MODEL_NAME)
topic_model = BERTopic(
    embedding_model=embedding_model,
    min_topic_size=2,
    hdbscan_model=HDBSCAN(min_cluster_size=2, min_samples=1, prediction_data=True),
    calculate_probabilities=True
)
topics, probs = topic_model.fit_transform(all_chunks)

# Build Document-Topic Matrix
# Get topic numbers in the order of probs (excluding -1)
topic_nums = [t for t in topic_model.get_topics().keys() if t != -1]
topic_indices = {t: i for i, t in enumerate(topic_nums)}

matrix = pd.DataFrame(0.0, index=file_names, columns=[f"Topic_{t}" for t in topic_nums])

for prob, file_name in zip(probs, chunk_to_file):
    for topic_num in topic_nums:
        idx = topic_indices[topic_num]
        matrix.loc[file_name, f"Topic_{topic_num}"] += prob[idx]

chunk_counts = pd.Series(chunk_to_file).value_counts()
for file_name in file_names:
    matrix.loc[file_name] /= chunk_counts[file_name]

matrix.to_csv("bertopic_document_topic_matrix.csv")
print("Document-topic matrix saved to bertopic_document_topic_matrix.csv")


#### generate csv for topic labeling
import pandas as pd

topic_nums = [t for t in topic_model.get_topics().keys() if t != -1]
topic_keywords = []

for topic_num in topic_nums:
    words_scores = topic_model.get_topic(topic_num)  # Default: 10 words
    top_words = [word for word, score in words_scores]
    topic_keywords.append({
        "topic_num": topic_num,
        "keywords": ", ".join(top_words)
    })

df_llm_labeling = pd.DataFrame(topic_keywords)
df_llm_labeling.to_csv("bertopic_llm_labeling_keywords.csv", index=False)
print("Saved topic keywords for LLM labeling.")
