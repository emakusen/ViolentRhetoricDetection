from pathlib import Path
import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
import optuna
import tiktoken
import pickle

# ------------------ SETTINGS ------------------
FOLDER = "..."
MODEL_NAME = "all-MiniLM-L6-v2"
LANGUAGE = "english"
MAX_TOKENS = 512
OVERLAP = 50  # tokens
N_TRIALS = 30
SAMPLE_SIZE = 500  # number of chunks for tuning

# ------------------ TOKENIZER ------------------
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

# ------------------ LOAD & CHUNK ------------------
folder = Path(FOLDER)
txt_files = list(folder.glob("*.txt"))
all_chunks = []

for file_path in txt_files:
    with open(file_path, "r", encoding="utf-8", errors='replace') as f:
        text = f.read()
    if count_tokens(text) > MAX_TOKENS:
        chunks = chunk_text(text)
    else:
        chunks = [text]
    all_chunks.extend(chunks)

# ------------------ SAMPLE & EMBED ------------------
subset_chunks = all_chunks[:SAMPLE_SIZE]
embedding_model = SentenceTransformer(MODEL_NAME)

# Check if embeddings already exist
embedding_file = "subset_embeddings.pkl"
try:
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
    print("Loaded existing subset embeddings.")
except FileNotFoundError:
    embeddings = embedding_model.encode(subset_chunks, show_progress_bar=True)
    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings, f)
    print("Saved subset embeddings to disk.")

# ------------------ OPTUNA OBJECTIVE ------------------
def objective(trial):
    min_cluster_size = trial.suggest_int("min_cluster_size", 10, 50)
    min_samples = trial.suggest_int("min_samples", 1, 10)
    min_topic_size = trial.suggest_int("min_topic_size", 2, 20)

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        prediction_data=True,
        metric="euclidean"
    )

    topic_model = BERTopic(
        embedding_model=embedding_model,
        min_topic_size=min_topic_size,
        hdbscan_model=hdbscan_model,
        language=LANGUAGE,
        verbose=False
    )

    try:
        topics, _ = topic_model.fit_transform(subset_chunks, embeddings)
    except Exception as e:
        print(f"Trial failed due to: {e}")
        return -1

    # Filter out outliers (-1)
    filtered_embeddings = []
    filtered_labels = []

    for topic, emb in zip(topics, embeddings):
        if topic != -1:
            filtered_labels.append(topic)
            filtered_embeddings.append(emb)

    if len(set(filtered_labels)) < 2:
        return -1

    score = silhouette_score(filtered_embeddings, filtered_labels)
    return score

# ------------------ RUN OPTUNA ------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

# ------------------ SAVE BEST PARAMETERS ------------------
best_params = study.best_params
pd.DataFrame([best_params]).to_csv("best_bertopic_hyperparameters.csv", index=False)
print("Best parameters:", best_params)
