import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

PASSAGES_FILE = '../data/pericopes_with_text.csv'
EMBEDDINGS_OUTPUT_FILE = '../data/pericopes_embeddings.npy'

def main():
    # Load the passages data from a CSV file
    passages_df = pd.read_csv(PASSAGES_FILE)
    print(f"Loaded {len(passages_df)} passages.")

    # Initialize the sentence transformer model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Create embeddings for each passage
    embeddings = model.encode(passages_df['Passage Text'].tolist(), show_progress_bar=True, batch_size=32)

    # Add embeddings to the DataFrame
    passages_df['Embedding'] = embeddings.tolist()

    # Save the DataFrame with embeddings to a new CSV file
    np.save(EMBEDDINGS_OUTPUT_FILE, embeddings)

if __name__ == "__main__":
    main()