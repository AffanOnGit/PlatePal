import os
import pandas as pd
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def build_index(csv_path, output_dir, sample_size=20000):
    print(f"--- PlatePal: Building Semantic RAG Index (Sample: {sample_size}) ---")
    
    # 1. Load Data
    print(f"Loading {csv_path}...")
    # Use chunking or just read a subset to save RAM
    df = pd.read_csv(csv_path, nrows=sample_size * 2) # Read more to allow filtering
    
    # Clean data: remove empty titles or directions
    df = df.dropna(subset=['title', 'ingredients', 'directions'])
    df = df.head(sample_size)
    
    print(f"Processing {len(df)} high-quality recipes...")
    
    # 2. Prepare Semantic Text (What we will search against)
    # We combine title and ingredients for the best semantic match
    df['semantic_text'] = df['title'] + " " + df['ingredients']
    
    # 3. Compute Embeddings
    model_name = 'all-MiniLM-L6-v2'
    print(f"Initializing embedding model: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(df['semantic_text'].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    
    # 4. Create FAISS Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
    index.add(embeddings)
    
    # 5. Save Artifacts
    os.makedirs(output_dir, exist_ok=True)
    
    # Save index
    index_path = os.path.join(output_dir, "recipe_index.faiss")
    faiss.write_index(index, index_path)
    
    # Save the recipe data (mapping)
    # We only save what we need for the UI to save space
    recipes_data = df[['title', 'ingredients', 'directions']].to_dict('records')
    data_path = os.path.join(output_dir, "recipe_data.pkl")
    with open(data_path, 'wb') as f:
        pickle.dump(recipes_data, f)
        
    print(f"\n--- SUCCESS ---")
    print(f"Index saved to: {index_path}")
    print(f"Data saved to: {data_path}")
    print(f"Total size: ~{os.path.getsize(index_path)/1024/1024:.1f} MB")

if __name__ == "__main__":
    CSV_PATH = "data/raw/RecipeNLG_dataset.csv"
    OUTPUT_DIR = "data/rag_index"
    
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Dataset not found at {CSV_PATH}")
    else:
        build_index(CSV_PATH, OUTPUT_DIR)
