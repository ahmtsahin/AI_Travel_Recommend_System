import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def split_features_file(input_file, output_dir, chunk_size=20):
    """
    Split a large pickle file into smaller chunks.
    
    Args:
        input_file (str): Path to the input pickle file
        output_dir (str): Directory to save the chunks
        chunk_size (int): Size of each chunk in MB
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load the DataFrame
    df = pd.read_pickle(input_file)
    
    # Calculate number of rows per chunk
    chunk_bytes = chunk_size * 1024 * 1024  # Convert MB to bytes
    total_size = df.memory_usage(deep=True).sum()
    rows_per_chunk = int(len(df) * (chunk_bytes / total_size))
    
    # Split into chunks
    chunks = [df[i:i + rows_per_chunk] for i in range(0, len(df), rows_per_chunk)]
    
    # Save chunks
    for i, chunk in enumerate(chunks):
        output_file = f"{output_dir}/features_chunk_{i}.pkl"
        chunk.to_pickle(output_file)
    
    # Save metadata
    metadata = {
        'num_chunks': len(chunks),
        'total_rows': len(df),
        'columns': df.columns.tolist()
    }
    with open(f"{output_dir}/metadata.pkl", 'wb') as f:
        pickle.dump(metadata, f)

def load_features(chunks_dir):
    """
    Load features from multiple chunk files.
    
    Args:
        chunks_dir (str): Directory containing the chunk files
    
    Returns:
        pandas.DataFrame: Combined DataFrame
    """
    # Load metadata
    with open(f"{chunks_dir}/metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    # Load and combine chunks
    chunks = []
    for i in range(metadata['num_chunks']):
        chunk_file = f"{chunks_dir}/features_chunk_{i}.pkl"
        chunk = pd.read_pickle(chunk_file)
        chunks.append(chunk)
    
    # Combine all chunks
    return pd.concat(chunks, ignore_index=True)

if __name__ == "__main__":
    # Example usage
    split_features_file(
        'data/image_features_model_2.pkl',
        'data/features_chunks',
        chunk_size=20
    )