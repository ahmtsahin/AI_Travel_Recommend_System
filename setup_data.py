import os
from pathlib import Path
import pandas as pd
import pickle
import shutil

def create_data_directories():
    """Create necessary data directories if they don't exist."""
    directories = [
        'data',
        'data/features_chunks',
        'data/faiss_index',
        'assets'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    print("✓ Created data directories")

def split_features_file(input_file):
    """Split the large features file into smaller chunks."""
    try:
        print(f"Reading features file: {input_file}")
        df = pd.read_pickle(input_file)
        
        # Calculate chunks
        chunk_size = 20 * 1024 * 1024  # 20MB in bytes
        total_size = df.memory_usage(deep=True).sum()
        num_chunks = (total_size // chunk_size) + 1
        chunk_size = len(df) // num_chunks + 1
        
        # Split and save chunks
        for i, start_idx in enumerate(range(0, len(df), chunk_size)):
            chunk = df.iloc[start_idx:start_idx + chunk_size]
            output_file = f"data/features_chunks/features_chunk_{i}.pkl"
            chunk.to_pickle(output_file)
            print(f"✓ Saved chunk {i+1}/{num_chunks}")
        
        # Save metadata
        metadata = {
            'num_chunks': num_chunks,
            'total_rows': len(df),
            'columns': df.columns.tolist()
        }
        with open('data/features_chunks/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        print("✓ Saved metadata")
        
    except Exception as e:
        print(f"Error splitting features file: {str(e)}")
        raise

def copy_required_files():
    """Copy or move required files to the correct locations."""
    required_files = {
        'image_features_model_2.pkl': 'data/',
        'combined.csv': 'data/',
        'styles.css': 'assets/'
    }
    
    for file, dest in required_files.items():
        if os.path.exists(file):
            shutil.copy2(file, os.path.join(dest, file))
            print(f"✓ Copied {file} to {dest}")
        else:
            print(f"⚠️ Warning: {file} not found in current directory")

def main():
    print("Setting up data directories and files...")
    try:
        # Create directories
        create_data_directories()
        
        # Copy required files
        copy_required_files()
        
        # Split features file if it exists
        if os.path.exists('image_features_model_2.pkl'):
            print("\nSplitting features file...")
            split_features_file('image_features_model_2.pkl')
        else:
            print("\n⚠️ Warning: image_features_model_2.pkl not found")
        
        print("\nSetup complete! You can now run the Streamlit app.")
        print("\nMake sure you have:")
        print("1. combined.csv in the data/ directory")
        print("2. styles.css in the assets/ directory")
        print("3. faiss_index files in the data/faiss_index/ directory")
        
    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")
        print("Please fix the error and try again.")

if __name__ == "__main__":
    main()