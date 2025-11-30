import pandas as pd
import numpy as np
import os
from src.preprocess import TextPreprocessor
from src.model_training import ModelTrainer
from src.model_optimization import ModelOptimizer

def load_and_prepare_data():
    """Load data, add labels, and combine."""
    print("Loading data...")
    
    # Check if files exist
    fake_path = 'data/raw/fake_news/Fake.csv'
    true_path = 'data/raw/real_news/True.csv'
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        # Fallback for different folder structure if needed, or error
        # Trying direct path if subfolders don't exist
        fake_path = 'data/raw/Fake.csv'
        true_path = 'data/raw/True.csv'
        
    try:
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
    except FileNotFoundError:
        print("Error: Could not find Fake.csv or True.csv in data/raw/")
        return None, None

    # Add labels: 1 for Fake, 0 for Real
    df_fake['label'] = 1
    df_true['label'] = 0

    # Combine
    df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # We only need text and label
    # Some datasets have 'text' column, others might have 'title' + 'text'
    # Let's combine title and text if available
    if 'title' in df.columns and 'text' in df.columns:
        df['content'] = df['title'] + " " + df['text']
    elif 'text' in df.columns:
        df['content'] = df['text']
    else:
        print("Error: Could not find 'text' column")
        return None, None

    print(f"Data loaded. Total samples: {len(df)}")
    print(f"Fake: {len(df[df['label']==1])}, Real: {len(df[df['label']==0])}")
    
    return df['content'].values, df['label'].values

def main():
    # 1. Load Data
    X, y = load_and_prepare_data()
    if X is None:
        return

    # 2. Preprocessing & Vectorization
    print("\nPreprocessing and Vectorizing...")
    # We'll use the optimizer's vectorizer as it's more configurable
    optimizer = ModelOptimizer(random_state=42)
    
    # Split first to avoid data leakage
    from sklearn.model_selection import train_test_split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Create and fit vectorizer
    # Using slightly lower max_features for speed on local machine if needed, but 20000 is standard
    optimizer.create_advanced_vectorizer(max_features=10000, ngram_range=(1, 2))
    
    X_train = optimizer.vectorizer.fit_transform(X_train_raw)
    X_test = optimizer.vectorizer.transform(X_test_raw)
    
    print(f"Training features shape: {X_train.shape}")

    # 3. Train Baseline Models
    trainer = ModelTrainer(random_state=42)
    # Manually set the split data since we already vectorized it
    trainer.X_train = X_train
    trainer.X_test = X_test
    trainer.y_train = y_train
    trainer.y_test = y_test
    
    print("\nTraining models...")
    results = trainer.train_all_models()
    trainer.print_results_summary()
    
    # 4. Save Best Model
    print("\nSaving best model...")
    trainer.save_best_model(output_dir='models')
    
    # Also save the vectorizer so we can use it in the app
    import pickle
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(optimizer.vectorizer.vectorizer, f)
    print("Saved vectorizer to models/vectorizer.pkl")

if __name__ == "__main__":
    main()
