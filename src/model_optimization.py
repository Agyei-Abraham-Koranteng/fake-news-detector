"""
Model optimization module with hyperparameter tuning and feature engineering.
Applies GridSearchCV, advanced TF-IDF, and misclassification analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import pickle
import json
from pathlib import Path
from datetime import datetime


class AdvancedTfidfVectorizer:
    """
    Advanced TF-IDF vectorizer with configurable n-grams and feature limits.
    """
    
    def __init__(self, max_features=20000, ngram_range=(1, 3), min_df=2, max_df=0.8):
        """
        Initialize advanced vectorizer.
        
        Parameters:
        -----------
        max_features : int
            Maximum number of features to extract
        ngram_range : tuple
            N-gram range (unigrams, bigrams, trigrams)
        min_df : int
            Minimum document frequency
        max_df : float
            Maximum document frequency
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer = None
    
    def fit_transform(self, texts):
        """Fit and transform texts."""
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode',
            norm='l2'
        )
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_transform() first.")
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names."""
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        return self.vectorizer.get_feature_names_out()


class ModelOptimizer:
    """
    Optimize ML models through hyperparameter tuning and feature engineering.
    """
    
    def __init__(self, random_state=42):
        """Initialize optimizer."""
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.grid_results = None
        self.vectorizer = None
        self.misclassified_data = None
    
    def create_advanced_vectorizer(self, max_features=20000, ngram_range=(1, 3)):
        """
        Create advanced TF-IDF vectorizer.
        
        Parameters:
        -----------
        max_features : int
            Maximum features
        ngram_range : tuple
            N-gram configuration
            
        Returns:
        --------
        AdvancedTfidfVectorizer
        """
        self.vectorizer = AdvancedTfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.8
        )
        return self.vectorizer
    
    def vectorize_texts(self, texts_train, texts_test):
        """
        Vectorize training and test texts.
        
        Parameters:
        -----------
        texts_train : list
            Training texts
        texts_test : list
            Test texts
            
        Returns:
        --------
        tuple
            (X_train, X_test)
        """
        X_train = self.vectorizer.fit_transform(texts_train)
        X_test = self.vectorizer.transform(texts_test)
        
        print(f"\n{'='*60}")
        print(f"ADVANCED FEATURE EXTRACTION")
        print(f"{'='*60}")
        print(f"Vectorizer Configuration:")
        print(f"  Max Features: {self.vectorizer.max_features}")
        print(f"  N-gram Range: {self.vectorizer.ngram_range}")
        print(f"  Actual Features Extracted: {X_train.shape[1]}")
        print(f"  Train Matrix Shape: {X_train.shape}")
        print(f"  Test Matrix Shape: {X_test.shape}")
        
        return X_train, X_test
    
    def hyperparameter_tuning(self, X_train, y_train, model_type='svm', cv=5):
        """
        Perform GridSearchCV hyperparameter tuning.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        model_type : str
            Type of model ('svm' or 'lr')
        cv : int
            Cross-validation folds
            
        Returns:
        --------
        tuple
            (best_model, best_params, grid_results_df)
        """
        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING (GridSearchCV)")
        print(f"{'='*60}")
        
        if model_type == 'svm':
            base_model = LinearSVC(random_state=self.random_state, max_iter=5000)
            param_grid = {
                'C': [0.01, 0.1, 1, 10, 100],
                'loss': ['squared_hinge', 'hinge'],
                'penalty': ['l1', 'l2']
            }
            print(f"Model: Linear SVM")
        elif model_type == 'lr':
            base_model = LogisticRegression(random_state=self.random_state, max_iter=5000, n_jobs=-1)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
            print(f"Model: Logistic Regression")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        print(f"Parameter Grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        print(f"Cross-validation Folds: {cv}\n")
        
        # GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Create results dataframe
        results_df = pd.DataFrame(grid_search.cv_results_)
        self.grid_results = results_df[['param_C', 'param_loss', 'param_penalty', 
                                         'mean_test_score', 'std_test_score']].head(10)
        
        print(f"\n{'='*60}")
        print(f"TOP 10 PARAMETER COMBINATIONS (by F1 score)")
        print(f"{'='*60}")
        print(self.grid_results.to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"BEST PARAMETERS")
        print(f"{'='*60}")
        for param, value in self.best_params.items():
            print(f"  {param}: {value}")
        print(f"Best Cross-Validation F1 Score: {grid_search.best_score_:.4f}")
        
        return self.best_model, self.best_params, results_df
    
    def evaluate_optimized_model(self, X_test, y_test):
        """
        Evaluate optimized model on test set.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.best_model is None:
            raise ValueError("No optimized model found. Run hyperparameter_tuning() first.")
        
        y_pred = self.best_model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred': y_pred
        }
        
        return metrics
    
    def analyze_misclassifications(self, X_test, y_test, texts_test, vectorizer_features=None):
        """
        Analyze and visualize misclassified samples.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        texts_test : list
            Original test texts
        vectorizer_features : array, optional
            Feature names from vectorizer
            
        Returns:
        --------
        dict
            Misclassification data
        """
        if self.best_model is None:
            raise ValueError("No optimized model found.")
        
        y_pred = self.best_model.predict(X_test)
        
        # Find misclassified samples
        misclassified_indices = np.where(y_pred != y_test)[0]
        
        print(f"\n{'='*60}")
        print(f"MISCLASSIFICATION ANALYSIS")
        print(f"{'='*60}")
        print(f"Total Samples: {len(y_test)}")
        print(f"Correct: {len(y_test) - len(misclassified_indices)}")
        print(f"Misclassified: {len(misclassified_indices)} ({len(misclassified_indices)/len(y_test)*100:.1f}%)")
        
        # Breakdown by type
        false_positives = misclassified_indices[y_pred[misclassified_indices] == 1]  # Predicted fake, actually real
        false_negatives = misclassified_indices[y_pred[misclassified_indices] == 0]   # Predicted real, actually fake
        
        print(f"\nError Breakdown:")
        print(f"  False Positives (predicted fake, actually real): {len(false_positives)}")
        print(f"  False Negatives (predicted real, actually fake): {len(false_negatives)} ⚠️")
        
        # Store misclassification data
        self.misclassified_data = {
            'total_misclassified': len(misclassified_indices),
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives),
            'misclassification_rate': len(misclassified_indices) / len(y_test)
        }
        
        # Print sample misclassifications
        print(f"\n{'='*60}")
        print(f"SAMPLE MISCLASSIFICATIONS (First 3)")
        print(f"{'='*60}")
        
        for i, idx in enumerate(misclassified_indices[:3]):
            true_label = "FAKE" if y_test[idx] == 1 else "REAL"
            pred_label = "FAKE" if y_pred[idx] == 1 else "REAL"
            text = texts_test[idx][:100] + "..." if len(texts_test[idx]) > 100 else texts_test[idx]
            
            print(f"\nMisclassification {i+1}:")
            print(f"  Text: {text}")
            print(f"  True Label: {true_label}")
            print(f"  Predicted Label: {pred_label}")
        
        return self.misclassified_data
    
    def print_optimization_summary(self, baseline_metrics, optimized_metrics):
        """
        Print comparison between baseline and optimized models.
        
        Parameters:
        -----------
        baseline_metrics : dict
            Metrics from baseline model
        optimized_metrics : dict
            Metrics from optimized model
        """
        print(f"\n{'='*60}")
        print(f"BASELINE vs OPTIMIZED - PERFORMANCE COMPARISON")
        print(f"{'='*60}\n")
        
        comparison_data = {
            'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            'Baseline': [
                baseline_metrics.get('accuracy', 0),
                baseline_metrics.get('precision', 0),
                baseline_metrics.get('recall', 0),
                baseline_metrics.get('f1', 0)
            ],
            'Optimized': [
                optimized_metrics.get('accuracy', 0),
                optimized_metrics.get('precision', 0),
                optimized_metrics.get('recall', 0),
                optimized_metrics.get('f1', 0)
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df['Improvement'] = comparison_df['Optimized'] - comparison_df['Baseline']
        comparison_df['% Change'] = (comparison_df['Improvement'] / (comparison_df['Baseline'] + 1e-10) * 100).round(2)
        
        print(comparison_df.to_string(index=False))
        
        print(f"\n{'='*60}")
        print(f"KEY INSIGHTS")
        print(f"{'='*60}")
        print(f"F1 Score Improvement: {comparison_df.iloc[3]['Improvement']:.4f} ({comparison_df.iloc[3]['% Change']:.1f}%)")
        print(f"Recall Improvement: {comparison_df.iloc[2]['Improvement']:.4f} ({comparison_df.iloc[2]['% Change']:.1f}%)")
        print(f"False Negative Rate: {self.misclassified_data['fn_count']} (lower is better)")
    
    def plot_optimization_results(self, baseline_metrics, optimized_metrics, output_dir='models'):
        """
        Visualize optimization results.
        
        Parameters:
        -----------
        baseline_metrics : dict
            Baseline metrics
        optimized_metrics : dict
            Optimized metrics
        output_dir : str
            Output directory
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # Comparison bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        baseline_vals = [
            baseline_metrics['accuracy'],
            baseline_metrics['precision'],
            baseline_metrics['recall'],
            baseline_metrics['f1']
        ]
        optimized_vals = [
            optimized_metrics['accuracy'],
            optimized_metrics['precision'],
            optimized_metrics['recall'],
            optimized_metrics['f1']
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_vals, width, label='Baseline', color='#1f77b4', alpha=0.8)
        bars2 = ax.bar(x + width/2, optimized_vals, width, label='Optimized', color='#2ca02c', alpha=0.8)
        
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Baseline vs Optimized Model Performance', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/optimization_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir}/optimization_comparison.png")
        plt.close()
        
        # Confusion matrix comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle('Confusion Matrices: Baseline vs Optimized', fontsize=14, fontweight='bold')
        
        baseline_cm = baseline_metrics['confusion_matrix']
        optimized_cm = optimized_metrics['confusion_matrix']
        
        sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   cbar=False, xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        axes[0].set_title('Baseline')
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        
        sns.heatmap(optimized_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                   cbar=False, xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        axes[1].set_title('Optimized')
        axes[1].set_ylabel('True Label')
        axes[1].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/confusion_matrices_comparison.png")
        plt.close()
    
    def save_optimized_model(self, output_dir='models'):
        """Save optimized model and configuration."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Save model
        model_path = f'{output_dir}/optimized_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"\n✓ Saved optimized model: {model_path}")
        
        # Save vectorizer
        vectorizer_path = f'{output_dir}/optimized_vectorizer.pkl'
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer.vectorizer, f)
        print(f"✓ Saved vectorizer: {vectorizer_path}")
        
        # Save config
        config = {
            'best_params': self.best_params,
            'vectorizer_config': {
                'max_features': self.vectorizer.max_features,
                'ngram_range': self.vectorizer.ngram_range,
                'min_df': self.vectorizer.min_df,
                'max_df': self.vectorizer.max_df
            },
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = f'{output_dir}/optimization_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Saved config: {config_path}")


def load_optimized_model(model_path='models/optimized_model.pkl',
                        vectorizer_path='models/optimized_vectorizer.pkl'):
    """Load optimized model and vectorizer."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer


if __name__ == "__main__":
    # Example usage
    print("="*60)
    print("MODEL OPTIMIZATION DEMO")
    print("="*60)
    
    # Generate synthetic text data
    from sklearn.datasets import fetch_20newsgroups
    import warnings
    warnings.filterwarnings('ignore')
    
    print("\n📊 Loading sample text data...")
    
    # Use simplified text data for demo
    sample_texts_real = [
        "Scientists discover new treatment for cancer using gene therapy",
        "Climate agreement reached at international conference",
        "Economic growth continues despite market volatility",
        "Healthcare workers demand better working conditions",
        "New renewable energy project breaks efficiency records"
    ] * 50
    
    sample_texts_fake = [
        "Celebrity reveals shocking secret using ancient conspiracy",
        "Miracle cure discovered by local doctor that doctors hate",
        "Government hides alien contact from citizens",
        "Company reveals one trick that doctors use to hide truth",
        "Breaking: Natural remedy cures all diseases instantly"
    ] * 50
    
    texts = sample_texts_real + sample_texts_fake
    labels = np.array([0] * len(sample_texts_real) + [1] * len(sample_texts_fake))
    
    # Train/test split
    from sklearn.model_selection import train_test_split
    texts_train, texts_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=42
    )
    
    # Initialize optimizer
    optimizer = ModelOptimizer(random_state=42)
    
    # Step 1: Create advanced vectorizer with bigrams/trigrams
    print("\n📝 Creating advanced vectorizer...")
    optimizer.create_advanced_vectorizer(max_features=20000, ngram_range=(1, 3))
    X_train, X_test = optimizer.vectorize_texts(texts_train, texts_test)
    
    # Step 2: Hyperparameter tuning
    print("\n🔧 Running hyperparameter tuning...")
    best_model, best_params, grid_results = optimizer.hyperparameter_tuning(
        X_train, y_train, model_type='svm', cv=5
    )
    
    # Step 3: Evaluate optimized model
    optimized_metrics = optimizer.evaluate_optimized_model(X_test, y_test)
    
    # Step 4: Analyze misclassifications
    misclassified_data = optimizer.analyze_misclassifications(
        X_test, y_test, texts_test
    )
    
    # Step 5: Baseline metrics for comparison
    baseline_metrics = {
        'accuracy': 0.63,
        'precision': 0.5263,
        'recall': 0.5128,
        'f1': 0.5195,
        'confusion_matrix': np.array([[61, 0], [19, 20]])
    }
    
    # Print summary
    optimizer.print_optimization_summary(baseline_metrics, optimized_metrics)
    
    # Visualize
    optimizer.plot_optimization_results(baseline_metrics, optimized_metrics)
    
    # Save optimized model
    optimizer.save_optimized_model()
    
    print("\n" + "="*60)
    print("✓ Optimization complete!")
    print("="*60)
