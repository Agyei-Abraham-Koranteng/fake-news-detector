"""
Model training and evaluation module for fake news detection.
Trains 4 baseline models and compares their performance.
"""

import pickle
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime


class ModelTrainer:
    """
    Train and evaluate multiple baseline models for fake news detection.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize model trainer.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
        # Initialize baseline models
        self.models_config = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1
            ),
            'Naïve Bayes': MultinomialNB(),
            'Linear SVM': LinearSVC(
                max_iter=2000,
                random_state=random_state,
                loss='squared_hinge'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=-1,
                max_depth=15
            )
        }
    
    def split_data(self, X, y, test_size=0.2, stratify=True):
        """
        Split data into train and test sets with stratification.
        
        Parameters:
        -----------
        X : array-like
            Feature matrix
        y : array-like
            Target labels
        test_size : float
            Proportion of data for testing
        stratify : bool
            Whether to stratify by labels
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        stratify_arg = y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=stratify_arg,
            random_state=self.random_state
        )
        
        print(f"\n{'='*60}")
        print(f"DATA SPLIT")
        print(f"{'='*60}")
        print(f"Train set size: {self.X_train.shape[0]} samples")
        print(f"Test set size:  {self.X_test.shape[0]} samples")
        print(f"Feature dimensionality: {self.X_train.shape[1]}")
        if stratify:
            print(f"\nClass distribution:")
            print(f"  Train: {np.bincount(self.y_train)}")
            print(f"  Test:  {np.bincount(self.y_test)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, model_name, model):
        """
        Train a single model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : sklearn estimator
            Model to train
            
        Returns:
        --------
        tuple
            (trained_model, training_time)
        """
        print(f"\nTraining {model_name}...", end=" ", flush=True)
        
        start_time = datetime.now()
        model.fit(self.X_train, self.y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✓ ({training_time:.2f}s)")
        return model, training_time
    
    def evaluate_model(self, model_name, model, training_time):
        """
        Evaluate model on test set.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        model : trained sklearn estimator
            Trained model
        training_time : float
            Training time in seconds
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # For binary classification, calculate ROC-AUC if possible
        try:
            # Get probability predictions if available
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(self.X_test)[:, 1]
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(self.X_test)
                # Normalize to [0, 1]
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
            else:
                y_proba = None
            
            roc_auc = roc_auc_score(self.y_test, y_proba) if y_proba is not None else None
        except:
            roc_auc = None
        
        # Cross-validation score
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='f1')
        
        results = {
            'model_name': model_name,
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std(),
            'training_time': training_time,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        
        return results
    
    def train_all_models(self):
        """
        Train and evaluate all baseline models.
        
        Returns:
        --------
        dict
            Dictionary with results for all models
        """
        print(f"\n{'='*60}")
        print(f"TRAINING BASELINE MODELS")
        print(f"{'='*60}")
        
        for model_name, model in self.models_config.items():
            # Train
            trained_model, training_time = self.train_model(model_name, model)
            
            # Evaluate
            results = self.evaluate_model(model_name, trained_model, training_time)
            self.results[model_name] = results
            
            # Store trained model
            self.models[model_name] = trained_model
        
        return self.results
    
    def print_results_summary(self):
        """Print comprehensive results summary."""
        print(f"\n{'='*60}")
        print(f"MODEL EVALUATION RESULTS")
        print(f"{'='*60}\n")
        
        # Create results dataframe
        results_data = []
        for model_name, results in self.results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1'],
                'ROC-AUC': results['roc_auc'],
                'CV F1 (mean±std)': f"{results['cv_f1_mean']:.4f}±{results['cv_f1_std']:.4f}",
                'Training Time (s)': results['training_time']
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Pretty print
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(results_df.to_string(index=False))
        
        # Find best model
        self.best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1'])
        self.best_model = self.results[self.best_model_name]['model']
        
        print(f"\n{'='*60}")
        print(f"🏆 BEST MODEL: {self.best_model_name}")
        print(f"{'='*60}")
        print(f"Accuracy: {self.results[self.best_model_name]['accuracy']:.4f}")
        print(f"Precision: {self.results[self.best_model_name]['precision']:.4f}")
        print(f"Recall: {self.results[self.best_model_name]['recall']:.4f} ⭐ (Important for fake detection)")
        print(f"F1 Score: {self.results[self.best_model_name]['f1']:.4f}")
        
        # Detailed classification report
        print(f"\n{'='*60}")
        print(f"DETAILED CLASSIFICATION REPORT (Best Model)")
        print(f"{'='*60}")
        y_pred_best = self.results[self.best_model_name]['y_pred']
        print(classification_report(self.y_test, y_pred_best, 
                                   target_names=['Real', 'Fake']))
    
    def plot_results(self, output_dir='models'):
        """
        Create visualization of model comparison.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save plots
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1. Metrics Comparison Bar Chart
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        
        for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax = axes[idx // 2, idx % 2]
            model_names = list(self.results.keys())
            values = [self.results[m][metric] for m in model_names]
            
            bars = ax.bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax.set_ylabel(label, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.set_title(label)
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved: {output_dir}/model_comparison.png")
        plt.close()
        
        # 2. Confusion Matrix for Best Model
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Confusion Matrices - All Models', fontsize=16, fontweight='bold')
        
        for idx, (model_name, results) in enumerate(self.results.items()):
            ax = axes[idx // 2, idx % 2]
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       cbar=False, xticklabels=['Real', 'Fake'],
                       yticklabels=['Real', 'Fake'])
            ax.set_title(f'{model_name}\n(F1: {results["f1"]:.3f})')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/confusion_matrices.png")
        plt.close()
        
        # 3. ROC Curves
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for model_name, results in self.results.items():
            model = results['model']
            
            try:
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(self.X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba = model.decision_function(self.X_test)
                    y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
                else:
                    continue
                
                fpr, tpr, _ = roc_curve(self.y_test, y_proba)
                auc = results['roc_auc']
                
                if auc is not None:
                    ax.plot(fpr, tpr, label=f'{model_name} (AUC: {auc:.3f})', linewidth=2)
            except:
                pass
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax.set_xlabel('False Positive Rate', fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontweight='bold')
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/roc_curves.png")
        plt.close()
    
    def save_best_model(self, output_dir='models'):
        """
        Save best model and metadata.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save model
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        if self.best_model is None:
            print("No model has been trained yet.")
            return
        
        # Save model
        model_path = f'{output_dir}/best_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"\n✓ Saved best model: {model_path}")
        
        # Save metadata
        metadata = {
            'model_name': self.best_model_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'accuracy': float(self.results[self.best_model_name]['accuracy']),
                'precision': float(self.results[self.best_model_name]['precision']),
                'recall': float(self.results[self.best_model_name]['recall']),
                'f1': float(self.results[self.best_model_name]['f1']),
                'roc_auc': float(self.results[self.best_model_name]['roc_auc']) if self.results[self.best_model_name]['roc_auc'] is not None else None
            },
            'data_split': {
                'train_size': int(self.X_train.shape[0]),
                'test_size': int(self.X_test.shape[0]),
                'feature_count': int(self.X_train.shape[1])
            }
        }
        
        metadata_path = f'{output_dir}/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_path}")
    
    def save_results_to_csv(self, output_dir='models'):
        """
        Save results summary to CSV.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        results_data = []
        for model_name, results in self.results.items():
            results_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1_Score': results['f1'],
                'ROC_AUC': results['roc_auc'],
                'CV_F1_Mean': results['cv_f1_mean'],
                'CV_F1_Std': results['cv_f1_std'],
                'Training_Time_Seconds': results['training_time']
            })
        
        results_df = pd.DataFrame(results_data)
        csv_path = f'{output_dir}/model_results.csv'
        results_df.to_csv(csv_path, index=False)
        print(f"✓ Saved results CSV: {csv_path}")


def load_best_model(model_path='models/best_model.pkl'):
    """
    Load saved best model.
    
    Parameters:
    -----------
    model_path : str
        Path to saved model
        
    Returns:
    --------
    trained model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("="*60)
    print("FAKE NEWS DETECTION - MODEL TRAINING DEMO")
    print("="*60)
    
    # Generate synthetic data (in practice, use real data)
    print("\n📊 Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=1000,
        n_informative=800,
        n_redundant=100,
        random_state=42,
        n_classes=2,
        weights=[0.6, 0.4]
    )
    
    # Convert to non-negative features (normalized between 0-1)
    # This is needed for MultinomialNB which expects non-negative counts
    X = (X - X.min()) / (X.max() - X.min() + 1e-10)
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Split data
    trainer.split_data(X, y, test_size=0.2, stratify=True)
    
    # Train all models
    trainer.train_all_models()
    
    # Print results
    trainer.print_results_summary()
    
    # Plot results
    trainer.plot_results(output_dir='models')
    
    # Save best model
    trainer.save_best_model(output_dir='models')
    trainer.save_results_to_csv(output_dir='models')
    
    print("\n" + "="*60)
    print("✓ Training pipeline complete!")
    print("="*60)
