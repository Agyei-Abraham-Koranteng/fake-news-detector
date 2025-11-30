"""
Text preprocessing module for NLP pipeline.
Provides modular functions for cleaning, tokenizing, lemmatizing, and vectorizing text.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    """
    Professional text preprocessing pipeline for NLP tasks.
    Handles cleaning, tokenization, lemmatization, and vectorization.
    """
    
    def __init__(self):
        """Initialize preprocessing tools."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = None
    
    def clean_text(self, text):
        """
        Clean text by removing URLs, HTML tags, numbers, and converting to lowercase.
        
        Parameters:
        -----------
        text : str
            Input text to clean
            
        Returns:
        --------
        str
            Cleaned text in lowercase
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_punctuation(self, text):
        """
        Remove punctuation from text.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        str
            Text without punctuation
        """
        # Keep only alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    
    def tokenize_text(self, text):
        """
        Tokenize text into words.
        
        Parameters:
        -----------
        text : str
            Input text
            
        Returns:
        --------
        list
            List of tokens (words)
        """
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from token list.
        
        Parameters:
        -----------
        tokens : list
            List of tokens
            
        Returns:
        --------
        list
            Filtered tokens without stopwords
        """
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return filtered_tokens
    
    def lemmatize_text(self, tokens):
        """
        Lemmatize tokens to their base form.
        
        Parameters:
        -----------
        tokens : list
            List of tokens
            
        Returns:
        --------
        list
            Lemmatized tokens
        """
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
    
    def preprocess_pipeline(self, text):
        """
        Complete preprocessing pipeline: clean → tokenize → remove stopwords → lemmatize.
        
        Parameters:
        -----------
        text : str
            Raw input text
            
        Returns:
        --------
        list
            Cleaned and lemmatized tokens
        """
        # Step 1: Clean text
        cleaned = self.clean_text(text)
        
        # Step 2: Remove punctuation
        cleaned = self.remove_punctuation(cleaned)
        
        # Step 3: Tokenize
        tokens = self.tokenize_text(cleaned)
        
        # Step 4: Remove stopwords
        tokens = self.remove_stopwords(tokens)
        
        # Step 5: Lemmatize
        tokens = self.lemmatize_text(tokens)
        
        return tokens
    
    def preprocess_batch(self, texts):
        """
        Preprocess multiple texts.
        
        Parameters:
        -----------
        texts : list
            List of text strings
            
        Returns:
        --------
        list
            List of preprocessed token lists
        """
        return [self.preprocess_pipeline(text) for text in texts]
    
    def tokens_to_string(self, tokens):
        """
        Convert token list back to string (for vectorization).
        
        Parameters:
        -----------
        tokens : list
            List of tokens
            
        Returns:
        --------
        str
            Tokens joined as string
        """
        return ' '.join(tokens)
    
    def tfidf_vectorize(self, texts, max_features=5000, min_df=2, max_df=0.8):
        """
        Vectorize texts using TF-IDF.
        
        Parameters:
        -----------
        texts : list
            List of text strings (raw or preprocessed)
        max_features : int, default=5000
            Maximum number of features to extract
        min_df : int, default=2
            Minimum document frequency
        max_df : float, default=0.8
            Maximum document frequency as proportion
            
        Returns:
        --------
        tuple
            (tfidf_matrix, feature_names, vectorizer_object)
        """
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            stop_words='english',
            lowercase=True,
            strip_accents='unicode'
        )
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        return tfidf_matrix, feature_names, self.vectorizer
    
    def transform_with_vectorizer(self, texts):
        """
        Transform new texts using fitted vectorizer.
        
        Parameters:
        -----------
        texts : list
            List of text strings
            
        Returns:
        --------
        sparse matrix
            TF-IDF vectors
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call tfidf_vectorize() first.")
        return self.vectorizer.transform(texts)
    
    def get_top_features(self, tfidf_matrix, feature_names, n=10):
        """
        Get top TF-IDF features.
        
        Parameters:
        -----------
        tfidf_matrix : sparse matrix
            TF-IDF matrix
        feature_names : array
            Feature names
        n : int, default=10
            Number of top features to return
            
        Returns:
        --------
        list
            Top n features sorted by importance
        """
        # Sum TF-IDF scores across documents
        tfidf_scores = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        
        # Get indices of top n features
        top_indices = tfidf_scores.argsort()[-n:][::-1]
        
        return [(feature_names[i], tfidf_scores[i]) for i in top_indices]


# Convenience functions for simple usage
def clean_text(text):
    """Standalone function to clean text."""
    preprocessor = TextPreprocessor()
    return preprocessor.clean_text(text)


def tokenize_text(text):
    """Standalone function to tokenize text."""
    preprocessor = TextPreprocessor()
    return preprocessor.tokenize_text(text)


def lemmatize_text(tokens):
    """Standalone function to lemmatize tokens."""
    preprocessor = TextPreprocessor()
    return preprocessor.lemmatize_text(tokens)


def tfidf_vectorize(texts, max_features=5000, min_df=2, max_df=0.8):
    """Standalone function to vectorize texts."""
    preprocessor = TextPreprocessor()
    return preprocessor.tfidf_vectorize(texts, max_features, min_df, max_df)


def preprocess_pipeline(text):
    """Standalone function for complete preprocessing."""
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess_pipeline(text)


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("TEXT PREPROCESSING DEMO")
    print("=" * 60)
    
    # Sample texts
    sample_texts = [
        "Check out this link: https://example.com/article - 5G is the future! #tech",
        "Breaking news! Visit www.news.com for updates on climate change.",
        "Machine Learning 101: Learn ML at example.org. 2024's best practices!"
    ]
    
    preprocessor = TextPreprocessor()
    
    print("\n1. Original Texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"   {i}. {text}")
    
    print("\n2. After Cleaning:")
    cleaned_texts = [preprocessor.clean_text(text) for text in sample_texts]
    for i, text in enumerate(cleaned_texts, 1):
        print(f"   {i}. {text}")
    
    print("\n3. After Full Preprocessing Pipeline:")
    for i, text in enumerate(sample_texts, 1):
        tokens = preprocessor.preprocess_pipeline(text)
        print(f"   {i}. {tokens}")
    
    print("\n4. TF-IDF Vectorization:")
    tfidf_matrix, features, vectorizer = preprocessor.tfidf_vectorize(sample_texts, max_features=20)
    print(f"   TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    print(f"   Total Features: {len(features)}")
    
    print("\n5. Top 5 Features by TF-IDF Score:")
    top_features = preprocessor.get_top_features(tfidf_matrix, features, n=5)
    for feature, score in top_features:
        print(f"   {feature}: {score:.4f}")
    
    print("\n" + "=" * 60)
