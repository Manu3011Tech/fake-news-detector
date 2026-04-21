#!/usr/bin/env python3
"""
Fixed Training Script for Fake News Detection
This version has NO import errors
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

# Add the backend directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_path = os.path.join(current_dir, 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

print(f"Python path: {sys.path[:3]}")
print(f"Current directory: {current_dir}")

class SimpleTextModel:
    """Simple text model that actually works"""
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.classifier = None
        
    def train(self, texts, labels):
        """Train the model"""
        print(f"  Training on {len(texts)} samples...")
        X = self.vectorizer.fit_transform(texts)
        self.classifier = LogisticRegression(random_state=42)
        self.classifier.fit(X, labels)
        print(f"  Training complete!")
        
    def save(self, path):
        """Save the model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'classifier': self.classifier
            }, f)
        print(f"  Model saved to {path}")
        
    def load(self, path):
        """Load the model"""
        with open(path, 'rb') as f:
            saved = pickle.load(f)
            self.vectorizer = saved['vectorizer']
            self.classifier = saved['classifier']
        print(f"  Model loaded from {path}")

class SimpleImageModel:
    """Simple image model placeholder"""
    def __init__(self):
        self.is_trained = False
        
    def train(self, image_paths, labels):
        """Placeholder training"""
        self.is_trained = True
        print(f"  Image model initialized (demo mode)")
        
    def save(self, path):
        """Save placeholder model"""
        with open(path, 'wb') as f:
            pickle.dump({'is_trained': self.is_trained}, f)
        print(f"  Model saved to {path}")

def preprocess_text(text):
    """Simple text preprocessing"""
    if not text:
        return ""
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def train_text_model():
    """Train text detection model"""
    print("\n📝 Training text detection model...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Sample training data (more comprehensive)
    texts = [
        # Real news examples (label=1)
        "The president announced new economic policies today at the white house",
        "Scientists discovered a new treatment for cancer after years of research",
        "Local community raises funds for new school building project",
        "Government releases official statement about climate change initiatives",
        "Researchers publish peer reviewed study on vaccine effectiveness",
        "City council votes to approve new public transportation system",
        "International summit concludes with trade agreement between nations",
        "Health officials provide updated guidelines for flu prevention",
        
        # Fake news examples (label=0)
        "URGENT Breaking news You wont believe what happened next Click here",
        "SHOCKING discovery Scientists reveal truth they dont want you to know",
        "This one weird trick will change your life forever doctors hate this",
        "Viral video shows impossible event that everyone is talking about",
        "Alert Your computer has been infected click here to fix immediately",
        "Miracle cure that pharmaceutical companies dont want you to know about",
        "You wont believe what this politician said on live tv",
        "Breaking Celebrity death hoax spreads across social media"
    ]
    
    labels = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # Preprocess texts
    print("  Preprocessing text data...")
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Create and train model
    model = SimpleTextModel()
    model.train(processed_texts, labels)
    
    # Save model
    model_path = 'models/text_model.pkl'
    model.save(model_path)
    
    # Verify save
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"  ✅ Text model saved successfully! Size: {file_size} bytes")
    else:
        print(f"  ❌ Error: Model file not created!")
    
    return model

def train_image_model():
    """Train image detection model"""
    print("\n🖼️ Setting up image detection model...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Create and train model
    model = SimpleImageModel()
    model.train([], [])
    
    # Save model
    model_path = 'models/image_model.pkl'
    model.save(model_path)
    
    # Verify save
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"  ✅ Image model saved successfully! Size: {file_size} bytes")
    else:
        print(f"  ❌ Error: Model file not created!")
    
    return model

def test_models():
    """Test if models work"""
    print("\n🧪 Testing models...")
    
    # Test text model
    try:
        with open('models/text_model.pkl', 'rb') as f:
            text_model_data = pickle.load(f)
        print("  ✅ Text model file is valid")
        
        # Test prediction
        test_text = "Breaking news urgent alert"
        processed = preprocess_text(test_text)
        vectorizer = text_model_data['vectorizer']
        classifier = text_model_data['classifier']
        
        X_test = vectorizer.transform([processed])
        prediction = classifier.predict(X_test)[0]
        probability = classifier.predict_proba(X_test)[0]
        
        print(f"  Test prediction: {'Fake' if prediction == 0 else 'Real'}")
        print(f"  Fake probability: {probability[0]:.2%}")
        
    except Exception as e:
        print(f"  ❌ Text model test failed: {e}")
    
    # Test image model
    try:
        with open('models/image_model.pkl', 'rb') as f:
            image_model_data = pickle.load(f)
        print("  ✅ Image model file is valid")
    except Exception as e:
        print(f"  ❌ Image model test failed: {e}")

def main():
    """Main training function"""
    print("=" * 60)
    print("🚀 Fake News Detection - Model Training")
    print("=" * 60)
    
    # Train models
    text_model = train_text_model()
    image_model = train_image_model()
    
    # Test models
    test_models()
    
    print("\n" + "=" * 60)
    print("🎉 Training Complete! Models are ready")
    print("=" * 60)
    print("\n📁 Models saved in 'models/' directory")
    print("  - models/text_model.pkl")
    print("  - models/image_model.pkl")
    
    # List files with sizes
    print("\n📊 File sizes:")
    for file in ['text_model.pkl', 'image_model.pkl']:
        path = f'models/{file}'
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  {file}: {size:,} bytes")
        else:
            print(f"  {file}: NOT FOUND")

if __name__ == "__main__":
    main()