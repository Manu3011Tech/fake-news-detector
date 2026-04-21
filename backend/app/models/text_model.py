import pickle
import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class TextFraudDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.is_trained = False
        self.max_length = 5000  # Maximum characters to process
        self.chunk_size = 1000  # Process in chunks for long text
        
    def load_model(self):
        """Load pre-trained model from file"""
        possible_paths = [
            'models/text_model.pkl',
            '../models/text_model.pkl',
            'C:/Users/Dell/OneDrive/Desktop/fake-news-detector/models/text_model.pkl',
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    print(f"[INFO] Loading model from: {path}")
                    with open(path, 'rb') as f:
                        saved = pickle.load(f)
                        self.vectorizer = saved['vectorizer']
                        self.model = saved['classifier']
                        self.is_trained = True
                        print(f"[SUCCESS] Model loaded successfully!")
                        return True
                except Exception as e:
                    print(f"[ERROR] Error loading from {path}: {e}")
                    continue
        
        print("[WARNING] Could not load model from any path")
        return False
    
    def preprocess(self, text):
        """Preprocess text for prediction"""
        if not text:
            return ""
        
        # Limit text length to avoid memory issues
        if len(text) > self.max_length:
            text = text[:self.max_length]
            print(f"[INFO] Text truncated to {self.max_length} characters")
        
        # Clean the text
        text = text.lower()
        # Remove special characters but keep structure
        text = re.sub(r'[^a-zA-Z\s\.\,\!\?\']', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def split_into_chunks(self, text, chunk_size=1000):
        """Split long text into smaller chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        
        return chunks
    
    def predict_long_text(self, text):
        """Handle long text by analyzing in chunks"""
        chunks = self.split_into_chunks(text)
        
        if len(chunks) == 1:
            # Short text, analyze directly
            return self.predict_single(text)
        
        print(f"[INFO] Text split into {len(chunks)} chunks for analysis")
        
        # Analyze each chunk
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"[INFO] Analyzing chunk {i+1}/{len(chunks)}")
            result = self.predict_single(chunk)
            chunk_results.append(result)
        
        # Combine results from all chunks
        return self.combine_chunk_results(chunk_results, text)
    
    def predict_single(self, text):
        """Predict a single text chunk"""
        if not self.is_trained:
            if not self.load_model():
                self._emergency_train()
        
        if not self.is_trained:
            return {
                'prediction': 0.5,
                'class': 'Uncertain',
                'fake_score': 0.5,
                'confidence': 0.5,
                'reasoning': "Model not loaded.",
                'chunk_index': 0
            }
        
        try:
            processed_text = self.preprocess(text)
            
            # Check if processed text is empty
            if not processed_text.strip():
                return {
                    'prediction': 0.5,
                    'class': 'Uncertain',
                    'fake_score': 0.5,
                    'confidence': 0.3,
                    'reasoning': "Text is empty after preprocessing",
                    'chunk_index': 0
                }
            
            features = self.vectorizer.transform([processed_text])
            proba = self.model.predict_proba(features)[0]
            
            fake_probability = proba[0]
            reasoning = self._generate_reasoning(text, fake_probability)
            
            return {
                'prediction': fake_probability,
                'class': 'Fake' if fake_probability > 0.5 else 'Real',
                'fake_score': float(fake_probability),
                'confidence': float(max(proba)),
                'reasoning': reasoning,
                'chunk_index': 0
            }
        except Exception as e:
            print(f"[ERROR] Prediction error: {e}")
            return {
                'prediction': 0.5,
                'class': 'Error',
                'fake_score': 0.5,
                'confidence': 0.0,
                'reasoning': f"Error: {str(e)[:100]}",
                'chunk_index': 0
            }
    
    def combine_chunk_results(self, chunk_results, original_text):
        """Combine predictions from multiple chunks"""
        # Calculate average fake score
        fake_scores = [r['fake_score'] for r in chunk_results]
        avg_fake_score = np.mean(fake_scores)
        
        # Get confidence (lower std = higher confidence)
        std_dev = np.std(fake_scores)
        confidence = 1 - min(std_dev, 0.5)
        
        # Generate overall reasoning
        reasoning_parts = []
        
        # Add chunk analysis summary
        if len(chunk_results) > 1:
            reasoning_parts.append(f"Analyzed {len(chunk_results)} sections of the article")
            
            # Check if chunks had different predictions
            fake_count = sum(1 for s in fake_scores if s > 0.5)
            real_count = len(chunk_results) - fake_count
            
            if fake_count > real_count:
                reasoning_parts.append(f"Majority of sections ({fake_count}/{len(chunk_results)}) show fake indicators")
            elif real_count > fake_count:
                reasoning_parts.append(f"Majority of sections ({real_count}/{len(chunk_results)}) appear legitimate")
        
        # Add overall assessment
        overall_reasoning = self._generate_reasoning(original_text[:500], avg_fake_score)
        reasoning_parts.append(overall_reasoning)
        
        combined_reasoning = " | ".join(reasoning_parts)
        
        return {
            'prediction': avg_fake_score,
            'class': 'Fake' if avg_fake_score > 0.5 else 'Real',
            'fake_score': float(avg_fake_score),
            'confidence': float(confidence),
            'reasoning': combined_reasoning,
            'chunk_analysis': [{'score': r['fake_score'], 'class': r['class']} for r in chunk_results]
        }
    
    def predict(self, text):
        """Main prediction method - handles both short and long text"""
        if not text:
            return {
                'prediction': 0.5,
                'class': 'Error',
                'fake_score': 0.5,
                'confidence': 0.0,
                'reasoning': "No text provided"
            }
        
        # Check text length
        word_count = len(text.split())
        char_count = len(text)
        
        print(f"[INFO] Text stats: {word_count} words, {char_count} characters")
        
        # For very long text (> 1500 words), use chunking
        if word_count > 1500 or char_count > 8000:
            print(f"[INFO] Long text detected, using chunked analysis")
            return self.predict_long_text(text)
        else:
            # Short to medium text, analyze directly
            return self.predict_single(text)
    
    def _emergency_train(self):
        """Emergency training with sample data"""
        print("[INFO] Running emergency training...")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        texts = [
            "The president announced new economic policies today",
            "Scientists discovered a new treatment for cancer",
            "URGENT Breaking news You wont believe what happened",
            "SHOCKING discovery Scientists reveal truth",
            "Government releases official statement about policy",
            "This one weird trick will change your life",
            "Local community raises funds for new school",
            "Viral video shows impossible event that everyone is talking about",
            "Official report shows decrease in unemployment",
            "Miracle cure that doctors dont want you to know"
        ]
        labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]
        
        processed = [self.preprocess(t) for t in texts]
        
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words='english')
        X = self.vectorizer.fit_transform(processed)
        
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(X, labels)
        self.is_trained = True
        
        os.makedirs('models', exist_ok=True)
        with open('models/text_model.pkl', 'wb') as f:
            pickle.dump({'vectorizer': self.vectorizer, 'classifier': self.model}, f)
        print("[SUCCESS] Emergency model trained!")
    
    def _generate_reasoning(self, text, fake_score):
        """Generate explanation for prediction"""
        text_lower = text.lower()
        reasoning = []
        
        # Check for sensational words
        sensational = ['breaking', 'urgent', 'shocking', 'viral', 'alert', 'warning']
        found = [w for w in sensational if w in text_lower]
        if found:
            reasoning.append(f"Sensational language: {', '.join(found)}")
        
        # Check for excessive caps
        caps_count = sum(1 for c in text if c.isupper())
        caps_ratio = caps_count / max(len(text), 1)
        if caps_ratio > 0.15:
            reasoning.append(f"Excessive capitalization detected")
        
        # Check for exclamations
        exclamation_count = text.count('!')
        if exclamation_count > 2:
            reasoning.append(f"Multiple exclamations found")
        
        # Check for question marks
        question_count = text.count('?')
        if question_count > 3:
            reasoning.append(f"Multiple questions detected")
        
        # Check for urgent language
        urgent_words = ['urgent', 'immediately', 'asap', 'now', 'breaking']
        if any(word in text_lower for word in urgent_words):
            reasoning.append("Urgency language detected")
        
        # Overall assessment
        if fake_score > 0.7:
            reasoning.append("High probability of misinformation")
        elif fake_score > 0.5:
            reasoning.append("Some indicators of potential fake news")
        elif fake_score < 0.3:
            reasoning.append("Patterns consistent with legitimate news")
        
        # Add confidence note for long texts
        if len(text.split()) > 1000:
            reasoning.append("Note: Analysis based on full article content")
        
        return " | ".join(reasoning) if reasoning else "No clear indicators found"

# Singleton instance
text_detector = TextFraudDetector()