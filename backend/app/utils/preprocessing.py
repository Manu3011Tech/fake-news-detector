import re
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_text(text):
    """
    Clean and preprocess text for analysis
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove very short words (less than 2 characters)
    words = text.split()
    words = [word for word in words if len(word) > 2]
    text = ' '.join(words)
    
    return text

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess image for model input
    """
    try:
        # Open image
        img = Image.open(image_path).convert('RGB')
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transforms
        img_tensor = transform(img)
        
        # Add batch dimension
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor
    
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_text_features(text):
    """
    Extract basic features from text without ML model
    """
    features = {}
    
    # Length features
    features['char_count'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = features['char_count'] / max(features['word_count'], 1)
    
    # Punctuation features
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['capital_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    
    # Sensational words
    sensational_words = ['breaking', 'urgent', 'shocking', 'viral', 'alert', 'warning']
    features['sensational_count'] = sum(1 for word in sensational_words if word in text.lower())
    
    # URL presence
    features['has_url'] = 1 if 'http' in text or 'www.' in text else 0
    
    return features

def extract_image_features_simple(image_path):
    """
    Extract basic features from image without deep learning
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        features = {}
        
        # Basic statistics
        features['mean_r'] = np.mean(img_array[:,:,0])
        features['mean_g'] = np.mean(img_array[:,:,1])
        features['mean_b'] = np.mean(img_array[:,:,2])
        features['std_r'] = np.std(img_array[:,:,0])
        features['std_g'] = np.std(img_array[:,:,1])
        features['std_b'] = np.std(img_array[:,:,2])
        
        # Image dimensions
        features['width'] = img.size[0]
        features['height'] = img.size[1]
        features['aspect_ratio'] = features['width'] / features['height']
        
        # Compression artifacts (simplified)
        features['unique_colors'] = len(np.unique(img_array.reshape(-1, 3), axis=0))
        
        # Edge detection (simplified)
        from scipy import ndimage
        gray = np.mean(img_array, axis=2)
        edges = np.abs(ndimage.sobel(gray))
        features['edge_density'] = np.mean(edges)
        
        return features
    
    except Exception as e:
        print(f"Error extracting image features: {e}")
        return {}