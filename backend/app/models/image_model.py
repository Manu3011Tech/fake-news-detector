import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import pickle

class ImageFraudDetector:
    def __init__(self):
        # Use a simple CNN for image classification
        self.model = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        self.is_trained = False
        
    def create_simple_cnn(self):
        """Create a simple CNN for image classification"""
        class SimpleCNN(nn.Module):
            def __init__(self):
                super(SimpleCNN, self).__init__()
                self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.fc1 = nn.Linear(128 * 28 * 28, 512)
                self.fc2 = nn.Linear(512, 2)
                self.dropout = nn.Dropout(0.5)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(-1, 128 * 28 * 28)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        return SimpleCNN()
    
    def train(self, image_paths, labels):
        """Train the model (simplified for demo)"""
        # In production, you'd load actual images
        # For demo, we'll create a dummy model
        self.model = self.create_simple_cnn()
        self.is_trained = True
        
        # Save model
        os.makedirs('models', exist_ok=True)
        torch.save(self.model.state_dict(), 'models/image_model.pth')
    
    def extract_image_features(self, image_path):
        """Extract features that might indicate manipulation"""
        try:
            img = Image.open(image_path).convert('RGB')
            
            # Check image metadata for inconsistencies
            features = {}
            
            # Check image size
            width, height = img.size
            features['aspect_ratio'] = width / height
            
            # Check for compression artifacts (simplified)
            img_array = np.array(img)
            features['std_dev'] = np.std(img_array)
            features['mean_brightness'] = np.mean(img_array)
            
            # Check for unnatural edges (simplified)
            from scipy import ndimage
            edges = np.abs(ndimage.sobel(img_array.mean(axis=2)))
            features['edge_density'] = np.mean(edges)
            
            return features
        except Exception as e:
            return {'error': str(e)}
    
    def predict(self, image_path):
        """Predict if image is real or fake"""
        if not self.is_trained:
            # Load pre-trained model if available
            if os.path.exists('models/image_model.pth'):
                self.model = self.create_simple_cnn()
                self.model.load_state_dict(torch.load('models/image_model.pth'))
                self.model.eval()
                self.is_trained = True
            else:
                # Return dummy prediction with feature analysis
                features = self.extract_image_features(image_path)
                
                # Simple heuristic based on features
                fake_score = 0.5
                reasoning = []
                
                if 'edge_density' in features:
                    if features['edge_density'] > 100:
                        fake_score += 0.2
                        reasoning.append("Unnatural edge artifacts detected")
                
                if 'std_dev' in features:
                    if features['std_dev'] < 30:
                        fake_score += 0.15
                        reasoning.append("Unusual color distribution")
                
                fake_score = min(max(fake_score, 0), 1)
                
                return {
                    'prediction': fake_score,
                    'class': 'Fake' if fake_score > 0.5 else 'Real',
                    'fake_score': float(fake_score),
                    'confidence': float(abs(fake_score - 0.5) * 2 + 0.3),
                    'reasoning': " | ".join(reasoning) if reasoning else "Image appears normal",
                    'features': features
                }
        
        # If we have a trained model, use it (simplified for demo)
        return self._dummy_prediction(image_path)
    
    def _dummy_prediction(self, image_path):
        """Generate dummy prediction with analysis"""
        features = self.extract_image_features(image_path)
        
        # Simple heuristic
        fake_score = 0.3  # Default to real
        reasoning = []
        
        if 'aspect_ratio' in features:
            if features['aspect_ratio'] > 2 or features['aspect_ratio'] < 0.5:
                fake_score += 0.25
                reasoning.append("Unusual aspect ratio detected")
        
        if 'std_dev' in features:
            if features['std_dev'] < 20:
                fake_score += 0.3
                reasoning.append("Possible over-smoothing (AI generation artifact)")
            elif features['std_dev'] > 150:
                fake_score += 0.2
                reasoning.append("High noise level detected")
        
        if 'edge_density' in features:
            if features['edge_density'] > 120:
                fake_score += 0.2
                reasoning.append("Inconsistent edge patterns detected")
        
        fake_score = min(max(fake_score, 0), 1)
        
        return {
            'prediction': fake_score,
            'class': 'Fake' if fake_score > 0.5 else 'Real',
            'fake_score': float(fake_score),
            'confidence': float(0.7 + abs(fake_score - 0.5) * 0.3),
            'reasoning': " | ".join(reasoning) if reasoning else "No obvious manipulation detected",
            'features': features
        }

# Singleton instance
image_detector = ImageFraudDetector()