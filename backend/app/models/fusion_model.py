import numpy as np
from .text_model import text_detector
from .image_model import image_detector

class FusionDetector:
    def __init__(self):
        self.text_weight = 0.6  # Text analysis weight
        self.image_weight = 0.4  # Image analysis weight
        
    def predict(self, text=None, image_path=None):
        """Combine predictions from text and image models"""
        results = {}
        final_fake_score = 0
        total_weight = 0
        
        # Get text prediction
        if text:
            text_result = text_detector.predict(text)
            results['text_analysis'] = text_result
            final_fake_score += text_result['fake_score'] * self.text_weight
            total_weight += self.text_weight
        
        # Get image prediction
        if image_path:
            image_result = image_detector.predict(image_path)
            results['image_analysis'] = image_result
            final_fake_score += image_result['fake_score'] * self.image_weight
            total_weight += self.image_weight
        
        # Normalize score
        if total_weight > 0:
            final_fake_score /= total_weight
        else:
            final_fake_score = 0.5
        
        # Generate combined reasoning
        reasoning = self._combine_reasoning(results)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(final_fake_score, results)
        
        return {
            'final_prediction': 'Fake' if final_fake_score > 0.5 else 'Real',
            'fake_score': float(final_fake_score),
            'confidence': float(1 - abs(final_fake_score - 0.5) * 2),
            'detailed_analysis': results,
            'reasoning': reasoning,
            'suggestions': suggestions,
            'visualization_data': self._prepare_visualization_data(final_fake_score, results)
        }
    
    def _combine_reasoning(self, results):
        """Combine reasoning from different modalities"""
        reasons = []
        
        if 'text_analysis' in results:
            reasons.append(f"Text: {results['text_analysis']['reasoning']}")
        
        if 'image_analysis' in results:
            reasons.append(f"Image: {results['image_analysis']['reasoning']}")
        
        return " | ".join(reasons)
    
    def _generate_suggestions(self, fake_score, results):
        """Generate actionable suggestions"""
        suggestions = []
        
        if fake_score > 0.7:
            suggestions.append("⚠️ This content shows strong indicators of being fake")
            suggestions.append("✓ Verify with official sources before sharing")
            suggestions.append("✓ Check fact-checking websites like Snopes or FactCheck.org")
            suggestions.append("✓ Look for the original source of the information")
        elif fake_score > 0.5:
            suggestions.append("⚠️ Some suspicious patterns detected")
            suggestions.append("✓ Cross-reference with multiple news sources")
            suggestions.append("✓ Check the publication date and author credentials")
        else:
            suggestions.append("✅ This content appears legitimate")
            suggestions.append("✓ Still verify critical claims with official sources")
            suggestions.append("✓ Be aware that even real news can contain errors")
        
        # Add specific suggestions based on analysis
        if 'text_analysis' in results:
            if results['text_analysis']['fake_score'] > 0.6:
                suggestions.append("📝 The writing style shows signs of clickbait or sensationalism")
        
        if 'image_analysis' in results:
            if results['image_analysis']['fake_score'] > 0.6:
                suggestions.append("🖼️ The image shows potential manipulation artifacts")
                suggestions.append("🔍 Try reverse image search to find original source")
        
        return suggestions
    
    def _prepare_visualization_data(self, final_score, results):
        """Prepare data for charts"""
        viz_data = {
            'final_score': final_score,
            'components': {}
        }
        
        if 'text_analysis' in results:
            viz_data['components']['text'] = results['text_analysis']['fake_score']
        
        if 'image_analysis' in results:
            viz_data['components']['image'] = results['image_analysis']['fake_score']
        
        return viz_data

# Singleton
fusion_detector = FusionDetector()