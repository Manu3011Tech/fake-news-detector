import re
import random

class FeedbackChatbot:
    def __init__(self):
        self.context = {}
        self.responses = {
            'greeting': [
                "Hello! I'm your fraud detection assistant. How can I help you understand the analysis?",
                "Hi there! Need help interpreting the results? I'm here to explain.",
                "Welcome! I can explain why content was flagged as fake or real."
            ],
            'explain_fake': [
                "The content was flagged as fake because {reason}. Would you like me to elaborate on any specific indicator?",
                "Based on the analysis, this appears fake due to {reason}. This is a common pattern in misinformation.",
                "Several red flags were detected: {reason}. This significantly increases the likelihood of fake content."
            ],
            'explain_real': [
                "This content appears authentic because {reason}. The analysis found no significant manipulation indicators.",
                "The system classified this as real based on {reason}. The patterns match legitimate content.",
                "Good news! This seems legitimate because {reason}. Always verify with official sources though."
            ],
            'verification_tips': [
                "You can verify by: 1) Checking the original source, 2) Looking for official statements, 3) Using reverse image search",
                "Tips for verification: Cross-reference with multiple news outlets, check publication dates, and look for author credentials.",
                "To verify: Search for the claim on fact-checking websites like Snopes, FactCheck.org, or PolitiFact."
            ],
            'help': [
                "I can help you understand why content was flagged, explain specific indicators, or provide verification tips. What would you like to know?",
                "Ask me about: why something is fake/real, how to verify information, or what specific indicators mean."
            ],
            'fallback': [
                "I'm not sure I understand. Could you rephrase? I can explain analysis results or give verification tips.",
                "Let me focus on helping with the fraud detection results. What specific aspect would you like to know about?"
            ]
        }
    
    def get_response(self, user_message, analysis_results=None):
        """Generate chatbot response based on user message"""
        user_message = user_message.lower().strip()
        
        # Check for greetings
        if re.search(r'\b(hi|hello|hey|greetings)\b', user_message):
            return random.choice(self.responses['greeting'])
        
        # Check for explanation requests
        if re.search(r'\b(why|explain|reason|tell me about)\b', user_message) and analysis_results:
            if 'fake' in user_message or analysis_results.get('final_prediction') == 'Fake':
                reasons = analysis_results.get('reasoning', 'multiple indicators')
                response = random.choice(self.responses['explain_fake']).replace('{reason}', reasons)
                return response
            elif 'real' in user_message or analysis_results.get('final_prediction') == 'Real':
                reasons = analysis_results.get('reasoning', 'consistent patterns')
                response = random.choice(self.responses['explain_real']).replace('{reason}', reasons)
                return response
        
        # Check for verification tips
        if re.search(r'\b(verify|check|confirm|trust)\b', user_message):
            return random.choice(self.responses['verification_tips'])
        
        # Check for help
        if re.search(r'\b(help|what can you do|capabilities)\b', user_message):
            return random.choice(self.responses['help'])
        
        # Default response
        return random.choice(self.responses['fallback'])

# Singleton
chatbot = FeedbackChatbot()