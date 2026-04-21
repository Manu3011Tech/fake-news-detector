const API_URL = window.location.origin;

let currentAnalysis = null;

function switchTab(tab) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tab}-tab`).classList.add('active');
    event.target.classList.add('active');
}

async function analyzeText() {
    const text = document.getElementById('news-text').value;
    if (!text.trim()) {
        alert('Please enter some text to analyze');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/predict/text`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const result = await response.json();
        displayResults(result);
        currentAnalysis = result;
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing text. Please try again.');
    }
}

async function analyzeImage() {
    const fileInput = document.getElementById('image-file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image to analyze');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/predict/image`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        displayResults(result);
        currentAnalysis = result;
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing image. Please try again.');
    }
}

async function analyzeBoth() {
    const text = document.getElementById('both-text').value;
    const fileInput = document.getElementById('both-image');
    const file = fileInput.files[0];
    
    if (!text.trim() || !file) {
        alert('Please provide both text and image');
        return;
    }
    
    const formData = new FormData();
    formData.append('text', text);
    formData.append('file', file);
    
    showLoading();
    
    try {
        const response = await fetch(`${API_URL}/predict/both`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        displayResults(result);
        currentAnalysis = result;
    } catch (error) {
        console.error('Error:', error);
        alert('Error analyzing content. Please try again.');
    }
}

function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.style.display = 'block';
    
    // Display verdict
    const verdictDiv = document.getElementById('verdict');
    const verdictClass = result.final_prediction.toLowerCase();
    verdictDiv.innerHTML = result.final_prediction;
    verdictDiv.className = `verdict ${verdictClass}`;
    
    // Display score and confidence
    document.getElementById('score').innerHTML = `Fake Score: ${(result.fake_score * 100).toFixed(1)}%`;
    document.getElementById('confidence').innerHTML = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;
    
    // Display reasoning
    document.getElementById('reasoning').innerHTML = result.reasoning;
    
    // Display suggestions
    const suggestionsList = document.getElementById('suggestions');
    suggestionsList.innerHTML = '';
    result.suggestions.forEach(suggestion => {
        const li = document.createElement('li');
        li.textContent = suggestion;
        suggestionsList.appendChild(li);
    });
    
    // Display charts
    if (result.visualization_chart) {
        document.getElementById('confidence-chart').src = result.visualization_chart;
    }
    
    if (result.comparison_chart) {
        document.getElementById('comparison-chart').src = result.comparison_chart;
    }
    
    // Scroll to results
    resultsDiv.scrollIntoView({ behavior: 'smooth' });
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message to chat
    addChatMessage(message, 'user');
    input.value = '';
    
    try {
        const response = await fetch(`${API_URL}/chatbot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                analysis_context: currentAnalysis
            })
        });
        
        const result = await response.json();
        addChatMessage(result.response, 'bot');
    } catch (error) {
        console.error('Error:', error);
        addChatMessage('Sorry, I encountered an error. Please try again.', 'bot');
    }
}

function addChatMessage(message, sender) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `${sender}-message`;
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showLoading() {
    // You can add a loading spinner here
    console.log('Loading...');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log('Fake News Detection System Ready');
});