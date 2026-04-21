import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import base64
from matplotlib.patches import Rectangle

def create_confidence_chart(fake_score, confidence):
    """Create a confidence gauge chart"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # Left chart: Fake Score Meter
    colors = ['green', 'yellow', 'red']
    bounds = [0, 0.3, 0.7, 1]
    
    # Create horizontal bar
    ax1.barh([0], [fake_score], color='red' if fake_score > 0.5 else 'green', height=0.5)
    ax1.barh([0], [1], alpha=0.3, color='gray', height=0.5)
    ax1.set_xlim(0, 1)
    ax1.set_title(f'Fake Score: {fake_score:.2%}')
    ax1.set_yticks([])
    ax1.axvline(x=0.5, color='black', linestyle='--', alpha=0.5)
    ax1.text(0.5, -0.5, 'Threshold', ha='center')
    
    # Right chart: Confidence
    ax2.pie([confidence, 1-confidence], 
            labels=[f'Confidence\n{confidence:.2%}', 'Uncertainty'],
            colors=['#2ecc71', '#e74c3c'],
            autopct='%1.1f%%')
    ax2.set_title('Prediction Confidence')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def create_comparison_chart(components):
    """Create a bar chart comparing different components"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    names = list(components.keys())
    scores = list(components.values())
    colors = ['#e74c3c' if s > 0.5 else '#2ecc71' for s in scores]
    
    bars = ax.bar(names, scores, color=colors, alpha=0.8)
    ax.axhline(y=0.5, color='black', linestyle='--', label='Fake/Real Threshold')
    ax.set_ylabel('Fake Score')
    ax.set_title('Analysis Breakdown by Component')
    ax.set_ylim(0, 1)
    ax.legend()
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"

def create_radar_chart(scores_dict):
    """Create a radar chart for multi-dimensional analysis"""
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    categories = list(scores_dict.keys())
    values = list(scores_dict.values())
    
    # Number of variables
    N = len(categories)
    
    # Compute angles for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add the first value to the end to close the circle
    values += values[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, color='#3498db')
    ax.fill(angles, values, alpha=0.25, color='#3498db')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Set y-axis limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
    
    ax.set_title('Multi-dimensional Analysis', pad=20)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return f"data:image/png;base64,{image_base64}"