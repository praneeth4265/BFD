"""
Streamlit frontend for bone fracture detection.
Provides an intuitive web interface with drag-and-drop upload and visualization.
"""

import streamlit as st
import requests
import numpy as np
import cv2
from PIL import Image
import io
import base64
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, Optional


# Page configuration
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2E86AB;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .result-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin: 1rem 0;
    }
    
    .fracture-detected {
        background-color: #ffe6e6;
        border-color: #ff9999;
    }
    
    .no-fracture {
        background-color: #e6ffe6;
        border-color: #99ff99;
    }
    
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    
    .info-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitApp:
    """Main Streamlit application for bone fracture detection."""
    
    def __init__(self):
        """Initialize the Streamlit app."""
        self.api_url = "http://localhost:8000"  # FastAPI backend URL
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        
        if 'current_image' not in st.session_state:
            st.session_state.current_image = None
        
        if 'current_result' not in st.session_state:
            st.session_state.current_result = None
    
    def check_api_health(self) -> bool:
        """Check if the FastAPI backend is running."""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def predict_image(self, image: np.ndarray, filename: str = "uploaded_image") -> Optional[Dict[str, Any]]:
        """
        Send image to API for prediction.
        
        Args:
            image: Image as numpy array
            filename: Name of the file
            
        Returns:
            Prediction result dictionary
        """
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image.astype(np.uint8))
            else:
                pil_image = Image.fromarray(image.astype(np.uint8), mode='L')
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            
            # Send request to API
            files = {'file': (filename, img_buffer, 'image/png')}
            response = requests.post(f"{self.api_url}/predict", files=files, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return None
    
    def display_prediction_result(self, result: Dict[str, Any], image: np.ndarray):
        """
        Display prediction results with visualization.
        
        Args:
            result: Prediction result from API
            image: Original image
        """
        if not result or not result.get('success', False):
            st.error("Prediction failed or returned invalid results")
            return
        
        # Main result display
        predicted_label = result['predicted_label']
        confidence = result['confidence']
        
        # Determine styling based on prediction
        if predicted_label == 'Fracture':
            card_class = "fracture-detected"
            icon = "⚠️"
            color = "#dc3545"
        else:
            card_class = "no-fracture"
            icon = "✅"
            color = "#28a745"
        
        # Confidence styling
        if confidence >= 0.8:
            conf_class = "confidence-high"
        elif confidence >= 0.6:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        
        # Display result card
        st.markdown(f"""
        <div class="result-card {card_class}">
            <h2 style="color: {color};">{icon} {predicted_label}</h2>
            <p class="{conf_class}">Confidence: {confidence:.1%}</p>
            <p><strong>Inference Time:</strong> {result.get('inference_time_ms', 0):.1f} ms</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create columns for detailed results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Prediction Probabilities")
            
            # Create probability chart
            class_names = result['class_names']
            probabilities = result['probabilities']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=probabilities,
                    marker_color=['#ff6b6b' if name == 'Fracture' else '#51cf66' for name in class_names],
                    text=[f"{prob:.1%}" for prob in probabilities],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Class Probabilities",
                xaxis_title="Prediction",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🖼️ Original Image")
            st.image(image, caption="Uploaded X-ray Image", use_column_width=True)
        
        # Model information
        if 'model_info' in result:
            with st.expander("🔧 Model Information"):
                model_info = result['model_info']
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Input Shape:**", model_info.get('input_shape', 'N/A'))
                    st.write("**Providers:**", ', '.join(model_info.get('providers', [])))
                
                with col2:
                    st.write("**File Info:**")
                    if 'file_info' in result:
                        file_info = result['file_info']
                        st.write(f"- Filename: {file_info.get('filename', 'N/A')}")
                        st.write(f"- Size: {file_info.get('size_bytes', 0) / 1024:.1f} KB")
                        st.write(f"- Type: {file_info.get('content_type', 'N/A')}")
    
    def display_prediction_history(self):
        """Display prediction history in sidebar."""
        if st.session_state.prediction_history:
            st.sidebar.subheader("📈 Prediction History")
            
            for i, (timestamp, filename, prediction, confidence) in enumerate(reversed(st.session_state.prediction_history[-10:])):
                with st.sidebar.expander(f"{filename[:20]}... - {timestamp}"):
                    icon = "⚠️" if prediction == 'Fracture' else "✅"
                    st.write(f"{icon} **{prediction}**")
                    st.write(f"Confidence: {confidence:.1%}")
        else:
            st.sidebar.info("No predictions yet. Upload an image to get started!")
    
    def create_example_images(self):
        """Create example images section."""
        st.subheader("📋 Example X-ray Images")
        st.info("Try these example images to test the model:")
        
        # In a real application, you would have actual X-ray example images
        example_text = """
        **To add example images:**
        1. Create an `examples` folder in the `app/static` directory
        2. Add X-ray images (both fracture and no-fracture examples)
        3. The API will automatically serve them at `/examples` endpoint
        
        **Recommended image types:**
        - Bone fractures: wrist, ankle, arm fractures
        - No fractures: normal bone X-rays
        - Format: JPEG, PNG
        - Size: Under 5MB
        """
        st.markdown(example_text)
    
    def main(self):
        """Main application interface."""
        # Header
        st.markdown('<h1 class="main-header">🦴 Bone Fracture Detection</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">AI-powered medical imaging analysis for bone fracture detection</p>', unsafe_allow_html=True)
        
        # Check API health
        if not self.check_api_health():
            st.error("❌ Cannot connect to the API backend. Please ensure the FastAPI server is running on http://localhost:8000")
            st.info("To start the backend, run: `python app/main.py`")
            return
        
        st.success("✅ Connected to API backend")
        
        # Sidebar
        st.sidebar.title("🔧 Settings")
        
        # File upload
        st.sidebar.subheader("📤 Upload X-ray Image")
        uploaded_file = st.sidebar.file_uploader(
            "Choose an X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a bone X-ray image for fracture detection"
        )
        
        # Upload and analyze button
        if uploaded_file is not None:
            # Display uploaded image info
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
            st.sidebar.write(f"Size: {len(uploaded_file.getvalue()) / 1024:.1f} KB")
            
            # Convert uploaded file to image
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            # Store in session state
            st.session_state.current_image = image_array
            
            # Analyze button
            if st.sidebar.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing X-ray image..."):
                    # Make prediction
                    result = self.predict_image(image_array, uploaded_file.name)
                    
                    if result and result.get('success', False):
                        # Store result
                        st.session_state.current_result = result
                        
                        # Add to history
                        timestamp = time.strftime("%H:%M:%S")
                        st.session_state.prediction_history.append((
                            timestamp,
                            uploaded_file.name,
                            result['predicted_label'],
                            result['confidence']
                        ))
                        
                        st.sidebar.success("✅ Analysis complete!")
                    else:
                        st.sidebar.error("❌ Analysis failed")
        
        # Display prediction history
        self.display_prediction_history()
        
        # Main content area
        if st.session_state.current_result and st.session_state.current_image is not None:
            # Display current result
            self.display_prediction_result(st.session_state.current_result, st.session_state.current_image)
            
            # Clear results button
            if st.button("🗑️ Clear Results"):
                st.session_state.current_result = None
                st.session_state.current_image = None
                st.rerun()
        
        else:
            # Welcome message and instructions
            st.markdown("""
            <div class="info-box">
                <h3>🚀 Getting Started</h3>
                <p>Follow these steps to analyze your X-ray images:</p>
                <ol>
                    <li><strong>Upload</strong> an X-ray image using the sidebar</li>
                    <li><strong>Click</strong> the "Analyze Image" button</li>
                    <li><strong>View</strong> the AI prediction results with confidence scores</li>
                    <li><strong>Review</strong> the prediction history in the sidebar</li>
                </ol>
            </div>
            """, unsafe_allow_html=True)
            
            # Model information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🤖 About the AI Model")
                st.markdown("""
                - **Architecture**: Deep learning ensemble with attention mechanisms
                - **Training**: Medical imaging dataset with expert annotations
                - **Accuracy**: High-performance fracture detection
                - **Speed**: Real-time inference (<100ms)
                - **Format**: Optimized ONNX model for deployment
                """)
            
            with col2:
                st.subheader("⚕️ Medical Disclaimer")
                st.warning("""
                **Important**: This tool is for educational and research purposes only. 
                It should not be used as a substitute for professional medical diagnosis. 
                Always consult with qualified healthcare professionals for medical decisions.
                """)
            
            # Example images section
            self.create_example_images()
        
        # Statistics and model performance
        if st.session_state.prediction_history:
            st.subheader("📊 Session Statistics")
            
            # Calculate statistics
            total_predictions = len(st.session_state.prediction_history)
            fractures_detected = sum(1 for _, _, pred, _ in st.session_state.prediction_history if pred == 'Fracture')
            avg_confidence = np.mean([conf for _, _, _, conf in st.session_state.prediction_history])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", total_predictions)
            
            with col2:
                st.metric("Fractures Detected", fractures_detected)
            
            with col3:
                st.metric("Average Confidence", f"{avg_confidence:.1%}")


def main():
    """Run the Streamlit application."""
    app = StreamlitApp()
    app.main()


if __name__ == "__main__":
    main()