import streamlit as st
import google.generativeai as genai
from PIL import Image
from typing import Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiVisionAnalyzer:
    """
    A class to handle Gemini API interactions for image analysis.
    """
    def __init__(self, api_key: str):
        """
        Initialize the Gemini client with an API key.
        """
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            error_message = f"Error initializing Gemini client: {str(e)}"
            logger.error(error_message)
            raise RuntimeError(error_message) from e

    def analyze_image(self, image: Image.Image, question: str) -> Tuple[bool, str]:
        """
        Analyze an image using the Gemini.flash API.
        """
        try:
            prompt = f"Please answer the question based on the image: {question}"
            response = self.model.generate_content([prompt, image])
            if response.text:
                return True, response.text
            else:
                return False, "No response from model."
        except Exception as e:
            error_message = f"Error during image analysis: {str(e)}"
            logger.error(error_message)
            return False, error_message

    @staticmethod
    def validate_image(uploaded_file) -> Tuple[bool, Optional[Image.Image], str]:
        """
        Validate an uploaded image file.
        """
        if uploaded_file is None:
            return False, None, "No file uploaded."

        try:
            if uploaded_file.size > 10 * 1024 * 1024:
                return False, None, "File size should be less than 10MB."

            image = Image.open(uploaded_file)
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            return True, image, "Successfully validated image."
        except Exception as e:
            return False, None, f"Invalid image file: {str(e)}"

@st.cache_resource
def get_gemini_analyzer(api_key: str):
    """
    Caches and returns a GeminiVisionAnalyzer instance.
    """
    return GeminiVisionAnalyzer(api_key)

def main():
    """
    Main Streamlit application function.
    """
    st.set_page_config(
        page_title="Image Processing App",
        page_icon="🔎",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS
    st.markdown("""
        <style>
            .main-header { font-size: 2.5rem; font-weight: bold; color: #1e3a8a; margin-bottom: 1rem; }
            .info-box { background-color: #eef2ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #3b82f6; }
            .error-box { background-color: #fef2f2; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #ef4444; }
            .success-box { background-color: #f0fdfa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #10b981; }
            .result-box { background-color: #ecfdf5; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #0ea5e9; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Gemini Vision Analyzer</h1>', unsafe_allow_html=True)

    # Sidebar for API key
    with st.sidebar:
        st.markdown('<h2 class="main-header">Configuration</h2>', unsafe_allow_html=True)
        api_key = st.text_input("API Key", type="password", help="Enter your Gemini API Key")
        if api_key:
            st.markdown('<div class="success-box">API Key configured</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="info-box">
                    <strong>Setup Instructions</strong><br>
                    1. Get an API key from Google AI Studio.<br>
                    2. Paste it here to use the app.
                </div>
            """, unsafe_allow_html=True)

    # Main content area
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h2 class="main-header">Image Upload</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            is_valid, image, message = GeminiVisionAnalyzer.validate_image(uploaded_file)
            if is_valid:
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown(f'<div class="success-box">{message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h2 class="main-header">Question & Analysis</h2>', unsafe_allow_html=True)
        question = st.text_area(
            "Enter your question",
            height=100,
            placeholder="What do you see in this image?",
            help="Be specific with your question for better results."
        )
        analyze_button = st.button("Analyze Image", use_container_width=True)

    # Handle analysis in a separate block for clarity
    if analyze_button:
        if not api_key:
            st.error("Please enter your API key.")
        elif not uploaded_file:
            st.error("Please upload an image.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            is_valid, image, _ = GeminiVisionAnalyzer.validate_image(uploaded_file)
            if not is_valid:
                st.markdown('<div class="error-box">Invalid image file.</div>', unsafe_allow_html=True)
            else:
                try:
                    with st.spinner("Initializing Gemini model..."):
                        analyzer = get_gemini_analyzer(api_key)
                    
                    with st.spinner("Analyzing image..."):
                        success, response = analyzer.analyze_image(image, question)
                        
                        st.markdown('<h3 class="main-header">Analysis Results</h3>', unsafe_allow_html=True)
                        if success:
                            st.markdown(f'<div class="result-box">{response}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="error-box">{response}</div>', unsafe_allow_html=True)
                except RuntimeError as e:
                    st.markdown(f'<div class="error-box">Initialization Error: {e}</div>', unsafe_allow_html=True)
                except Exception:
                    st.markdown('<div class="error-box">An unexpected error occurred during analysis.</div>', unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("Application Information"):
        st.markdown("""
        **Gemini Vision Analyzer**

        This application uses Google's Gemini 1.5 Vision model to analyze images and 
        answer questions about them.

        **Features:**
        - Upload and validate images
        - Ask custom questions
        - Get intelligent responses
        - Error handling and logging

        **Supported Formats:** PNG, JPG, JPEG  
        **Max Size:** 10MB  
        **Note:** You must provide a valid Gemini API Key.
        """)

if __name__ == "__main__":
    main()