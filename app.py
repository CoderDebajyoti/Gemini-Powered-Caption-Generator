import streamlit as st
from PIL import Image
from typing import Optional, Tuple
import logging
import google.generativeai as genai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Caching the Model Initialization ---
@st.cache_resource
def get_gemini_analyzer(api_key: str):
    """
    Initializes and returns a cached GeminiVisionAnalyzer instance.
    The @st.cache_resource decorator ensures this function is run only once.
    """
    logger.info("Initializing GeminiVisionAnalyzer for the first time...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        return model
    except Exception as e:
        logger.error(f"Error initializing Gemini client: {str(e)}")
        st.error(f"Failed to initialize Gemini model. Please check your API key. Error: {e}")
        return None

def analyze_image(model, image: Image.Image, question: str) -> Tuple[bool, str]:
    """
    Analyze an image using the provided Gemini model.
    Now a standalone function that accepts the cached model.
    """
    try:
        prompt = f"Please answer the question based on the image: {question}"
        response = model.generate_content([prompt, image])
        return True, response.text
    except Exception as e:
        error_message = f"Error during image analysis: {str(e)}"
        logger.error(error_message)
        return False, error_message

def validate_image(uploaded_file) -> Tuple[bool, Optional[Image.Image], str]:
    """
    Validate an uploaded image file.
    """
    if uploaded_file is None:
        return False, None, "No file uploaded"
    try:
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, None, "File size should be less than 10MB"
        image = Image.open(uploaded_file)
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        return True, image, "Successfully validated image"
    except Exception as e:
        return False, None, f"Invalid image file: {str(e)}"

def main():
    """
    Main Streamlit application function.
    """
    st.set_page_config(
        page_title="Image Processing App",
        page_icon="",
        layout="wide"
    )

    # Custom CSS remains the same
    st.markdown("""
        <style>
            .main-header { font-size: 2.5rem; font-weight: bold; color: #1e3a8a; }
            .info-box { background-color: #eef2ff; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #3b82f6; }
            .error-box { background-color: #fef2f2; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #ef4444; }
            .success-box { background-color: #f0fdfa; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #10b981; }
            .result-box { background-color: #ecfdf5; padding: 1rem; border-radius: 0.5rem; margin: 1rem 0; border-left: 4px solid #0ea5e9; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Gemini Vision Analyzer</h1>', unsafe_allow_html=True)
    
    # Initialize session state for storing results
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = ""

    # Sidebar for API Key configuration
    with st.sidebar:
        st.markdown('<h2>Configuration</h2>', unsafe_allow_html=True)
        # Try loading from secrets first, then fall back to text input
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            api_key = st.text_input("API Key", type="password", help="Enter your Gemini API Key")
            st.markdown("""
                <div class="info-box">
                    <strong>Setup Instructions</strong><br>
                    1. Get a key from Google AI Studio<br>
                    2. Paste it above or add it to your Streamlit secrets (`secrets.toml`)
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">API Key loaded from secrets.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<h2>Image Upload</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            is_valid, image, message = validate_image(uploaded_file)
            if is_valid:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            else:
                st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h2>Question & Analysis</h2>', unsafe_allow_html=True)
        question = st.text_area("Enter your question", height=100, placeholder="What do you see in this image?")
        analyze_button = st.button("Analyze Image", use_container_width=True, type="primary")

        if analyze_button:
            # Input validation
            if not api_key:
                st.error("Please enter your API key.")
            elif not uploaded_file:
                st.error("Please upload an image.")
            elif not question.strip():
                st.error("Please enter a question.")
            else:
                is_valid, image, _ = validate_image(uploaded_file)
                if is_valid:
                    with st.spinner("Analyzing image... This may take a moment."):
                        # Get the cached model
                        model = get_gemini_analyzer(api_key)
                        if model:
                            success, response = analyze_image(model, image, question)
                            if success:
                                # Store result in session state
                                st.session_state.analysis_result = response
                            else:
                                st.session_state.analysis_result = f'<div class="error-box">{response}</div>'
                        else:
                            st.session_state.analysis_result = '<div class="error-box">Could not initialize Gemini model.</div>'
        
        # Display the result from session state
        if st.session_state.analysis_result:
            st.markdown('<h3>Analysis Results</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()