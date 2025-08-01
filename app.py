import streamlit as st
from PIL import Image
from typing import Optional, Tuple
import logging
import google.generativeai as genai

# Set up logging for Gemini Powered Caption Generator
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Caching the Model Initialization for Gemini Powered Caption Generator ---
@st.cache_resource
def get_gemini_analyzer(api_key: str):
    """
    Initializes and returns a cached Gemini model instance for Gemini Powered Caption Generator.
    The @st.cache_resource decorator ensures this function is run only once.
    """
    logger.info("Initializing Gemini model for the first time...")
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
    Analyze an image using the provided Gemini model for Gemini Powered Caption Generator.
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
    Validate an uploaded image file for Gemini Powered Caption Generator.
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
    Main Streamlit application function for Gemini Powered Caption Generator.
    """
    st.set_page_config(
        page_title="Gemini Powered Caption Generator",
        page_icon="♊",
        layout="wide"
    )

    # Custom CSS for Gemini Powered Caption Generator: bold colors, new font, modern look
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;700&display=swap');
            html, body, [class*="css"]  {
                font-family: 'Fira Sans', 'Segoe UI', Arial, sans-serif !important;
                background: #18122B !important;
                color: #F7F7F7 !important;
            }
            .main-header {
                font-size: 2.8rem;
                font-weight: 700;
                color: #A084E8;
                letter-spacing: 1px;
                margin-bottom: 0.7rem;
                text-shadow: 0 2px 12px #5A189A;
            }
            .info-box, .error-box, .success-box, .result-box {
                padding: 1.3rem 1.7rem;
                border-radius: 1rem;
                margin: 1.3rem 0;
                box-shadow: 0 4px 24px 0 rgba(160,132,232,0.10);
                font-size: 1.13rem;
                border-left-width: 7px;
                border-left-style: solid;
                font-family: 'Fira Sans', 'Segoe UI', Arial, sans-serif !important;
            }
            .info-box {
                background: linear-gradient(90deg, #393053 60%, #635985 100%);
                border-left-color: #A084E8;
                color: #F7F7F7;
            }
            .error-box {
                background: linear-gradient(90deg, #FF6B6B 60%, #F7D6D6 100%);
                border-left-color: #D7263D;
                color: #2D142C;
            }
            .success-box {
                background: linear-gradient(90deg, #00C9A7 60%, #B8FFF9 100%);
                border-left-color: #00C9A7;
                color: #18122B;
            }
            .result-box {
                background: linear-gradient(90deg, #A084E8 60%, #8BE8E5 100%);
                border-left-color: #5A189A;
                color: #18122B;
            }
            textarea, .stTextInput>div>div>input {
                background: #393053 !important;
                border-radius: 0.7rem !important;
                border: 2px solid #A084E8 !important;
                font-size: 1.08rem !important;
                color: #F7F7F7 !important;
            }
            .stButton>button {
                background: linear-gradient(90deg, #A084E8 60%, #8BE8E5 100%);
                color: #18122B;
                font-weight: 700;
                border-radius: 0.7rem;
                border: none;
                padding: 0.7rem 1.4rem;
                box-shadow: 0 2px 12px 0 rgba(160,132,232,0.18);
                transition: background 0.2s, color 0.2s;
                font-family: 'Fira Sans', 'Segoe UI', Arial, sans-serif !important;
            }
            .stButton>button:hover {
                background: linear-gradient(90deg, #5A189A 60%, #00C9A7 100%);
                color: #F7F7F7;
            }
            .stTextArea textarea {
                color: #F7F7F7 !important;
            }
            .stFileUploader, .stTextInput, .stTextArea {
                background: #393053 !important;
                border-radius: 0.7rem !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Gemini Powered Caption Generator</h1>', unsafe_allow_html=True)
    
    # Initialize session state for storing results
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = ""

    # Sidebar for API Key configuration
    with st.sidebar:
        st.markdown('<h2>Configuration</h2>', unsafe_allow_html=True)
        
        # Directly ask the user for their API key
        api_key = st.text_input(
            "Enter your Google API Key", 
            type="password", 
            help="Get your key from Google AI Studio."
        )

        if not api_key:
            st.markdown("""
                <div class="info-box">
                    Please enter your API key to enable analysis.
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">API Key entered.</div>', unsafe_allow_html=True)

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
        st.markdown('<h2>Caption Options</h2>', unsafe_allow_html=True)
        user_description = st.text_input(
            "Describe your photo (optional)",
            placeholder="e.g. A group of friends at the beach during sunset"
        )
        caption_length = st.radio(
            "Caption length",
            options=["Short", "Moderate", "Long & Detailed"],
            index=1,
            horizontal=True
        )
        generate_button = st.button("Generate Caption", use_container_width=True, type="primary")

        if generate_button:
            # Input validation for Gemini Powered Caption Generator
            if not api_key:
                st.error("Please enter your API key.")
            elif not uploaded_file:
                st.error("Please upload an image.")
            else:
                is_valid, image, _ = validate_image(uploaded_file)
                if is_valid:
                    with st.spinner("Generating caption... This may take a moment."):
                        # Get the cached model for Gemini Powered Caption Generator
                        model = get_gemini_analyzer(api_key)
                        if model:
                            # Build prompt
                            prompt = "Write a"
                            if caption_length == "Short":
                                prompt += " short"
                            elif caption_length == "Moderate":
                                prompt += " moderate-length"
                            else:
                                prompt += " long, detailed"
                            prompt += " caption for this photo."
                            if user_description.strip():
                                prompt += f" Here is some context: {user_description.strip()}"
                            else:
                                prompt += " Predict everything from the image."
                            success, response = analyze_image(model, image, prompt)
                            if success:
                                # Store result in session state
                                st.session_state.analysis_result = response
                            else:
                                st.session_state.analysis_result = f'<div class="error-box">{response}</div>'
                        else:
                            st.session_state.analysis_result = '<div class="error-box">Could not initialize Gemini model.</div>'

        # Display the result from session state
        if st.session_state.analysis_result:
            st.markdown('<h3>Generated Caption</h3>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-box">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()