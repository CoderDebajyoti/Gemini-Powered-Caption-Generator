import streamlit as st
from PIL import Image
from typing import Optional, Tuple
import logging
import google.generativeai as genai
from streamlit_extras.stylable_container import stylable_container # --- NEW: Import for copy button styling ---

# Set up logging for the application
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Caching the Model Initialization ---
@st.cache_resource
def get_gemini_model(api_key: str):
    """
    Initializes and returns a cached Gemini model instance.
    """
    logger.info("Initializing Gemini model for the first time...")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        return model
    except Exception as e:
        logger.error(f"Error initializing Gemini client: {str(e)}")
        st.error(f"Failed to initialize Gemini model. Please check your API key. Error: {e}")
        return None

def generate_content(model, image: Image.Image, prompt: str) -> Tuple[bool, str]:
    """
    Generates content using the provided Gemini model and a prompt.
    """
    try:
        response = model.generate_content([prompt, image])
        return True, response.text
    except Exception as e:
        error_message = f"Error during content generation: {str(e)}"
        logger.error(error_message)
        return False, error_message

def validate_image(uploaded_file) -> Tuple[bool, Optional[Image.Image], str]:
    """
    Validates the uploaded image file.
    """
    if uploaded_file is None:
        return False, None, "No file uploaded"
    try:
        if uploaded_file.size > 10 * 1024 * 1024: # 10MB limit
            return False, None, "File size should be less than 10MB"
        image = Image.open(uploaded_file).convert('RGB')
        return True, image, "Successfully validated image"
    except Exception as e:
        return False, None, f"Invalid image file: {str(e)}"

def main():
    """
    Main Streamlit application function.
    """
    st.set_page_config(
        page_title="Gemini Powered Caption Generator",
        page_icon="♊",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Fira+Sans:wght@400;700&display=swap');
            /* (Your existing CSS is great, no changes needed here) */
            html, body, [class*="css"]  {
                font-family: 'Fira Sans', 'Segoe UI', Arial, sans-serif !important;
                background: #18122B !important;
                color: #F7F7F7 !important;
            }
            .main-header { font-size: 2.8rem; font-weight: 700; color: #A084E8; letter-spacing: 1px; margin-bottom: 0.7rem; text-shadow: 0 2px 12px #5A189A; }
            .info-box, .error-box, .success-box, .result-box { padding: 1.3rem 1.7rem; border-radius: 1rem; margin: 1.3rem 0; box-shadow: 0 4px 24px 0 rgba(160,132,232,0.10); font-size: 1.13rem; border-left-width: 7px; border-left-style: solid; }
            .info-box { background: linear-gradient(90deg, #393053 60%, #635985 100%); border-left-color: #A084E8; }
            .error-box { background: linear-gradient(90deg, #5c2222 60%, #8d4f4f 100%); border-left-color: #D7263D; color: #F7F7F7; }
            .success-box { background: linear-gradient(90deg, #007562 60%, #00ab8e 100%); border-left-color: #00C9A7; }
            .result-box { background: linear-gradient(90deg, #393053 60%, #635985 100%); border-left-color: #8BE8E5; color: #F7F7F7; }
            .stTextInput label, .stFileUploader label, .stRadio label { font-size: 1.2rem !important; font-weight: 700 !important; margin-bottom: 0.5rem; }
            .stButton>button { background: linear-gradient(90deg, #A084E8 60%, #8BE8E5 100%); color: #18122B; font-weight: 700; border-radius: 0.7rem; border: none; padding: 0.7rem 1.4rem; box-shadow: 0 2px 12px 0 rgba(160,132,232,0.18); transition: all 0.2s; }
            .stButton>button:hover { background: linear-gradient(90deg, #5A189A 60%, #00C9A7 100%); color: #F7F7F7; transform: translateY(-2px); }
            .stFileUploader, .stTextInput, .stTextArea { background: #393053 !important; border-radius: 0.7rem !important; padding: 1rem; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-header">Gemini Powered Caption Generator</h1>', unsafe_allow_html=True)
    
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = ""

    # --- Sidebar ---
    with st.sidebar:
        st.markdown('<h2>Configuration</h2>', unsafe_allow_html=True)
        api_key = st.text_input("Enter your Google API Key", type="password")
        if not api_key:
            st.markdown('<div class="info-box">Please enter your API key to enable analysis.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">API Key entered. Ready to generate!</div>', unsafe_allow_html=True)
        st.expander("Usage Information").info("Please be aware of API rate limits on the free tier.")

    # --- Main Layout ---
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<h2>Image Upload</h2>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
        if uploaded_file:
            is_valid, image, message = validate_image(uploaded_file)
            if is_valid:
                st.image(image, caption="Uploaded Image", use_container_width=True)
            else:
                st.markdown(f'<div class="error-box">{message}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h2>Caption Options</h2>', unsafe_allow_html=True)
        user_description = st.text_input("Describe your photo (optional)", placeholder="e.g. A group of friends at the beach")
        caption_length = st.radio("Caption length", options=["Short", "Moderate", "Long & Detailed"], index=1, horizontal=True)
        generate_button = st.button("✨ Generate Content", use_container_width=True)

        if generate_button:
            if not api_key or not uploaded_file:
                st.error("Please provide an API Key and an image.")
            else:
                is_valid, image, _ = validate_image(uploaded_file)
                if is_valid:
                    with st.spinner("Crafting the perfect post..."):
                        model = get_gemini_model(api_key)
                        if model:
                            # --- NEW: Updated prompt to include song suggestions ---
                            prompt_parts = [
                                "Act as a creative social media expert.",
                                f"Generate a caption for the provided image. The caption's tone should be engaging and its length should be '{caption_length}'."
                            ]
                            if user_description.strip():
                                prompt_parts.append(f"Use this user-provided context: '{user_description.strip()}'.")
                            
                            prompt_parts.append(
                                "Format the response exactly like this, with each item on a new line:\n"
                                "CAPTION: [Your generated caption here]\n"
                                "EMOJI: [A single emoji here]\n"
                                "HASHTAGS: [hashtag1, hashtag2, hashtag3, hashtag4, hashtag5]\n"
                                "SONGS: [Song Title 1 - Artist 1, Song Title 2 - Artist 2, Song Title 3 - Artist 3]"
                            )
                            prompt = "\n".join(prompt_parts)
                            
                            success, response = generate_content(model, image, prompt)
                            st.session_state.analysis_result = response if success else f'<div class="error-box">{response}</div>'
                        else:
                            st.session_state.analysis_result = '<div class="error-box">Could not initialize Gemini model.</div>'
        
        if st.session_state.analysis_result:
            st.markdown('<h3>Generated Content</h3>', unsafe_allow_html=True)
            with stylable_container(key="results_container", css_styles=".result-box { /* (already defined above) */ }") as container:
                # --- NEW: Updated parsing logic for all parts ---
                try:
                    lines = st.session_state.analysis_result.strip().split('\n')
                    parts = {line.split(':')[0]: line.split(':', 1)[1].strip() for line in lines if ':' in line}

                    caption_part = parts.get("CAPTION", "")
                    emoji_part = parts.get("EMOJI", "")
                    hashtags_part = parts.get("HASHTAGS", "")
                    songs_part = parts.get("SONGS", "")
                    
                    full_caption = f"{caption_part} {emoji_part}"
                    hashtags = [f"#{tag.strip()}" for tag in hashtags_part.split(',')]
                    songs = [song.strip() for song in songs_part.split(',')]

                    # --- NEW: Use st.expander for a cleaner look ---
                    with st.expander("📝 **Caption & Hashtags**", expanded=True):
                        st.markdown(f'<p style="font-size: 1.2rem; margin-bottom: 0.5rem;">{full_caption}</p>', unsafe_allow_html=True)
                        st.code(" ".join(hashtags), language=None)
                        if st.button("📋 Copy Caption", key="copy_caption"):
                            st.toast("Caption copied to clipboard!")
                            st.components.v1.html(f"<script>navigator.clipboard.writeText('{full_caption.replace('`', '').replace('$', '')}');</script>", height=0)


                    if songs:
                        with st.expander("🎵 **Song Suggestions**"):
                            for song in songs:
                                st.text(f"• {song}")

                except Exception as e:
                    # Fallback for unstructured or error responses
                    st.error(f"Could not parse the AI's response. Error: {e}")
                    st.markdown(f'<div class="result-box">{st.session_state.analysis_result}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()