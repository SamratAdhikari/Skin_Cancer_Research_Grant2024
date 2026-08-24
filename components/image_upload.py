import streamlit as st
from PIL import Image
from utils.api import analyze_image

def show_image_upload():
    """Display the image upload section"""
    st.markdown("### Upload an image")
    st.write("Upload a high-quality image of the skin area you want to analyze.")
    
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        analyze_button = st.button("Analyze Image")
        
        if analyze_button:
            with st.spinner("Analyzing image..."):
                try:
                    # Ensure consistent format (convert all to RGB)
                    image = Image.open(uploaded_file).convert("RGB")

                    # Pass the PIL image or bytes depending on API function
                    result, error = analyze_image(image)
                    
                    if error:
                        st.error(f"Error: {error}")
                    else:
                        st.session_state.result = result
                        st.success("Analysis complete!")

                except Exception as e:
                    st.error(f"Unexpected error: {e}")
        
        if analyze_button or 'result' in st.session_state:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
