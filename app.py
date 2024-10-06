import streamlit as st
import base64
from PIL import Image
import io
import os
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def analyze_image(image, question, model_choice, system_prompt):
    base64_image = encode_image(image)
    
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"### Instructions\n{system_prompt}\n### Question\n{question}"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=model_choice,
    )
    
    return chat_completion.choices[0].message.content

def summarize_descriptions(descriptions, system_prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Summarize the following image descriptions:\n\n{descriptions}"}
        ],
        model="llama-3.1-70b-versatile",
    )
    
    return chat_completion.choices[0].message.content

def main():
    st.set_page_config(layout="wide")

    st.title("Groq Batch Vision")
    st.image("imgs/powered-by-groq.svg", width=300)


    # Initialize session state for settings
    if 'vision_system_prompt' not in st.session_state:
        st.session_state.vision_system_prompt = "You are an AI assistant that analyzes images. Describe the image in detail."
    if 'summary_system_prompt' not in st.session_state:
        st.session_state.summary_system_prompt = "You are an AI assistant that summarizes image descriptions. Provide a concise summary of the collection of images based on their descriptions."

    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        if uploaded_files:
            st.subheader("Uploaded Images")
            cols = st.columns(3)
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=uploaded_file.name, use_column_width=True, width=200)

        run_button = st.button("Run Analysis")

        if run_button and uploaded_files:
            descriptions = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                description = analyze_image(image, "Describe this image in detail.", "llama-3.2-11b-vision-preview", st.session_state.vision_system_prompt)
                st.write(f"**Description for {uploaded_file.name}:** {description}")
                descriptions.append(description)

            if len(descriptions) > 0:
                st.write("## Summary of Images")
                summary = summarize_descriptions("\n".join(descriptions), st.session_state.summary_system_prompt)
                st.write(summary)

    with col2:
        with st.expander("Settings", expanded=False):
            st.session_state.vision_system_prompt = st.text_area("Vision System Prompt", st.session_state.vision_system_prompt)
            st.session_state.summary_system_prompt = st.text_area("Summary System Prompt", st.session_state.summary_system_prompt)

if __name__ == "__main__":
    main()