import streamlit as st
import base64
from PIL import Image
import io
import os
from groq import Groq

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def compress_image(image, max_size_mb=4):
    """
    Compress the image to be under max_size_mb by adjusting quality and size.
    """
    max_bytes = max_size_mb * 1024 * 1024
    quality = 85  # Start with high quality
    min_quality = 10  # Lowered to allow more compression
    step = 5

    # Convert image to RGB mode
    image = image.convert("RGB")
    original_dimensions = image.size

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    current_size = buffered.getbuffer().nbytes

    # Reduce quality
    while current_size > max_bytes and quality >= min_quality:
        quality -= step
        buffered.seek(0)
        buffered.truncate(0)
        image.save(buffered, format="JPEG", quality=quality)
        current_size = buffered.getbuffer().nbytes

    # If still too large, reduce the image size
    if current_size > max_bytes:
        original_size = image.size
        # Calculate the scaling factor
        ratio = (max_bytes / current_size) ** 0.5
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        image = image.resize(new_size, Image.LANCZOS)
        # Reset quality to initial value
        quality = 85
        buffered.seek(0)
        buffered.truncate(0)
        image.save(buffered, format="JPEG", quality=quality)
        current_size = buffered.getbuffer().nbytes
        # Reduce quality again if necessary
        while current_size > max_bytes and quality >= min_quality:
            quality -= step
            buffered.seek(0)
            buffered.truncate(0)
            image.save(buffered, format="JPEG", quality=quality)
            current_size = buffered.getbuffer().nbytes

    # Final check
    if current_size > max_bytes:
        st.warning("Could not compress the image below the desired size. The image may be too large or too detailed.")
        st.write(f"Final image size: {current_size / 1024:.2f} KB")

    buffered.seek(0)
    return Image.open(buffered)

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

def parse_groq_stream(stream):
    for chunk in stream:
        if chunk.choices:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content


def chat_with_assistant(messages, model_choice):
    response = client.chat.completions.with_streaming_response.create(
        messages=messages,
        model=model_choice,
    )
    
    for chunk in response:  # Iterate over streamed response chunks
        if 'choices' in chunk:
            for choice in chunk['choices']:
                yield choice['message']['content']  # Yield the content for incremental display

def main():
    st.set_page_config(layout="wide")

    st.title("Groq Batch Vision")
    st.image("imgs/groqlabs_logo-black-orange.svg", width=300)

    # Initialize session state for settings
    if 'vision_system_prompt' not in st.session_state:
        st.session_state.vision_system_prompt = "You are an AI assistant that analyzes images. Describe the image in detail."
    if 'chat_system_prompt' not in st.session_state:
        st.session_state.chat_system_prompt = "You are an AI assistant that helps analyze and discuss images. You have been provided with some images that you already viewed and described to later answer questions about."

    uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        if uploaded_files:
            st.subheader("Uploaded Images")
            cols = st.columns(3)
            for idx, uploaded_file in enumerate(uploaded_files):
                with cols[idx % 3]:
                    image = Image.open(uploaded_file)
                    compressed_image = compress_image(image)
                    st.image(compressed_image, caption=f"{uploaded_file.name} (Compressed)", use_column_width=True, width=200)

        run_button = st.button("Run Analysis")

        if run_button and uploaded_files:
            descriptions = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file)
                compressed_image = compress_image(image)
                st.write(f"Analyzing {uploaded_file.name}...")
                description = analyze_image(
                    compressed_image,
                    "Describe this image in detail.",
                    "llama-3.2-11b-vision-preview",
                    st.session_state.vision_system_prompt,
                )
                st.write(f"**Description for {uploaded_file.name}:** {description}")
                descriptions.append(description)

            if len(descriptions) > 0:
                # Store descriptions in session state
                descriptions_as_string = "\n\n".join(
                    [f"{uploaded_file.name}: {desc}" for uploaded_file, desc in zip(uploaded_files, descriptions)]
                )
                st.session_state['descriptions'] = descriptions
                st.session_state['descriptions_as_string'] = descriptions_as_string

                # Initialize chat messages
                if 'messages' not in st.session_state:
                    st.session_state['messages'] = [
                        {'role': 'system', 'content': st.session_state.chat_system_prompt},
                        # Include the descriptions in the system prompt but omit from UI
                        {'role': 'assistant', 'content': f"Image descriptions have been processed."}
                    ]

    # Display chat interface if messages are initialized
    if 'messages' in st.session_state:
        st.write("## Chat with Groq Vision")

        # Improve the chat UI using st.chat_message
        for idx, message in enumerate(st.session_state['messages']):
            if idx == 1 and message['role'] == 'assistant':
                # Skip displaying the first assistant message with the descriptions
                continue
            if message['role'] == 'assistant':
                with st.chat_message("assistant"):
                    st.markdown(message['content'])
            elif message['role'] == 'user':
                with st.chat_message("user"):
                    st.markdown(message['content'])

        # Use st.chat_input for a better chat experience
        user_message = st.chat_input("Type your message here...")

        if user_message:
            # Append the user's message to session state and display it immediately
            st.session_state['messages'].append({'role': 'user', 'content': user_message})

            # Display the user's message in the chat interface
            with st.chat_message("user"):
                st.markdown(user_message)

            # Append the descriptions to the messages before sending to the assistant
            assistant_messages = st.session_state['messages'].copy()
            assistant_messages.insert(1, {'role': 'assistant', 'content': st.session_state['descriptions_as_string']})

            # Create the Groq stream and parse it with `parse_groq_stream`
            stream = client.chat.completions.create(
                model="llama-3.1-70b-versatile",
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in assistant_messages
                ],
                stream=True,
            )

            # Use a placeholder for the assistant's streaming response
            with st.chat_message("assistant"):
                assistant_reply_placeholder = st.empty()  # Placeholder for streaming text
                assistant_reply_text = ""

                # Stream the assistant's response token by token
                for token in parse_groq_stream(stream):
                    assistant_reply_text += token  # Append each token to the text
                    assistant_reply_placeholder.markdown(assistant_reply_text)  # Update the placeholder progressively

            # After streaming is complete, add the full assistant's reply to session state
            st.session_state['messages'].append({'role': 'assistant', 'content': assistant_reply_text})

            # Rerun to display the new messages
            st.rerun()

    with col2:
        with st.expander("Settings", expanded=False):
            st.session_state.vision_system_prompt = st.text_area(
                "Vision System Prompt", st.session_state.vision_system_prompt
            )
            st.session_state.chat_system_prompt = st.text_area(
                "Chat System Prompt", st.session_state.chat_system_prompt
            )

if __name__ == "__main__":
    main()
