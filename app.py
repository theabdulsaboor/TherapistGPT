from transformers import AutoTokenizer, AutoModelForCausalLM
import streamlit as st
import os
import docx  # For docx files
import PyPDF2  # For PDF files

# Function to load the model and tokenizer
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Function to generate response
def generate_response(prompt, tokenizer, model, temperature=0.7, top_k=50, top_p=0.95, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, 
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            do_sample=True)  # Enable sampling for temperature, top_k, top_p to take effect
    token_ids = output[0].cpu().numpy()
    response = tokenizer.decode(token_ids, skip_special_tokens=True)
    return response

# Set up the Streamlit app with a dark theme
st.set_page_config(page_title="Therapy Chatbot", layout="wide")
st.markdown("<style>body {background-color: #333; color: #fff;}</style>", unsafe_allow_html=True)
st.markdown("<style>.ea3mdgi5 {margin-top: -12rem;}</style>", unsafe_allow_html=True)
st.markdown("<style>p {font-size:17px; font-weight: 400;}</style>", unsafe_allow_html=True)
st.markdown("<style>.st-emotion-cache-1gwvy71 h1 {font-size:35px; font-weight: 800;}</style>", unsafe_allow_html=True)
st.markdown("<style>.st-emotion-cache-ysk9xe p {font-size:16px; font-weight: 800;}</style>", unsafe_allow_html=True)
st.markdown("<style>.stTextArea p {font-size:18px; font-weight: 600;}</style>", unsafe_allow_html=True)
st.markdown("<style>.stTextInput {margin-top: -2.2rem;}</style>", unsafe_allow_html=True)
st.markdown("<style>.stTextArea {margin-top: -1.6rem;}</style>", unsafe_allow_html=True)
st.markdown("<style>.stFileUploader {margin-top: -2rem;}</style>", unsafe_allow_html=True)
st.markdown("<style>.st-emotion-cache-1n47svx {margin-top: 1.1rem;}</style>", unsafe_allow_html=True)

# Initialize session state for spaces and chat history
if 'spaces' not in st.session_state:
    st.session_state['spaces'] = []
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = {}

# Sidebar configuration
st.sidebar.markdown("# TherapyAI")

# Select AI model
model_mapping = {
       "gpt2": "theabdulsaboor/gpt2-therapist-finetuned",
       "distilgpt2": "Mrbean01/uzair_amiin", "gptneo": "Trigonometrippin/gpt2_therapy_generator2"
   }
selected_model_name = st.sidebar.selectbox("Change Model", list(model_mapping.keys()))  
selected_model = model_mapping[selected_model_name]

# Load the selected model
tokenizer, model = load_model(selected_model)

# Update spaces dropdown
selected_space = st.sidebar.selectbox("Spaces", st.session_state['spaces'] + ["Welcome!"])

# Function to create a new space and clear the input box
def create_new_space():
    st.session_state['spaces'].append(st.session_state.new_space_name)
    st.session_state['chat_history'][st.session_state.new_space_name] = []
    st.session_state.new_space_name = ""  # Clear the input box value
    st.rerun()

# Create multiple spaces
new_space_name = st.sidebar.text_input("", placeholder="Space Name...", key="new_space_name")
if st.sidebar.button("Create New Space", on_click=create_new_space):  # Use on_click
    pass  # The actual space creation is handled in the callback function

# Main content area
col1, col2 = st.columns([0.9, 0.1])  # Adjust the column ratios as needed
with col1:
    st.markdown(f"## {selected_space}")
with col2:
    if st.button("✖", key=f"delete_{selected_space}"):  # Unique key for each button
        if selected_space in st.session_state['spaces']:
            st.session_state['spaces'].remove(selected_space)
            del st.session_state['chat_history'][selected_space]  # Remove chat history
            # After deleting, reset selected_space to "Welcome!" or a default
            st.session_state['selected_space'] = "Welcome!"  
            st.rerun()

# Initialize chat history for the selected space if it doesn't exist
if selected_space not in st.session_state['chat_history']:
    st.session_state['chat_history'][selected_space] = []

# Display chat messages from history on app rerun
for message in st.session_state['chat_history'][selected_space]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initialize a key for the text area in session state
if 'text_area_key' not in st.session_state:
    st.session_state['text_area_key'] = 0

# Text box for user input
user_prompt = st.text_area("", height=120, placeholder="Talk to your TherapistAI...", key=st.session_state['text_area_key'])
  

# File upload functionality
uploaded_file = st.file_uploader("", type=["txt", "pdf", "docx"])

if uploaded_file:
    # Process the uploaded file
    if uploaded_file.name.endswith(".txt"):
        user_prompt = uploaded_file.read().decode("utf-8")  # Decode for text files
    elif uploaded_file.name.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        user_prompt = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    elif uploaded_file.name.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        user_prompt = "\n".join([page.extract_text() for page in pdf_reader.pages])

# Button to trigger response generation
if st.button("Get Response"):
    try:
        # Display user message in chat message container
        st.session_state['chat_history'][selected_space].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Generate and display assistant response in chat message container
        response = generate_response(user_prompt, tokenizer, model, temperature=0.8, top_k=40)
        st.session_state['chat_history'][selected_space].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Increment the key to effectively reset the text area
        st.session_state['text_area_key'] += 1 
        st.rerun() 

    except Exception as e:
        st.error(f"An error occurred: {e}")
