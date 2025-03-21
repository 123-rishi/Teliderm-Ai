import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Concatenate
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Error: GOOGLE_API_KEY is missing! Please set it in your .env file.")
genai.configure(api_key=api_key)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()   #save each text from pdf in page_text
            if page_text:
                text += page_text + "\n"      #if other pages exists
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)   #whole text is been divided into chunks using this langchain and been saved

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not available in the provided context, say "answer is not available in the context".
    
    Context:
    {context}?
    
    Question:
    {question}
  
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


#FAISS
def user_input(user_question):
    if not os.path.exists("faiss_index"):
        st.error("Error: 'faiss_index' not found! Please process PDFs first.")
        return ""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])
    return response["output_text"]

#tensorflow and other models
def predict_image(image):
    if not os.path.exists("my_model_weight.weights.h5"):
        raise FileNotFoundError("Model weight file 'my_model_weight.weights.h5' not found!")
    
    num_classes = 23
    image_resize = 224
    input_shape = (image_resize, image_resize, 3)
    
    input_layer = Input(shape=input_shape)
    resnet_base = ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
    efficientnet_base = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    
    resnet_output = resnet_base(input_layer)
    efficientnet_output = efficientnet_base(input_layer)
    resnet_gap = GlobalAveragePooling2D()(resnet_output)
    efficientnet_gap = GlobalAveragePooling2D()(efficientnet_output)
    combined_output = Concatenate()([resnet_gap, efficientnet_gap])
    combined_output = Dropout(0.5)(combined_output)
    dense_layer = Dense(256, activation='relu')(combined_output)
    output_layer = Dense(num_classes, activation='softmax')(dense_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.load_weights("my_model_weight.weights.h5")
    
    img = cv2.resize(image, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_class, confidence

#streamlit
def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with TelidermAI to get Your Doubts Cleared💁")
    
    user_question = st.text_input("Ask a Question related to the diseases")
    if user_question:
        user_input(user_question)
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if pdf_docs and st.button("Submit & Process"):
            with st.spinner("Creating Embeddings..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")
        
        st.title("Image Prediction")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image is None:
                st.error("Error loading image! Please upload a valid image file.")
            else:
                st.image(image, caption='Uploaded Image', use_column_width=True)
                if st.button('Predict'):
                    with st.spinner("Predicting..."):
                        predicted_class, confidence = predict_image(image)
                    st.write(f"Predicted Class Index: {predicted_class}")
                    st.write(f"Confidence: {confidence}")

if _name_ == "_main_":
    main()
