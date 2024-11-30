import streamlit as st
import joblib
import numpy as np
from langchain.chains import RetrievalQA
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

# Load pre-trained potability prediction model
model = joblib.load("water_potability_model.pkl")

# FAISS Index (Assume this is already created and saved)
index = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Prompt Template
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Using the information provided:\n{context}\nAnswer the question:\n{question}"
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Function to predict water potability
def predict_potability(features):
    prediction = model.predict([features])[0]
    return "Potable" if prediction == 1 else "Not Potable"

# Function to retrieve similar documents
def retrieve_similar_documents(query, k=5):
    retriever = index.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    return "\n".join([doc.page_content for doc in docs])

# Streamlit App
st.title("Water Potability Analysis & Knowledge Retrieval")

# Section 1: Potability Prediction
st.header("Water Potability Prediction")
features = st.text_input("Enter features (comma-separated)", "7.2, 204.5, 20791, 7.2, 333.3, 17.2, 6.2, 325.3, 0.5")
if st.button("Predict Potability"):
    features_list = list(map(float, features.split(",")))
    prediction = predict_potability(features_list)
    st.write(f"Water is predicted to be: **{prediction}**")

# Section 2: Question Answering
st.header("Ask Questions About Water Quality")
user_query = st.text_input("Enter your question")
if st.button("Get Answer"):
    # Retrieve similar documents
    retrieved_context = retrieve_similar_documents(user_query)

    # Generate answer with LLM
    chain = RetrievalQA(llm=llm, retriever=index.as_retriever())
    response = chain.run({"context": retrieved_context, "question": user_query})

    # Display results
    st.subheader("Retrieved Context")
    st.write(retrieved_context)
    st.subheader("Generated Answer")
    st.write(response)
