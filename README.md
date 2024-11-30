# AI_Project

## RAG Implementation with LangChain

### Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline using the LangChain framework. The code leverages a publicly available blog to retrieve information and generate context-aware responses using OpenAI's GPT-3.5 model.

### Features

Web Scraping: Extracts relevant content from a given webpage.
Text Splitting: Divides the content into manageable chunks for processing.
Vector Storage: Embeds and stores the chunks in a vector database (Chroma) for efficient retrieval.
RAG Pipeline: Combines information retrieval with generative AI to answer specific questions based on the webpage content.

### Prerequisites

Before running the code, ensure the following:

- Python version: Python 3.8 or later.
- API Key: An OpenAI API key. Obtain it from OpenAI.
- Dependencies: Install the required Python libraries.

### Installation

1. Clone this repository

2. Install dependencies:
   pip install --quiet --upgrade langchain langchain-community langchain-chroma
   pip install -qU langchain-openai

3. Set your OpenAI API key:

- Option 1: Add the API key as an environment variable

- Option 2(not secure, especially if you share it on github on somewhere public): Hardcode it in the script:
  os.environ["OPENAI_API_KEY"] = "your_openai_api_key"

### Usage

1. Run the script in a Python environment or Jupyter Notebook, by clicking on Run all

2. The script performs the following steps:

- Loads and processes the content of the given webpage.
- Splits the text into smaller chunks for efficient storage and retrieval.
- Embeds the chunks and stores them in a Chroma vector database.
- Uses the RAG pipeline to retrieve relevant snippets and generate a response to a question.

3. Sample invocation:

rag_chain.invoke("quelle est la qualité de l'eau?")

-> Output: The pipeline retrieves relevant information from the blog and generates a response using GPT-3.5.

### ML model training:

1. Download Dataset:

- Place the water_potability.csv dataset in the project directory.

2. Train the Model:

- The code provided includes data preprocessing, model training, and saving the model for reuse.

## Outputs

- Potable: The water meets potability standards.
- Not Potable: The water does not meet potability standards.
