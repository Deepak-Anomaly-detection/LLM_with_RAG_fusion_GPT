# Install Required Libraries
#!pip install -qqq langchain_openai langchain_chroma langchain_huggingface

import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Define Paths
folder = "./answer_files"  # Folder containing JSON files
db_path = "knowledge_db"  # Path to save ChromaDB

# Function to Load JSON Documents
def load_json_documents(folder):
    json_documents = []
    
    for filename in os.listdir(folder):
        if filename.endswith(".json"):
            file_path = os.path.join(folder, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)

                for key, value in data.items():
                    if isinstance(value, dict) and "question" in value:
                        text_content = f"Question: {value['question']}\nAnswer: {value.get('answer', 'No answer provided.')}\nURL: {value.get('url', 'No URL available.')}"
                        json_documents.append(Document(
                            page_content=text_content,
                            metadata={"source": filename, "title": key}
                        ))
    
    return json_documents

# Load and Process JSON Documents
print("📂 Loading JSON files...")
documents = load_json_documents(folder)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en", encode_kwargs={'normalize_embeddings': True})

# Create & Save ChromaDB
print("⚡ Creating ChromaDB...")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_path)
print(f"✅ ChromaDB saved successfully in {db_path}!")