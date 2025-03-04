# Install Required Libraries
#!pip install -qqq langchain_openai langchain_chroma langchain_huggingface

import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
#from langchain_community.embeddings import HuggingFaceEmbeddings
st.set_page_config(page_title="AI-Powered RAG Chatbot", layout="wide")
st.title("💬 AI Chatbot with ChromaDB & RAG")
# Load Existing ChromaDB
db_path = "knowledge_db"
print(f"🔍 Loading existing ChromaDB from {db_path}...")

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en", encode_kwargs={'normalize_embeddings': True})
print("✅ HuggingFaceEmbeddings Loaded Successfully!")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en", encode_kwargs={'normalize_embeddings': True})
vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# Initialize OpenAI LLM
from langchain_openai import ChatOpenAI
#from google.colab import userdata


llm = ChatOpenAI(api_key="Enter your OpenAI Key")

# Create Retrieval Chain
prompt_search_query = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("system", "Based on the above conversation, generate a concise and relevant search query "
               "to retrieve the most useful information related to the discussion."),
])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt_search_query)

# Define Answer Generation Prompt
prompt_get_answer = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an AI assistant that answers user queries based on the given context.\n\n"
     "{context}\n\n"
     "If no relevant document is found, respond with 'I could not find relevant information in the database.'\n"
     "Do not generate an answer if the retrieved context is not relevant.\n"
     "If an image URL is available related to the answer, attach the image to the response.\n"
     "Also add the page number of the document where the answer was found.\n"
     "Do not include system messages or raw context in your response."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# Create Document and Retrieval Chains
document_chain = create_stuff_documents_chain(llm, prompt_get_answer)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)

# Function to Get AI Response
from langchain_core.messages import HumanMessage, AIMessage

def get_response(user_input, chat_history=[]):
    """
    Retrieves relevant documents, ensures they are meaningful, and generates a response.
    """
    # Ensure `user_input` is a string
    if isinstance(user_input, dict):
        print("❌ ERROR: `user_input` is a dictionary, expected a string.")
        user_input = user_input.get("query", "")

    print(f"🔍 Query Sent to Retriever: {user_input}")

    # Retrieve relevant documents
    retrieved_docs = retriever.invoke(user_input)

    # Check if retrieval returned results
    if not retrieved_docs or len(retrieved_docs) == 0:
        return "⚠️ No relevant information found in the database. Try refining your query."

    # Extract text from retrieved documents
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # Generate response using retrieved context
    response = retrieval_chain.invoke({
        "chat_history": chat_history,
        "input": user_input,
        "context": context
    })

    # Append messages in the correct format
    chat_history.append(HumanMessage(content=user_input))  # ✅ Proper user message
    chat_history.append(AIMessage(content=response["answer"]))  # ✅ Proper AI response

    return response["answer"]

#user_input = st.sidebar.text_area("Enter your question:", key="user_input_textarea")
chat_history = st.session_state.get("chat_history", [])
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# Unique key for text input
user_input = st.sidebar.text_area("Enter your question:", key="user_input_textarea")

if st.sidebar.button("Submit", key="submit_button"):
    if user_input:
        with st.spinner("Fetching response..."):
            retrieved_docs = retriever.invoke(user_input)

            if retrieved_docs and len(retrieved_docs) > 0:
                st.subheader("📚 Retrieved Documents:")
                for doc in retrieved_docs:
                    st.write(f"🔹 {doc.page_content[:200]}...")  # Display snippet of document
                
                # Get response from AI
                response_data = get_response(user_input, st.session_state["chat_history"])

                # 🔍 Debugging output
                print("DEBUG: Full Response Data ->", response_data)

                # ✅ Ensure response_data is correctly formatted
                if isinstance(response_data, str):  
                    response_text = response_data  # Just store the string response
                    response_images = []  # No images available
                else:
                    response_text = response_data.get("answer", "⚠️ No answer found.")
                    response_images = response_data.get("images", [])

                # 🔍 Debugging extracted images
                print("DEBUG: Extracted Images ->", response_images)

                # **Only add new messages to chat history**
                if len(st.session_state["chat_history"]) == 0 or st.session_state["chat_history"][-1].content != response_text:
                    st.session_state["chat_history"].append(HumanMessage(content=user_input))
                    st.session_state["chat_history"].append(AIMessage(content=response_text))

                    # Store images in chat history correctly
                    if response_images:
                        st.session_state["chat_history"].append({"type": "image", "urls": response_images})

                    # 🔍 Debugging chat history updates
                    print("DEBUG: Updated Chat History ->", st.session_state["chat_history"])

            else:
                response_text = "⚠️ No relevant information found in the database. Please refine your query."
                if len(st.session_state["chat_history"]) == 0 or st.session_state["chat_history"][-1].content != response_text:
                    st.session_state["chat_history"].append(HumanMessage(content=user_input))
                    st.session_state["chat_history"].append(AIMessage(content=response_text))

# Display AI response (Fix: Prevent duplicate rendering)
st.subheader("📝 Bot Response:")
for message in st.session_state["chat_history"]:
    if isinstance(message, HumanMessage):
        st.write(f"**User:** {message.content}")
    elif isinstance(message, AIMessage):
        st.write(f"**Response:** {message.content}")
    elif isinstance(message, dict) and message.get("type") == "image":  # Handle images
        st.subheader("📷 Relevant Image(s):")
        for img_url in message["urls"]:
            print("DEBUG: Displaying Image ->", img_url)  # 🔍 Debugging Output
            st.image(img_url, caption="Reference Image", use_column_width=True)
# Example Query
#user_input = "Is there any limitation when using Double-Precision Floating-Point Coprocessor on RXv3 devices (RX66N, RX72N, RX72M)?"
#response = get_response(user_input)
#print("🤖 AI Response:\n", response)
