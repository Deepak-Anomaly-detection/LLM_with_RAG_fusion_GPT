# Install Required Libraries
#pip install -qqq langchain_openai langchain_chroma langchain_huggingface

import os
import streamlit as st
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import asyncio
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
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

llm = ChatOpenAI(api_key="")

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

from langchain.memory import ConversationBufferMemory
from googlesearch import search  # Import web search tool

# AI Agent function (Newly Added)
def get_ai_agent_response(query):
    """
    If ChromaDB has no relevant data, this function triggers an AI Agent to search resense.com.
    """
    @tool("search_resense")
    def search_resense(query: str):
        """Searches resense.com for relevant information."""
        search_results = search(f"site:https://www.renesas.com/en {query}")

        if isinstance(search_results, str):
            return f"Web search returned an unexpected result: {search_results}"

        if isinstance(search_results, dict):
            search_results = [search_results]

        if isinstance(search_results, list):
            structured_results = []
            for res in search_results[:3]:  # Extract top 3 results safely
                if isinstance(res, dict) and "title" in res and "snippet" in res:
                    structured_results.append(f"{res['title']}: {res['snippet']}")
                else:
                    structured_results.append("Invalid result format received.")

            if structured_results:
                return "\n".join(structured_results)

        return "No relevant information found on resense.com."
    
    # Initialize AI Agent
    tools = [search_resense]
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=tools,
        llm=ChatOpenAI(api_key=""),
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        handle_parsing_errors=True
    )

    # Run the AI agent for web search
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(agent.arun(query))
    except RuntimeError:
        print("⚠️ Async execution failed. Running synchronously...")
        response = agent.run(query)  # Fallback to synchronous execution

    return response
# Function to Get AI Response
from langchain_core.messages import HumanMessage, AIMessage

def get_response(user_input, chat_history=[]):
    """
    Retrieves relevant documents, ensures they are meaningful, and generates a response.
    If no relevant documents are found, it triggers the AI agent for a web search.
    """
    print(f"🔍 Query Sent to Retriever: {user_input}")

    # Retrieve relevant documents from ChromaDB (limit results to top 5)
    retrieved_docs = retriever.invoke(user_input)[:2]  

    # 🔥 Stricter Relevance Check
    valid_docs = []
    query_words = set(user_input.lower().split())  # Extract key words from query
    
    for doc in retrieved_docs:
        text = doc.page_content.strip().lower()

        # Ensure the document has meaningful content (at least 10 words)
        if len(text.split()) < 10:
            continue

        # Ensure at least 30% of query keywords appear in the document
        match_count = sum(1 for word in query_words if word in text)
        match_percentage = match_count / len(query_words) if query_words else 0

        if match_percentage >= 0.7:  # Require at least 30% keyword match
            valid_docs.append(text)

    # ✅ If relevant documents exist, use them
    if valid_docs:
        print("✅ Found relevant documents in ChromaDB.")

        # Extract text from retrieved documents
        context = "\n".join(valid_docs)

        # Generate response using retrieved context
        response = retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": user_input,
            "context": context
        })
       
        # Append conversation history
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response["answer"]))
        
        return response["answer"]

    # ❌ No relevant documents found → Trigger AI Agent for Web Search
    print("⚠️ No relevant information found in the database. Switching to AI Agent...")
    web_response = get_ai_agent_response(user_input)
    return web_response

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
