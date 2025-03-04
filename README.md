RAG-Based Electronic Design Assistant for Renesas Electronics

Overview
This project is a Retrieval-Augmented Generation (RAG) based AI assistant designed to answer queries related to Renesas Electronics by consuming information from the Renesas KnowledgeBase: https://en-support.renesas.com/knowledgeBase.

The assistant is built using LLMs (Large Language Models) with an advanced retrieval system to fetch relevant documents, process images, and provide accurate, reference-backed answers. The bot is deployed on the cloud with a Streamlit interface.
Features

✅ Knowledge-Based Q&A: Extracts and retrieves relevant information from the Renesas KnowledgeBase. Provides precise, reference-backed answers with source links.

✅ Image Processing: Includes images from the KnowledgeBase in responses when relevant. Accepts hand-drawn sketches and user-uploaded images as input for enhanced assistance.

✅ Web-Based Search (Fallback Mechanism): If the required information is not found with high confidence, the bot can invoke another agent to search across Renesas.com for possible answers.

✅ Cloud Deployment: Deployed on a cloud platform for easy access. Uses a Streamlit UI for an interactive experience.

Architecture
1️⃣ Data Ingestion: Scrapes and indexes Renesas KnowledgeBase articles. Stores structured data in a vector database for fast retrieval.

1️⃣ Query Processing: User query is vectorized and matched against indexed data. Relevant documents and images are retrieved.

1️⃣ Answer Generation (RAG): Retrieved content is passed to an LLM for natural language response generation. Includes relevant images and reference links in responses.

1️⃣ Image-Based Q&A: Accepts hand-drawn circuit sketches or uploaded images. Uses image processing + LLM to extract insights and provide recommendations.

1️⃣ Fallback Search (Renesas.com): If no high-confidence answer is found in the KnowledgeBase, the bot triggers a web search for additional sources.

Installation
1. Clone the Repository
```bash
git clone https://github.com/yourusername/renesas-design-assistant.git
cd renesas-design-assistant
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Set Up Environment Variables
Create a `.env` file and configure the following:
```env
OPENAI_API_KEY=your_openai_api_key
VECTOR_DB_PATH=./vectorstore
```
4. Run the Streamlit App
```bash
streamlit run app.py
```
Deployment
### Cloud Deployment Options
- AWS Lambda + API Gateway
- Google Cloud Run
- Azure App Services
The model is packaged as a FastAPI backend, with a Streamlit frontend for interaction.
Usage
- Ask questions about Renesas components, circuit design, and technical documentation.
- Upload images of hand-drawn circuits to get AI-powered insights.
- Receive reference-backed answers with images and links.
- Fallback to web search when answers are unavailable in the knowledge base.
Roadmap
- [ ] Enhance image processing for better sketch recognition.
- [ ] Fine-tune LLM for improved accuracy on technical queries.
- [ ] Integrate multi-modal capabilities for deeper circuit analysis.

