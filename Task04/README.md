# Context-Aware Chatbot with LangChain and RAG

## Introduction
This repository contains a context-aware chatbot built using LangChain and Retrieval-Augmented Generation (RAG). The chatbot leverages a local large language model (LLM), `tiiuae/falcon-rw-1b`, to generate responses, augmented by a custom knowledge base stored in a FAISS vector store for retrieval. It maintains conversation history using LangChain's `ConversationBufferMemory` to ensure context-aware interactions. The application is deployed with a Streamlit web interface, allowing users to interact with the chatbot seamlessly. The project is designed to be modular, reproducible, and extensible for various knowledge bases, making it suitable for applications requiring intelligent, context-sensitive responses.

## Table of Contents
- Introduction
- Objective of the Task
- Dataset
- Methodology / Approach
- Key Results or Observations
- Repository Contents
- Installation and Usage
- Dependencies
- Future Improvements
- Contact

## Objective of the Task
The objective of this project is to develop a context-aware chatbot that retrieves relevant information from a custom knowledge base and maintains conversational context. The chatbot uses LangChain's RAG framework to combine document retrieval (via FAISS and HuggingFace embeddings) with a local LLM (`tiiuae/falcon-rw-1b`) for response generation. The Streamlit interface provides a user-friendly way to interact with the chatbot, while the modular design allows for easy updates to the knowledge base or model.

## Dataset
The knowledge base is a custom text file (`data/knowledge.txt`) containing information about Artificial Intelligence (AI), Machine Learning (ML), and Natural Language Processing (NLP). The dataset is small (~500 words across three documents) for demonstration purposes, with each document describing a specific topic (e.g., AI overview, ML basics, NLP applications). The text is split into chunks (500 characters with 100-character overlap) to optimize retrieval. The system supports expansion to larger corpora (e.g., Wikipedia pages or internal documents) using LangChain's document loaders.

## Methodology / Approach
The project follows a structured workflow implemented in `app.py`:

### Data Preprocessing:
- Loaded the knowledge base from `data/knowledge.txt` using `TextLoader`.
- Split documents into chunks using `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 100).
- Generated embeddings with `sentence-transformers/all-MiniLM-L6-v2` and stored them in a FAISS vector store for efficient retrieval.

### LLM Setup:
- Used the `tiiuae/falcon-rw-1b` model from HuggingFace, loaded locally with `transformers`.
- Configured model offloading to the `offload/` directory to manage memory usage.
- Created a text generation pipeline with `max_new_tokens=512`, `temperature=0.5`, `top_k=50`, and `top_p=0.95` for controlled response generation.
- Integrated the pipeline with LangChain using `HuggingFacePipeline`.

### RAG Pipeline:
- Built a `ConversationalRetrievalChain` combining the local LLM, FAISS retriever (top-2 documents), and `ConversationBufferMemory` for context retention.
- Cached the vector store with Streamlit's `@st.cache_resource` for performance.

### Interface:
- Deployed the chatbot with a Streamlit web interface, featuring a text input for user queries and a display of conversation history.

The approach ensures modularity, context awareness, and efficient retrieval, suitable for knowledge-driven conversational tasks.

## Key Results or Observations
- **Response Quality**: The `tiiuae/falcon-rw-1b` model, combined with RAG, provides accurate responses for queries related to AI, ML, and NLP, leveraging the knowledge base effectively.
- **Context Retention**: The `ConversationBufferMemory` successfully maintains conversation history, enabling coherent follow-up responses.
- **Retrieval Accuracy**: The FAISS vector store retrieves relevant document chunks (e.g., NLP applications for queries about sentiment analysis), improving response relevance.
- **Performance**: The lightweight `falcon-rw-1b` model runs on modest hardware (4GB+ RAM), with offloading reducing memory usage. Response times are ~1-2 seconds per query.
- **Scalability**: The RAG pipeline supports larger knowledge bases, and the Streamlit interface is user-friendly for non-technical users.
- **Limitations**: The small model may generate less nuanced responses compared to larger models (e.g., Mixtral-8x7B). The knowledge base size limits the scope of answers.

## Repository Contents
- `app.py`: Main Streamlit application implementing the RAG chatbot with LangChain and `tiiuae/falcon-rw-1b`.
- `data/knowledge.txt`: Sample knowledge base with AI, ML, and NLP information.
- `requirements.txt`: List of Python dependencies for the project.
- `.env`: Environment file for storing the HuggingFace API key (not required for local model but included for extensibility).
- `offload/`: Directory for model offloading to manage memory(Will be formed after loading model).

## Installation and Usage

### Clone the Repository:
```bash
git https://github.com/Usmansarwar143/developers-hub-internship-assignment-2/Task04
cd your-repo-name
```

### Set Up a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```
Alternatively, install manually (see Dependencies).

### Prepare the Knowledge Base:
Ensure `data/knowledge.txt` exists. You can use the provided sample or replace it with your own corpus.

### Run the Application:
```bash
streamlit run app.py
```
Open the provided URL (e.g., `http://localhost:8501`) in a browser to interact with the chatbot.

### Usage:
- Enter a question in the text input (e.g., "What is machine learning?").
- View the chatbot’s response and conversation history in the Streamlit interface.
- The chatbot retrieves relevant information from `knowledge.txt` and maintains context for follow-up questions.

## Dependencies
- Python 3.8+
- streamlit==1.38.0
- langchain==0.2.16
- langchain-community==0.2.16
- faiss-cpu==1.8.0
- sentence-transformers==3.2.0
- transformers==4.44.2
- torch==2.4.1

Install via:
```bash
pip install streamlit==1.38.0 langchain==0.2.16 langchain-community==0.2.16 faiss-cpu==1.8.0 sentence-transformers==3.2.0 transformers==4.44.2 torch==2.4.1
```

## Future Improvements
- **Larger Model**: Replace `falcon-rw-1b` with a more powerful model (e.g., `mistralai/Mixtral-8x7B-Instruct-v0.1`) for better response quality, if hardware allows.
- **Expanded Corpus**: Integrate larger datasets (e.g., Wikipedia or internal documents) using LangChain loaders like `WikipediaLoader` or `PyPDFLoader`.
- **CLI Option**: Add a command-line interface for users who prefer terminal-based interaction.
- **Persistence**: Save conversation history to a file or database for long-term use.
- **Performance Optimization**: Use quantization (e.g., `bitsandbytes`) to reduce memory usage for larger models.
- **Multimodal Support**: Incorporate image or audio inputs using multimodal LLMs.

## Contact
For questions, feedback, or collaboration, please reach out:

- **Email**: [Your Email](mailto:muhammadusman.becsef22@iba-suk.edu.pk)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/muhammad-usman-018535253)
- **GitHub**: [Your GitHub Profile](https://www.github.com/Usmansarwar143)

Contributions and suggestions are welcome! Please create an issue or submit a pull request.
