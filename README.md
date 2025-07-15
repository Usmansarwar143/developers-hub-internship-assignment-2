# AI/ML Engineering Internship - DevelopersHub Corporation

## Overview
This repository contains the completed tasks for the AI/ML Engineering Internship at DevelopersHub Corporation, submitted as part of the Advanced Internship Tasks due on July 24, 2025. I completed three tasks: **Task 2 (End-to-End ML Pipeline for Customer Churn Prediction)**, **Task 4 (Context-Aware Chatbot with LangChain and RAG)**, and **Task 5 (Support Ticket Auto-Tagging System with Prompt Engineering)**. Each task leverages cutting-edge machine learning and AI techniques, including scikit-learn, LangChain, Streamlit, and prompt engineering, to build scalable, production-ready solutions. The projects are implemented in Python, documented with Jupyter notebooks or scripts, and designed for modularity and reproducibility.

## Table of Contents
1. [Task 2: End-to-End ML Pipeline for Customer Churn Prediction](#task-2-end-to-end-ml-pipeline-for-customer-churn-prediction)
   - [Introduction](#introduction-task-2)
   - [Objective](#objective-task-2)
   - [Dataset](#dataset-task-2)
   - [Methodology](#methodology-task-2)
   - [Key Results](#key-results-task-2)
   - [Repository Contents](#repository-contents-task-2)
   - [Installation and Usage](#installation-and-usage-task-2)
   - [Dependencies](#dependencies-task-2)
   - [Future Improvements](#future-improvements-task-2)
2. [Task 4: Context-Aware Chatbot with LangChain and RAG](#task-4-context-aware-chatbot-with-langchain-and-rag)
   - [Introduction](#introduction-task-4)
   - [Objective](#objective-task-4)
   - [Dataset](#dataset-task-4)
   - [Methodology](#methodology-task-4)
   - [Key Results](#key-results-task-4)
   - [Repository Contents](#repository-contents-task-4)
   - [Installation and Usage](#installation-and-usage-task-4)
   - [Dependencies](#dependencies-task-4)
   - [Future Improvements](#future-improvements-task-4)
3. [Task 5: Support Ticket Auto-Tagging System with Prompt Engineering](#task-5-support-ticket-auto-tagging-system-with-prompt-engineering)
   - [Introduction](#introduction-task-5)
   - [Objective](#objective-task-5)
   - [Dataset](#dataset-task-5)
   - [Methodology](#methodology-task-5)
   - [Key Results](#key-results-task-5)
   - [Repository Contents](#repository-contents-task-5)
   - [Installation and Usage](#installation-and-usage-task-5)
   - [Dependencies](#dependencies-task-5)
   - [Future Improvements](#future-improvements-task-5)
4. [Contact](#contact)

## Task 2: End-to-End ML Pipeline for Customer Churn Prediction

### Introduction (Task 2)
This project implements a production-ready machine learning pipeline to predict customer churn using the Telco Churn Dataset. Customer churn is a critical challenge for telecommunications companies, leading to revenue loss. The pipeline automates data preprocessing, trains Logistic Regression and Random Forest models, optimizes hyperparameters with GridSearchCV, and exports the pipeline using joblib for scalability. The solution is modular and suitable for customer retention systems.

### Objective (Task 2)
Develop a reusable machine learning pipeline to predict customer churn, automating preprocessing (handling missing values, encoding, scaling), training and comparing models, optimizing for F1-score to address class imbalance, and exporting the pipeline for production use.

### Dataset (Task 2)
The Telco Churn Dataset (from Kaggle) contains ~7,000 records with 20 features, including demographic details (e.g., gender), service usage (e.g., internet service), and account information (e.g., tenure, MonthlyCharges). The target variable, `Churn`, is binary (Yes/No), with ~26% churners, presenting challenges like missing values and class imbalance.

### Methodology (Task 2)
- **Data Preprocessing**: Loaded dataset, converted `TotalCharges` to numeric (handling empty strings as NaN), dropped `customerID`, and encoded `Churn` (Yes=1, No=0). Used `ColumnTransformer` for numerical (impute median, `StandardScaler`) and categorical (impute "missing", `OneHotEncoder`) features.
- **Pipeline Construction**: Built a scikit-learn `Pipeline` integrating preprocessing and a classifier (Logistic Regression or Random Forest).
- **Model Training**: Split data (80% train, 20% test, stratified), used `class_weight='balanced'` for imbalance.
- **Hyperparameter Tuning**: Applied `GridSearchCV` (5-fold CV) to tune Logistic Regression (C: [0.1, 1, 10], penalty: [l2]) and Random Forest (n_estimators: [100, 200], max_depth: [10, 20, None], min_samples_split: [2, 5]), optimizing F1-score.
- **Evaluation**: Measured precision, recall, F1-score, and AUC-ROC on the test set.
- **Export**: Saved the pipeline as `churn_pipeline.pkl` using joblib.

### Key Results (Task 2)
- **Model Performance**: Random Forest outperformed Logistic Regression, achieving an F1-score of ~0.60 (test set) and AUC-ROC of ~0.85 for churn class.
- **Feature Importance**: `tenure`, `MonthlyCharges`, `Contract`, and `PaymentMethod` were key predictors.
- **Robustness**: Pipeline handled missing values and unseen categories (`handle_unknown='ignore'`).
- **Scalability**: Exported pipeline enables fast predictions (<1s/sample) for production.

### Repository Contents (Task 2)
- `task2/main.ipynb`: Jupyter notebook with preprocessing, pipeline construction, training, tuning, evaluation, and export.
- `task2/churn_pipeline.pkl`: Trained pipeline for production use.
- `task2/WA_Fn-UseC_-Telco-Customer-Churn.csv`: Dataset.

### Installation and Usage (Task 2)
1. **Clone Repository**:
   ```bash
   git clone https://github.com/Usmansarwar143/developers-hub-internship-assignment-2
   cd developers-hub-internship-assignment-2/task2
   ```
2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run Notebook**:
   - Open `main.ipynb` in Jupyter Notebook/Lab.
   - Ensure dataset is in `task2/`.
   - Execute to train, tune, and export pipeline.
5. **Use Pipeline**:
   ```python
   import joblib
   import pandas as pd
   pipeline = joblib.load('task2/churn_pipeline.pkl')
   sample_data = pd.DataFrame({...})
   prediction = pipeline.predict(sample_data)
   ```

### Dependencies (Task 2)
- Python 3.8+
- pandas, numpy, scikit-learn, joblib, jupyter
- Install:
  ```bash
  pip install pandas numpy scikit-learn joblib jupyter
  ```

### Future Improvements (Task 2)
- Add text data (e.g., customer feedback) using Hugging Face Transformers.
- Experiment with XGBoost or LightGBM for better performance.
- Deploy via Streamlit/Flask for non-technical users.
- Explore SMOTE for class imbalance.
- Add model monitoring for production.

## Task 4: Context-Aware Chatbot with LangChain and RAG

### Introduction (Task 4)
This project develops a context-aware chatbot using LangChain and Retrieval-Augmented Generation (RAG). It leverages the `tiiuae/falcon-rw-1b` LLM and a FAISS vector store for retrieval from a custom knowledge base, with conversation history maintained via `ConversationBufferMemory`. The chatbot is deployed with a Streamlit interface for seamless user interaction, designed for modularity and extensibility.

### Objective (Task 4)
Build a chatbot that retrieves information from a custom knowledge base, maintains conversational context, and provides accurate responses. Deploy it with Streamlit for user interaction, ensuring scalability for larger corpora.

### Dataset (Task 4)
The knowledge base (`data/knowledge.txt`) contains ~500 words on AI, ML, and NLP, split into three documents. Text is chunked (500 characters, 100-character overlap) for retrieval. The system supports expansion to larger corpora (e.g., Wikipedia).

### Methodology (Task 4)
- **Data Preprocessing**: Loaded `knowledge.txt` with `TextLoader`, split into chunks using `RecursiveCharacterTextSplitter`, generated embeddings with `sentence-transformers/all-MiniLM-L6-v2`, and stored in FAISS.
- **LLM Setup**: Configured `tiiuae/falcon-rw-1b` with `transformers`, using offloading to `offload/`. Set text generation parameters: `max_new_tokens=512`, `temperature=0.5`, `top_k=50`, `top_p=0.95`.
- **RAG Pipeline**: Built `ConversationalRetrievalChain` with LLM, FAISS retriever (top-2 documents), and `ConversationBufferMemory`. Cached vector store with `@st.cache_resource`.
- **Interface**: Deployed Streamlit app for query input and response/history display.

### Key Results (Task 4)
- **Response Quality**: Accurate responses for AI/ML/NLP queries, leveraging knowledge base (e.g., sentiment analysis explained via NLP chunks).
- **Context Retention**: `ConversationBufferMemory` enabled coherent follow-up responses.
- **Retrieval Accuracy**: FAISS retrieved relevant chunks, improving response relevance.
- **Performance**: `falcon-rw-1b` ran on 4GB+ RAM, with ~1-2s/query response time.
- **Scalability**: Pipeline supports larger corpora; Streamlit interface is user-friendly.

### Repository Contents (Task 4)
- `task4/app.py`: Streamlit app for RAG chatbot.
- `task4/data/knowledge.txt`: Sample knowledge base.
- `task4/requirements.txt`: Dependencies.
- `task4/.env`: Environment file for Hugging Face API key (optional).
- `task4/offload/`: Model offloading directory (created after loading).

### Installation and Usage (Task 4)
1. **Clone Repository**:
   ```bash
   git clone https://github.com/Usmansarwar143/developers-hub-internship-assignment-2
   cd developers-hub-internship-assignment-2/task4
   ```
2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare Knowledge Base**:
   - Ensure `data/knowledge.txt` exists or replace with custom corpus.
5. **Run Application**:
   ```bash
   streamlit run app.py
   ```
   - Access `http://localhost:8501` to interact with the chatbot.

### Dependencies (Task 4)
- Python 3.8+
- streamlit==1.38.0, langchain==0.2.16, langchain-community==0.2.16, faiss-cpu==1.8.0, sentence-transformers==3.2.0, transformers==4.44.2, torch==2.4.1
- Install:
  ```bash
  pip install streamlit==1.38.0 langchain==0.2.16 langchain-community==0.2.16 faiss-cpu==1.8.0 sentence-transformers==3.2.0 transformers==4.44.2 torch==2.4.1
  ```

### Future Improvements (Task 4)
- Use larger LLM (e.g., mistralai/Mixtral-8x7B-Instruct-v0.1) for better responses.
- Expand corpus with WikipediaLoader or PyPDFLoader.
- Add CLI option for terminal-based interaction.
- Save conversation history to a database.
- Use quantization (e.g., bitsandbytes) for memory optimization.

## Task 5: Support Ticket Auto-Tagging System with Prompt Engineering

### Introduction (Task 5)
This project implements a support ticket auto-tagging system using prompt engineering. It assigns the top three tags (e.g., Billing, Technical Issue) to tickets using zero-shot and few-shot learning with a mock LLM based on keyword matching, designed for integration with a real LLM (e.g., xAI’s Grok API). Outputs are JSON-formatted with confidence scores, suitable for customer support workflows.

### Objective (Task 5)
Develop an automated system to tag support tickets with the top three most relevant tags, comparing zero-shot and few-shot performance using prompt engineering. Ensure extensibility for real-world datasets and LLMs.

### Dataset (Task 5)
Mock dataset (`MOCK_DATASET`) in `main.py` with five tickets and ground-truth tags (e.g., Billing, Technical Issue, Account Management). Example: “I was charged twice for my subscription” (Tags: Billing, General Inquiry, Account Management). Supports expansion to larger datasets via CSV/JSON.

### Methodology (Task 5)
- **Data Preprocessing**: Used raw ticket text from `MOCK_DATASET`, requiring no additional preprocessing.
- **LLM Setup**: Built a mock LLM with keyword-based scoring (e.g., “charge” for Billing). Designed prompts for zero-shot (no examples) and few-shot (four example tickets) classification.
- **Tagging Pipeline**: Processed tickets to output JSON with top three tags and confidence scores (0-100%). Ensured three tags per ticket.
- **Evaluation**: Calculated accuracy by comparing primary predicted tag to ground-truth tags.

### Key Results (Task 5)
- **Tagging Accuracy**: Few-shot achieved ~80% accuracy, zero-shot ~60%, due to example-driven pattern recognition.
- **Response Quality**: Mock LLM correctly identified primary tags but struggled with secondary tags.
- **Performance**: Processed tickets in <1s/ticket on 4GB+ RAM.
- **Scalability**: Pipeline supports custom datasets and real LLM integration.
- **Output Example**:
  ```json
  {
    "ticket": "I was charged twice for my subscription this month. Please help!",
    "tags": [
      {"tag": "Billing", "confidence": 52},
      {"tag": "General Inquiry", "confidence": 28},
      {"tag": "Account Management", "confidence": 19}
    ]
  }
  ```

### Repository Contents (Task 5)
- `task5/main.py`: Script for ticket tagging and evaluation.
- `task5/requirements.txt`: Dependencies.

### Installation and Usage (Task 5)
1. **Clone Repository**:
   ```bash
   git clone https://github.com/Usmansarwar143/developers-hub-internship-assignment-2
   cd developers-hub-internship-assignment-2/task5
   ```
2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare Dataset**:
   - Mock dataset in `main.py`. For custom data:
     ```python
     import pandas as pd
     df = pd.read_csv("tickets.csv")
     MOCK_DATASET = [{"ticket": row["text"], "true_tags": row["tags"].split(",")} for _, row in df.iterrows()]
     ```
5. **Run Application**:
   ```bash
   python main.py
   ```
   - Outputs tagged results and accuracy.
6. **Tag Single Ticket**:
   ```python
   from main import process_ticket
   import json
   result = process_ticket("My app won’t load.", "few_shot")
   print(json.dumps(result, indent=2))
   ```

### Dependencies (Task 5)
- Python 3.8+
- requests==2.32.3 (for real LLM integration)
- Install:
  ```bash
  pip install requests==2.32.3
  ```

### Future Improvements (Task 5)
- Integrate real LLM (e.g., xAI’s Grok 3 via https://x.ai/api).
- Support larger datasets via CSV/JSON.
- Add Streamlit/Flask interface for interactive tagging.
- Enhance evaluation with precision, recall, and F1-score.
- Cache prompt results for efficiency.

## Contact
For questions, contact [Usmansarwar143] via GitHub or email. 
### Repository: https://github.com/Usmansarwar143/developers-hub-internship-assignment-2
