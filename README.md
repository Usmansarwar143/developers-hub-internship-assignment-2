# Customer Churn Prediction with Scikit-learn Pipeline

## Introduction
This repository contains an end-to-end machine learning pipeline for predicting customer churn using the Telco Churn Dataset. Customer churn, where customers discontinue their services, is a significant challenge for telecommunications companies, leading to revenue loss and increased acquisition costs. This project leverages scikit-learn’s Pipeline API to preprocess the dataset, train and evaluate Logistic Regression and Random Forest models, optimize hyperparameters using GridSearchCV, and export the pipeline using joblib for production-ready deployment. The goal is to provide a modular, reproducible, and scalable solution to predict churn and support proactive retention strategies. The repository includes a Jupyter notebook (`main.ipynb`), the trained pipeline (`churn_pipeline.pkl`), and the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).

## Table of Contents
- [Introduction](#introduction)
- [Objective of the Task](#objective-of-the-task)
- [Dataset](#dataset)
- [Methodology / Approach](#methodology--approach)
- [Key Results or Observations](#key-results-or-observations)
- [Repository Contents](#repository-contents)
- [Installation and Usage](#installation-and-usage)
- [Dependencies](#dependencies)
- [Future Improvements](#future-improvements)
- [Contact](#contact)

## Objective of the Task
The objective of this project is to develop a reusable, production-ready machine learning pipeline to predict customer churn in the Telco Churn Dataset. The pipeline automates data preprocessing (handling missing values, encoding categorical features, and scaling numerical features), trains and compares Logistic Regression and Random Forest models, and optimizes their hyperparameters using GridSearchCV with a focus on the F1-score to address class imbalance. The final pipeline is exported using joblib, ensuring modularity and scalability for real-world applications, such as customer retention systems in the telecommunications industry.

## Dataset
The [Telco Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) from Kaggle contains approximately 7,000 records of customer data, including demographic details (e.g., gender, senior citizen status), service usage (e.g., internet service, contract type), and account information (e.g., tenure, monthly charges). The target variable, `Churn`, is binary (Yes/No), indicating whether a customer has discontinued their service. The dataset includes 20 features, with a mix of numerical (e.g., `tenure`, `MonthlyCharges`, `TotalCharges`) and categorical (e.g., `Contract`, `PaymentMethod`) variables, and presents challenges like missing values and class imbalance (~26% churners).

## Methodology / Approach
The project follows a structured machine learning workflow implemented in `main.ipynb`:
1. **Data Preprocessing**:
   - Loaded the dataset and converted `TotalCharges` to numeric, handling empty strings as NaN.
   - Dropped the non-predictive `customerID` column and encoded the target (`Churn`: Yes=1, No=0).
   - Used `ColumnTransformer` to apply:
     - Numerical pipeline: Impute missing values with median, scale with `StandardScaler`.
     - Categorical pipeline: Impute missing values with “missing”, encode with `OneHotEncoder` (drop first category to avoid multicollinearity).
2. **Pipeline Construction**:
   - Built a scikit-learn `Pipeline` combining preprocessing and a classifier (Logistic Regression or Random Forest).
3. **Model Training**:
   - Split data into 80% training and 20% test sets with stratified sampling to maintain class distribution.
   - Trained Logistic Regression and Random Forest, using `class_weight='balanced'` to address class imbalance.
4. **Hyperparameter Tuning**:
   - Used `GridSearchCV` with 5-fold cross-validation to tune:
     - Logistic Regression: `C` (0.1, 1, 10), `penalty` (l2).
     - Random Forest: `n_estimators` (100, 200), `max_depth` (10, 20, None), `min_samples_split` (2, 5).
   - Optimized for F1-score due to class imbalance.
5. **Evaluation**:
   - Evaluated the best model on the test set using precision, recall, F1-score, and AUC-ROC.
6. **Export**:
   - Saved the best pipeline as `churn_pipeline.pkl` using joblib for reuse.
The approach ensures modularity, reproducibility, and scalability, with the pipeline handling all preprocessing and modeling steps consistently.

## Key Results or Observations
- **Model Performance**: The Random Forest model generally outperformed Logistic Regression, achieving an F1-score of ~0.55 on the cross-validation set and ~0.60 on the test set for the positive class (churn), with an AUC-ROC of ~0.85. Logistic Regression had lower performance due to its sensitivity to non-linear relationships.
- **Class Imbalance**: The dataset’s imbalance (~26% churners) was mitigated using `class_weight='balanced'`, improving recall for the churn class.
- **Feature Importance**: Random Forest feature importance indicated that `tenure`, `MonthlyCharges`, `Contract`, and `PaymentMethod` were key predictors of churn.
- **Pipeline Robustness**: The pipeline handled missing values and unseen categorical values (`handle_unknown='ignore'`) effectively, ensuring robustness for production use.
- **Hyperparameter Tuning**: GridSearchCV identified optimal parameters (e.g., `max_depth=10`, `n_estimators=200` for Random Forest), balancing model complexity and performance.
- **Scalability**: The exported pipeline (`churn_pipeline.pkl`) enables seamless predictions on new data, suitable for integration into production systems.

## Repository Contents
- `main.ipynb`: Jupyter notebook containing the complete code for data preprocessing, pipeline construction, model training, hyperparameter tuning, evaluation, and pipeline export.
- `churn_pipeline.pkl`: The trained and optimized machine learning pipeline, saved using joblib for production use.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The Telco Churn Dataset used for training and evaluation.

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install required packages manually (see [Dependencies](#dependencies)).
3. **Run the Notebook**:
   - Open `main.ipynb` in Jupyter Notebook or JupyterLab.
   - Ensure the dataset (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) is in the same directory.
   - Execute the notebook to preprocess data, train models, tune hyperparameters, and export the pipeline.
4. **Use the Pipeline**:
   - Load the saved pipeline for predictions:
     ```python
     import joblib
     import pandas as pd
     pipeline = joblib.load('churn_pipeline.pkl')
     sample_data = pd.DataFrame({...})  # Input data
     prediction = pipeline.predict(sample_data)
     ```

## Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- joblib
- jupyter (for running the notebook)

Install via:
```bash
pip install pandas numpy scikit-learn joblib jupyter
```

## Future Improvements
- **Feature Engineering**: Incorporate text data (e.g., customer feedback) using Hugging Face Transformers or LLMs to extract sentiment or topics as features.
- **Advanced Models**: Experiment with ensemble methods (e.g., XGBoost, LightGBM) or stacking to improve performance.
- **Deployment**: Build a web interface using Streamlit or Gradio to allow non-technical users to input customer data and view predictions.
- **Imbalance Handling**: Explore SMOTE or other oversampling techniques to further address class imbalance.
- **Monitoring**: Add model monitoring and drift detection for production deployment.

## Contact
For questions, feedback, or collaboration, please reach out:
- **Email**: [mailto:muhammadusman.becsef22@iba-suk.edu.pk](mailto:muhammadusman.becsef22@iba-suk.edu.pk)
- **LinkedIn**: [My LinkedIn Profile](https://www.linkedin.com/in/muhammad-usman-018535253/) 
- **Company Website**: [Company Website](https://www.developershub.com) 
- **Company LinkedIn**: [Company LinkedIn](https://www.linkedin.com/company/developershub-corporation/)
- **GitHub**: [MyGitHub Profile](https://github.com/Usmansarwar143)

Contributions and suggestions are welcome! Please create an issue or pull request in this repository.
