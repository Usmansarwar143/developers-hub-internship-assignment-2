# Support Ticket Auto-Tagging System with Prompt Engineering

## Introduction
This repository contains a support ticket auto-tagging system built using Python and prompt engineering techniques. The system leverages a large language model (LLM) to classify customer support tickets into predefined categories (e.g., Billing, Technical Issue) using zero-shot and few-shot learning approaches. A mock LLM based on keyword matching is implemented for demonstration, but the system is designed to integrate with a real LLM (e.g., xAI’s Grok API) for production use. The project outputs the top 3 most probable tags per ticket with confidence scores in JSON format and compares the performance of zero-shot and few-shot methods. The modular design allows for easy extension to custom datasets or additional tags, making it suitable for automated customer support workflows.

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
The objective of this project is to develop an automated system for tagging customer support tickets using prompt engineering with an LLM. The system assigns the top 3 most relevant tags (e.g., Billing, Technical Issue, Account Management) to each ticket based on its text content. It implements zero-shot and few-shot learning to compare their performance, avoiding the need for fine-tuning. The output is a JSON object per ticket with tags and confidence scores, enabling integration into customer support pipelines. The system is designed to be lightweight, extensible, and adaptable to real-world datasets.

## Dataset
The dataset is a mock collection of 5 support tickets stored in `main.py` as a Python list (`MOCK_DATASET`). Each ticket includes a free-text description and ground-truth tags for evaluation. Example tickets include:
- "I was charged twice for my subscription this month. Please help!" (Tags: Billing, General Inquiry, Account Management)
- "The app crashes every time I try to upload a photo." (Tags: Technical Issue, Feature Request, General Inquiry)
The tags used are: Billing, Technical Issue, Account Management, Feature Request, and General Inquiry. The dataset is small for demonstration but can be replaced with a larger corpus (e.g., CSV or JSON files) using standard Python data loading techniques.

## Methodology / Approach
The project follows a structured workflow implemented in `main.py`:

### Data Preprocessing:
- Defined a mock dataset (`MOCK_DATASET`) with 5 tickets and their ground-truth tags for testing.
- Tickets are processed as raw text strings, requiring no additional preprocessing for the mock LLM.

### LLM Setup:
- Implemented a mock LLM using keyword-based scoring to simulate tag prediction. Keywords (e.g., "charge" for Billing, "crash" for Technical Issue) are matched against ticket text to assign confidence scores.
- Designed prompts to be compatible with a real LLM (e.g., xAI’s Grok API) for future integration.
- Zero-shot prompt: Instructs the LLM to classify tickets without examples, relying on its general understanding.
- Few-shot prompt: Includes 4 example tickets with tagged outputs to guide the LLM, improving accuracy for nuanced cases.

### Tagging Pipeline:
- Processes each ticket using zero-shot or few-shot prompts.
- Outputs a JSON object per ticket with the ticket text and top 3 tags with confidence scores (0-100%).
- Ensures exactly 3 tags per ticket by assigning low-confidence tags if needed.

### Evaluation:
- Compares zero-shot and few-shot performance by calculating accuracy (percentage of tickets where the primary predicted tag matches a ground-truth tag).
- Uses a simple keyword-based mock LLM for demonstration, with logic to boost scores for few-shot patterns based on example tickets.

The approach emphasizes prompt engineering over fine-tuning, ensuring flexibility and ease of adaptation to new tag sets or datasets.

## Key Results or Observations
- **Tagging Accuracy**: The few-shot approach achieves higher accuracy (~80%) than zero-shot (~60%) on the mock dataset, as the examples help the mock LLM recognize patterns (e.g., "password" and "reset" for Account Management).
- **Response Quality**: The mock LLM correctly identifies primary tags for most tickets (e.g., "Billing" for payment-related issues) but may assign less accurate secondary tags due to its simplicity.
- **Performance**: The script runs efficiently on modest hardware (e.g., 4GB+ RAM), with processing times of <1 second per ticket.
- **Scalability**: The system supports custom datasets and tag sets, with prompts designed for real LLM integration (e.g., xAI’s Grok API).
- **Limitations**: The mock LLM relies on keyword matching, which is less robust than a real LLM. Confidence scores are heuristic-based and may not reflect true probabilities. The small dataset limits generalizability.

## Repository Contents
- `main.py`: Main Python script implementing the ticket tagging system with zero-shot and few-shot prompt engineering.
- `README.md`: This documentation file describing the project, setup, and usage.

## Installation and Usage
### Clone the Repository:
```bash
git clone https://github.com/Usmansarwar143/developers-hub-internship-assignment-2/Task#05
cd your-repo-name
```
*Note*: Replace `https://github.com/YourUsername/your-repo-name` with the actual repository URL. If the project is not yet hosted, create a GitHub repository and push the code.

### Set Up a Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```
Alternatively, install manually (see [Dependencies](#dependencies)).

### Prepare the Dataset:
- The mock dataset is embedded in `main.py` (`MOCK_DATASET`). To use a custom dataset, modify `MOCK_DATASET` or load data from a file (e.g., CSV or JSON). Example:
  ```python
  import pandas as pd
  df = pd.read_csv("tickets.csv")
  MOCK_DATASET = [{"ticket": row["text"], "true_tags": row["tags"].split(",")} for _, row in df.iterrows()]
  ```

### Run the Application:
```bash
python main.py
```
- The script processes the mock dataset, prints tagged results in JSON format, and displays accuracy for zero-shot and few-shot approaches.
- Example output:
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

### Usage:
- Run `main.py` to tag the mock dataset and compare zero-shot vs. few-shot performance.
- To tag a single ticket, call the `process_ticket` function:
  ```python
  from main import process_ticket
  result = process_ticket("My app won’t load.", "few_shot")
  print(json.dumps(result, indent=2))
  ```
- For real-world use, replace the mock LLM with a real LLM API (see [Dependencies](#dependencies)).

## Dependencies
- Python 3.8+
- No external packages are required for the mock LLM implementation.
- For real LLM integration (e.g., xAI’s Grok API), install:
  ```bash
  pip install requests==2.32.3
  ```

Install via:
```bash
pip install requests==2.32.3
```
*Note*: The mock LLM uses only Python’s standard library (`json`, `re`, `typing`). If using a real LLM, additional dependencies may be required based on the API (e.g., `requests` for HTTP calls).

## Future Improvements
- **Real LLM Integration**: Replace the mock LLM with a real LLM (e.g., xAI’s Grok 3 via https://x.ai/api) for improved tagging accuracy and semantic understanding.
- **Larger Dataset**: Support larger, real-world datasets (e.g., CSV or JSON files) using pandas or other data loaders.
- **Enhanced Evaluation**: Implement precision, recall, and F1-score metrics for all tags, not just the primary tag.
- **Custom Tags**: Allow dynamic tag sets via configuration files or user input.
- **Performance Optimization**: Cache prompt results for repeated tickets to reduce processing time.
- **User Interface**: Add a Streamlit or Flask web interface for interactive ticket tagging.
- **Logging**: Save tagging results and logs to a file or database for analysis.

## Contact
For questions, feedback, or collaboration, please reach out:

- **Email**: [Your Email](mailto:muhammadusman.becsef22iba-suk.edu.pk)
- **LinkedIn**: [Your LinkedIn Profile](https://www.linkedin.com/in/muhammad-usman-018535253)
- **GitHub**: [Your GitHub Profile](https://www.github.com/Usmansarwar143)

*Note*: Replace the placeholders above with your actual contact details. If you don’t have a GitHub repository yet, create one at https://github.com and update the clone URL in [Installation and Usage](#installation-and-usage).

Contributions and suggestions are welcome! Please create an issue or submit a pull request.
