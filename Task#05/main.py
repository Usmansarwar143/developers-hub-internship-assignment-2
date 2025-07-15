import json
from typing import List, Dict
import re

# Mock dataset of support tickets with ground-truth tags for evaluation
MOCK_DATASET = [
    {
        "ticket": "I was charged twice for my subscription this month. Please help!",
        "true_tags": ["Billing", "General Inquiry", "Account Management"]
    },
    {
        "ticket": "The app crashes every time I try to upload a photo.",
        "true_tags": ["Technical Issue", "Feature Request", "General Inquiry"]
    },
    {
        "ticket": "Can you add a dark mode to the app?",
        "true_tags": ["Feature Request", "General Inquiry", "Technical Issue"]
    },
    {
        "ticket": "I forgot my password and the reset link isn’t working.",
        "true_tags": ["Account Management", "Technical Issue", "General Inquiry"]
    },
    {
        "ticket": "My payment failed, and now I can’t access my account.",
        "true_tags": ["Billing", "Account Management", "Technical Issue"]
    }
]

# Possible tags
TAGS = ["Billing", "Technical Issue", "Account Management", "Feature Request", "General Inquiry"]

# Zero-shot prompt template with escaped curly braces
ZERO_SHOT_PROMPT = """
You are a support ticket classification system. Your task is to analyze a customer support ticket and assign the top 3 most relevant tags from the following categories: {tags}. For each tag, provide a confidence score (0-100%) based on how likely it applies to the ticket. Output the result in the following JSON format:

{{
  "ticket": "{ticket}",
  "tags": [
    {{"tag": "<tag1>", "confidence": <score>}},
    {{"tag": "<tag2>", "confidence": <score1>}},
    {{"tag": "<tag3>", "confidence": <score2>}}
  ]
}}

Ticket: "{ticket}"
"""

# Few-shot prompt template with escaped curly braces
FEW_SHOT_PROMPT = """
You are a support ticket classification system. Your task is to analyze a customer support ticket and assign the top 3 most relevant tags from the following categories: {tags}. For each tag, provide a confidence score (0-100%) based on how likely it applies to the ticket. Below are examples to guide your tagging:

**Example 1**:
Ticket: "I was charged twice for my subscription this month. Please help!"
Tags: [
  {{"tag": "Billing", "confidence": 95}},
  {{"tag": "General Inquiry", "confidence": 60}},
  {{"tag": "Account Management", "confidence": 30}}
]

**Example 2**:
Ticket: "The app crashes every time I try to upload a photo."
Tags: [
  {{"tag": "Technical Issue", "confidence": 90}},
  {{"tag": "Feature Request", "confidence": 50}},
  {{"tag": "General Inquiry", "confidence": 20}}
]

**Example 3**:
Ticket: "Can you add a dark mode to the app?"
Tags: [
  {{"tag": "Feature Request", "confidence": 98}},
  {{"tag": "General Inquiry", "confidence": 40}},
  {{"tag": "Technical Issue", "confidence": 10}}
]

**Example 4**:
Ticket: "I forgot my password and the reset link isn’t working."
Tags: [
  {{"tag": "Account Management", "confidence": 90}},
  {{"tag": "Technical Issue", "confidence": 70}},
  {{"tag": "General Inquiry", "confidence": 30}}
]

Now, classify the following ticket and output the result in the same JSON format:

Ticket: "{ticket}"
"""

# Mock LLM: Simulates tag prediction using keyword-based rules
def mock_llm_predict(ticket: str, prompt: str) -> Dict:
    """
    Simulates an LLM by assigning tags based on keyword matching.
    Returns a dictionary with the ticket and top 3 tags with confidence scores.
    """
    keyword_scores = {
        "Billing": ["charge", "payment", "subscription", "billing", "refund"],
        "Technical Issue": ["crash", "error", "bug", "not working", "slow"],
        "Account Management": ["account", "login", "password", "access", "reset"],
        "Feature Request": ["add", "feature", "new", "mode", "improve"],
        "General Inquiry": ["help", "please", "question", "how"]
    }
    
    scores = {tag: 0 for tag in TAGS}
    ticket_lower = ticket.lower()
    
    # Assign scores based on keyword matches
    for tag, keywords in keyword_scores.items():
        for keyword in keywords:
            if re.search(r'\b' + keyword + r'\b', ticket_lower):
                scores[tag] += 30  # Base score for keyword match
                if tag == "General Inquiry":  # Lower weight for generic tag
                    scores[tag] = min(scores[tag], 50)
    
    # Adjust scores for few-shot prompt based on example patterns
    if "Example 1" in prompt:  # Few-shot
        if "charge" in ticket_lower and "subscription" in ticket_lower:
            scores["Billing"] = min(scores["Billing"] + 20, 95)
        if "crash" in ticket_lower:
            scores["Technical Issue"] = min(scores["Technical Issue"] + 20, 90)
        if "dark mode" in ticket_lower or "add" in ticket_lower:
            scores["Feature Request"] = min(scores["Feature Request"] + 30, 98)
        if "password" in ticket_lower and "reset" in ticket_lower:
            scores["Account Management"] = min(scores["Account Management"] + 20, 90)
    
    # Normalize and select top 3 tags
    sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
    total_score = sum(score for _, score in sorted_tags) or 1
    normalized_tags = [
        {"tag": tag, "confidence": min(int(score / total_score * 100), 100)}
        for tag, score in sorted_tags if score > 0
    ]
    
    # Ensure 3 tags by adding low-confidence tags if needed
    while len(normalized_tags) < 3:
        remaining_tags = [tag for tag in TAGS if tag not in [t["tag"] for t in normalized_tags]]
        if remaining_tags:
            normalized_tags.append({"tag": remaining_tags[0], "confidence": 10})
    
    return {
        "ticket": ticket,
        "tags": normalized_tags
    }

# Function to process a ticket using a prompt
def process_ticket(ticket: str, prompt_type: str = "zero_shot") -> Dict:
    """
    Processes a ticket using zero-shot or few-shot prompting.
    Returns a dictionary with the ticket and predicted tags.
    """
    try:
        prompt = ZERO_SHOT_PROMPT if prompt_type == "zero_shot" else FEW_SHOT_PROMPT
        filled_prompt = prompt.format(tags=", ".join(TAGS), ticket=ticket)
        
        # Replace with real LLM API call (e.g., xAI Grok API) if available
        response = mock_llm_predict(ticket, filled_prompt)
        
        return response
    except Exception as e:
        print(f"Error processing ticket '{ticket}': {e}")
        return {"ticket": ticket, "tags": [], "error": str(e)}

# Evaluate performance on mock dataset
def evaluate_performance(dataset: List[Dict], prompt_type: str) -> float:
    """
    Evaluates the accuracy of the tagger by checking if the primary predicted tag
    is in the true tags. Returns accuracy as a percentage.
    """
    correct = 0
    total = len(dataset)
    
    for item in dataset:
        ticket = item["ticket"]
        true_tags = item["true_tags"]
        prediction = process_ticket(ticket, prompt_type)
        predicted_tags = [tag["tag"] for tag in prediction["tags"]]
        
        # Check if primary tag (first predicted) is in true tags
        if predicted_tags and predicted_tags[0] in true_tags:
            correct += 1
    
    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy

# Main function to run the tagger and compare zero-shot vs few-shot
def main():
    """
    Runs the ticket tagger using zero-shot and few-shot prompts, prints results,
    and compares accuracy.
    """
    print("Running Zero-Shot Tagging...")
    zero_shot_results = [process_ticket(item["ticket"], "zero_shot") for item in MOCK_DATASET]
    zero_shot_accuracy = evaluate_performance(MOCK_DATASET, "zero_shot")
    
    print("\nRunning Few-Shot Tagging...")
    few_shot_results = [process_ticket(item["ticket"], "few_shot") for item in MOCK_DATASET]
    few_shot_accuracy = evaluate_performance(MOCK_DATASET, "few_shot")
    
    # Print results
    print("\nZero-Shot Results:")
    for result in zero_shot_results:
        print(json.dumps(result, indent=2))
    
    print("\nFew-Shot Results:")
    for result in few_shot_results:
        print(json.dumps(result, indent=2))
    
    print(f"\nZero-Shot Accuracy: {zero_shot_accuracy:.2f}%")
    print(f"Few-Shot Accuracy: {few_shot_accuracy:.2f}%")

if __name__ == "__main__":
    main()