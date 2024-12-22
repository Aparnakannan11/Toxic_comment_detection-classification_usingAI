

 # Toxic Comment Detection & Classification using AI

Welcome to the **Toxic Comment Detection & Classification** project! 🌟

In today's digital world, online platforms are full of user-generated content. While many interactions are positive, toxic comments can lead to harassment, bullying, and a negative user experience. This project leverages **Artificial Intelligence (AI)** to detect and classify these toxic comments in real-time, promoting a safer and more respectful online environment.

---

## 🚀 Overview

This project integrates advanced **Natural Language Processing (NLP)** techniques and **AI models** to classify comments as **toxic** or **non-toxic**, and categorize them based on toxicity type (e.g., **insults, profanity, hate speech, etc.**).

With a user-friendly **Streamlit** interface, users can easily input comments and receive instant feedback on the toxicity and its type. It uses the **Meta-Llama-3.1-8B-Instruct-Turbo** model for precise analysis.

---

## 💡 Key Features

- **Real-Time Toxicity Detection**: Instant classification of comments as toxic or non-toxic.
- **Toxicity Categories**: Toxic comments are categorized into specific types (e.g., insults, profanity, hate speech, etc.).
- **User Interface**: Interactive web app built with **Streamlit** for easy use.
- **Powerful AI Model**: Uses **Meta-Llama** API to detect and classify comments accurately.

---

## 🛠️ Installation

To run this project locally, follow these steps:

### Prerequisites

1. **Python 3.x** installed on your machine.
2. Install the required dependencies:
   ```bash
   pip install requests streamlit langchain langchain-community
git clone https://github.com/your-username/toxic-comment-detection.git
cd toxic-comment-detection

---

## 🔸Usage 

1. Run the streamlit app
   ```bash
   streamlit run app.py
2. Open your browser and go to http://localhost:8501.
3. Enter a comment into the text box and click Check Toxicity to get the analysis result.
   
---

## Example 
1. Input Comment 
   ```bash
   You're so stupid, I can't stand you!
   
2. Output Result
   ```bash
   Toxic: Insults
   Reason: The comment contains derogatory terms like "stupid" and "can't stand you".
---

## 🤖 How It Works
1. Text Splitting: The user’s input comment is split into smaller chunks using RecursiveCharacterTextSplitter.
2. Embedding: The text is embedded using OllamaEmbeddings.
3. FAISS: The embedded text is stored in a FAISS vector store to facilitate efficient similarity search.
4. Toxicity Classification: The input is sent to the Meta-Llama-3.1-8B-Instruct-Turbo model via an API call for toxicity detection and classification.

---

## Contributing 

We welcome contributions to improve this project! Feel free to:

1. Fork the repository.
2. Create a new branch.
3. Submit a pull request.

---

## 🌟 Acknowledgements

Meta-Llama for providing the powerful model used for toxicity classification.

Streamlit is used to enable the creation of a smooth and interactive user interface.

FAISS for efficient vector search capabilities.

---

## 🚨 Disclaimer

This system aims to provide high-level toxicity detection and classification. However, it may not catch every nuance of online conversations, and its results should be reviewed accordingly.


---

Thank you for exploring the Toxic Comment Detection & Classification project! 🎉


  


