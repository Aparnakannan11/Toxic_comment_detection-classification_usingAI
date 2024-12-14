import requests
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import time
from langchain.schema import Document

# Your API key
API_KEY = 'xxxxxxxxxxxxxxx'         # Enter Your API key

# Sample data (you can add your own comments)
sample_data = [
    "You're so stupid, I can't stand you!",
    "You did an amazing job on the project!",
    "I'm going to make sure you regret this.",
    "Don't worry, you can do it! I believe in you.",
    "Stop being such a loser, no one cares about you.",
    "I think your presentation could be improved by adding more data visualizations."
]

# Convert the sample data into Document objects
documents = [Document(page_content=comment) for comment in sample_data]

# Split the sample data into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(documents)

# Initialize Ollama embeddings model
embedding_model = OllamaEmbeddings()

# Embed the documents using the correct method
embeddings = embedding_model.embed_documents([doc.page_content for doc in documents])

# Create FAISS vector store
faiss_db = FAISS.from_texts([doc.page_content for doc in documents], embedding_model)

# Define a function to call the Meta-Llama API
def llama3_api_call(prompt):
    url = "https://api.together.xyz/v1/chat/completions"  # Correct API endpoint
    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json'
    }
    data = {
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # Correct model name
        "messages": [
            {
                "role": "system", 
                "content": """
                    You are a toxic comment detector and classifier. Your task is to assess whether the given comment is toxic or non-toxic and classify it into specific categories. 
                    If the comment is toxic, respond with the toxicity type and the reason, otherwise, respond with 'non-toxic' and the category of the comment. 
                    
                    Toxic categories:
                    1. Insults: Examples - Stupid, Idiot, Dumb, Moron, Fool, Worthless, Useless, Trash, Pathetic, Loser
                    2. Profanity: Examples - Damn, Hell, Crap, Bloody, Shit, Fuck, Asshole, Bastard, Bitch, Piss
                    3. Hate Speech: Examples - Racist, Bigot, Sexist, Homophobic, Xenophobic, Misogynist, Anti-Semitic, Discrimination, Intolerance, Prejudice
                    4. Threats: Examples - Kill, Attack, Harm, Hurt, Destroy, Die, Murder, Stab, Bomb, Explode
                    5. Harassment: Examples - Bully, Troll, Mock, Shame, Humiliate, Ridicule, Tease, Criticize, Abuse, Demean
                    6. Sexual Harassment: Examples - Creepy, Pervert, Obscene, Molest, Assault, Rape, Stalker, Lust, Voyeur, Predatory
                    7. Aggression: Examples - Angry, Rage, Violent, Fight, Smash, Punch, Shout, Scream, Argue, Break
                    8. Sarcasm/Passive Aggression: Examples - Sure, Right, Wow, Amazing, Great, Fine, Whatever, Really, Nice, Perfect
                    9. Derogatory Slurs: Examples - N-word, F-word, C-word, Slut, Whore, Retard, Fat, Ugly, Gay, Poor
                    10. Cyberbullying: Examples - Loser, Worthless, Dumb, Block, Ignore, Hate, Cancel, Expose, Shame, Report
                    
                    Non-toxic categories:
                    1. Compliments: Examples - Amazing, Beautiful, Brilliant, Excellent, Fantastic, Great, Lovely, Outstanding, Perfect, Wonderful
                    2. Encouragement: Examples - Keep going, Well done, You can do it, Don’t give up, Great job, Proud, Believe, Impressive, Fantastic work, Nice effort
                    3. Politeness: Examples - Please, Thank you, Sorry, Excuse me, Appreciate, Kind, Welcome, Respect, Grateful, Courteous
                    4. Empathy: Examples - Understand, Feel, Care, Support, Listen, Kindness, Compassion, Sympathy, Considerate, Warm
                    5. Friendliness: Examples - Hello, Hi, Nice to meet, Good day, Take care, Welcome, Greetings, Good evening, How are you, Pleasure
                    6. Constructive Feedback: Examples - Suggest, Recommend, Improve, Idea, Enhance, Adjust, Consider, Thought, Option, Suggestion
                    7. Neutral Questions: Examples - Why, How, What, When, Where, Can you, Could you, Would you, Is this, Does this
                    8. Positive Affirmations: Examples - Yes, Agree, Correct, True, Absolutely, Sure, Right, Certainly, Of course, Definitely
                    9. Motivational Phrases: Examples - You’ve got this, Don’t worry, It’s okay, Everything will be fine, Believe in yourself, Stay positive, Keep pushing, Keep smiling, Never give up, Bright future
                    10. Expressions of Gratitude: Examples - Thanks, Appreciate, Grateful, Thank you so much, Much obliged, Heartfelt thanks, Many thanks, Cheers, Blessed, Thankful
                """
            },
            {"role": "user", "content": prompt}  # User's input prompt
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        result = response.json()
        return result.get('choices', [{}])[0].get('message', {}).get('content', 'No response from model')
    else:
        return f"Error: {response.status_code}"

# Streamlit web interface
def main():
    st.set_page_config(page_title="Toxic Comment Detection Bot", page_icon=":question:", layout="centered")

    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Toxic Comment Detection</title>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&family=Open+Sans:wght@300;400;600&display=swap" rel="stylesheet">
        <style>
            body {
                font-family: 'Open Sans', sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #ff80ff, #7e3fff);
                color: #fff;
                height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                flex-direction: column;
            }

            .container {
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                padding: 40px;
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
                max-width: 500px;
                width: 100%;
            }

            h1 {
                font-size: 36px;
                margin-bottom: 20px;
                color: #fff;
            }

            h2 {
                font-size: 18px;
                color: #ddd;
                margin-bottom: 30px;
            }

            .text-area {
                width: 80%;
                height: 150px;
                padding: 15px;
                border-radius: 8px;
                border: 2px solid #7e3fff;
                background-color: #1a1a1a;
                color: #fff;
                font-size: 16px;
                resize: none;
                margin-bottom: 20px;
                transition: all 0.3s;
            }

            .text-area:focus {
                border-color: #ff80ff;
                outline: none;
            }

            .btn {
                padding: 15px 30px;
                background: #7e3fff;
                color: #fff;
                font-size: 18px;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                width: 80%;
                transition: background 0.3s;
            }

            .btn:hover {
                background: #ff80ff;
            }

            .result {
                margin-top: 30px;
                font-size: 20px;
                padding: 20px;
                border-radius: 10px;
                background: rgba(255, 255, 255, 0.2);
            }

            .toxicity {
                font-weight: 700;
                color: #ff4040;
            }

            .non-toxic {
                font-weight: 700;
                color: #4caf50;
            }

            .footer {
                font-size: 12px;
                color: #ddd;
                position: absolute;
                bottom: 20px;
            }

            @media (max-width: 768px) {
                h1 {
                    font-size: 28px;
                }

                .container {
                    padding: 30px;
                }

                .text-area {
                    width: 90%;
                    font-size: 14px;
                }

                .btn {
                    width: 90%;
                }
            }
        </style>
    </head>
    <body>

    <div class="container">
        <h1>Toxic Comment Detection Bot</h1>
        <h2>Enter a comment to check if it’s toxic or non-toxic</h2>

        <textarea class="text-area" id="userComment"></textarea>

        <button class="btn" onclick="checkToxicity()">Check Toxicity</button>

        <div id="result" class="result"></div>

        <script>
            function checkToxicity() {
                let userComment = document.getElementById("userComment").value;
                if(userComment) {
                    fetch(`http://localhost:8501/toxic_check?comment=${userComment}`)
                        .then(response => response.json())
                        .then(data => {
                            document.getElementById("result").innerHTML = data.result;
                        });
                }
            }
        </script>
    </div>

    </body>
    </html>
    """, unsafe_allow_html==True

    with st.spinner('Loading...'):
        time.sleep(2)  # Simulating a delay for the animation

    st.success("You got your answer!")

    # Text input for user comment
    user_input = st.text_area("Enter your comment here:")

    # Button to trigger the toxic comment check
    if st.button('Check Toxicity'):
        if user_input:
            # Display the comment entered by the user
            st.write(f"User input: {user_input}")

            # Call the Meta-Llama API to check for toxicity
            response = llama3_api_call(user_input)
            
            # Display the result
            st.markdown(f"### {response}")
        else:
            st.warning("Please enter a comment to check.")

if __name__ == "__main__":
    main()
