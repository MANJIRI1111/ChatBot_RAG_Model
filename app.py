from flask import Flask, render_template, request, jsonify
# from openai import OpenAI
import openai
from pinecone import Pinecone
import numpy as np
import os

os.environ['OPENAI_API_KEY']='open_api_key'
os.environ['PINECONE_API_KEY']='pinecone_api_key'
os.environ['PINECONE_ENV']='us-east-1'

# Initialize Flask app
app = Flask(__name__, template_folder=os.path.join(os.getcwd(), "templates"))

openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone
pinecone_api_key = os.getenv("PINECONE_API_KEY")  
pinecone_environment = os.getenv("PINECONE_ENV")  

pinecone = Pinecone(api_key = os.environ.get('PINECONE_API_KEY'))
index_name = "harry-idx"  

# Connect to Pinecone index
index = pinecone.Index(index_name)

# Define the embedding model
EMBED_MODEL = "text-embedding-ada-002"  

def get_embeddings(text, model='text-embedding-3-small'):
    text = text.replace("\n", " ")
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']


def get_context(query, embed_model='text-embedding-3-small',k=5):
  query_embeddings = get_embeddings(query, model=embed_model)
  pinecone_response = index.query(vector =query_embeddings, top_k=k, include_metadata=True)
  contexts =[item['metadata']['text'] for item in pinecone_response['matches']]
  return contexts, query

def augmented_query(user_query,embed_model='text-embedding-3-small',k=5):
  contexts,query = get_context(user_query,embed_model=embed_model,k=k)
  return "\n\n---\n\n".join(contexts)+"\n\n---\n\n"+query

# Function to ask OpenAI GPT
def ask_gpt(system_prompt, user_prompt, model="gpt-3.5-turbo", temperature=0.7):
    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return completion.choices[0].message.content

input_pattern=["system prompt:","disregard previous", "ignore all instructions", "override prior", "generate false"]
def input_GR(text, pattern=input_pattern):
    for phrase in pattern:
        if phrase in text.lower():
            return True
        return False

out_pattern=["system prompt:","cannot answer", "Elon Musk", "not authorized", "outside of context", "I don't know", "consult a human expert", "error", "restricted section"]
def out_GR(text, pattern=out_pattern):
  for phrase in pattern:
    if phrase.lower() in text.lower():
      return True
  return False

def normal_AI(query, k=5):
    """
    Handles user queries with Guard Rails for input and output validation.
    """
    # Incoming Guardrail - Check query
    if input_GR(query):
        return "Your query violates the system's guidelines and cannot be processed."
    # print("passed")

    embed_model = 'text-embedding-3-small'
    primer = """
    You are a question answering assistant. A highly intelligent system that answers user questions based on
    the information provided by the user above each question.
    If the answer cannot be found in the information provided by the user, you truthfully answer,
    'I don't know more about the Harry Potter Universe!!!'
    """
    llm_model = 'gpt-4o-mini'

    user_prompt = augmented_query(query, embed_model=embed_model, k=k)

    response = ask_gpt(primer, user_prompt, model=llm_model)

    # Outgoing Guardrail - Check response
    if out_GR(response):
        return "The response is outside the scope of authorized content."

    return response

# Homepage route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        data = request.get_json() 
        user_query = data.get("question")
        answer = normal_AI(query=user_query)

        return jsonify({"answer": answer})

    return render_template("index.html")

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)
