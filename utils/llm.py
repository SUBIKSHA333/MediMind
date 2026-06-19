from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to Groq LLM
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def ask_llm(prompt, system_message="You are a helpful medical AI assistant."):
    """Send a question to the LLM and get a response"""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content

# Test it
if __name__ == "__main__":
    answer = ask_llm("What are the symptoms of diabetes?")
    print("🤖 AI Response:")
    print(answer)