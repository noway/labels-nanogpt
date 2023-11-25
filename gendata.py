from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv("KEY"))

def ask_gpt4_chat(question):
    try:
        response = client.chat.completions.create(model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question}
        ])
        return response.choices[0].message.content
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    question = "Generate Kindergarten level K-12 math problem. No emojis. Only output the question, no assistant text."
    answer = ask_gpt4_chat(question)
    print("GPT-4 says:", answer)
