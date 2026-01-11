from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client()
history = []

while True:
    prompt = input("Ja: ")
    history.append({"role": "user", "parts": [{"text": prompt}]})
    response = client.models.generate_content(
        model="gemini-2.5-flash", contents=history
    )
    history.append({"role": "model", "parts": [{"text": response.text}]})
    print("G:", response.text)