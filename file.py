import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=api_key)

# model = genai.GenerativeModel("gemini-2.5-flash") # OBSOLETE, not accessible for new users (keys)
model = genai.GenerativeModel("gemini-flash-latest")
# model = genai.GenerativeModel("gemini-3.5-flash")

# response = model.generate_content("How does AI work?")
# response = model.generate_content("What does the weather like in Berlin now?")
response = model.generate_content("What do you think about urgent 'business' trip to Spain and mediterranean sea? :D")

print(response.text)

