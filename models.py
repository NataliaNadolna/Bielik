import google.generativeai as genai
from openai import OpenAI
from openai import RateLimitError
import time

class GeminiModel():
    def __init__(self, model_name, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def generate_answer(self, prompt):
        response = self.model.generate_content(
            contents=prompt,
            generation_config={
                "max_output_tokens": 1
            }
        )
        return response.text.strip()

class OpenAIModel():
    def __init__(self, model_name, api_key, api_url=None):
        self.client = OpenAI(api_key=api_key)
        if api_url:
            self.client.base_url = api_url
        self.model_name = model_name

    def generate_answer(self, prompt):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1,
                    temperature=0.0
                )
                return completion.choices[0].message["content"].strip()
            except RateLimitError:
                print("Przekroczono limit API OpenAI, czekam 60s")
                time.sleep(60)
                return self.generate_answer(prompt)