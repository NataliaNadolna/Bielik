import google.generativeai as genai
from openai import OpenAI

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

# OpenAI jeszcze nie działa, pracuję nad tym
class OpenAIModel():
    def __init__(self, model_name, api_key, api_url):
        self.client = OpenAI(api_key=api_key, api_base=api_url)
        self.model_name = model_name

    def generate_answer(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=1,
            temperature=0.0,
        )
        text = completion.choices[0].message.content.strip()
        return text