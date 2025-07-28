import pandas as pd
import argparse
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI
import os
import json
import time

CONFIG_PATH = 'config.json'

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def parse_arguments(config_defaults):
    parser = argparse.ArgumentParser(description='Benchmark test for LLMs')

    parser.add_argument('--test', type=str, default=config_defaults.get('test', './test.csv'),
                        help='Path to the test file (CSV, Excel or URL)')

    parser.add_argument('--results', type=str, default=config_defaults.get('results'),
                        help='Path to save results')

    parser.add_argument('--llm', type=str, default=config_defaults.get('llm'),
                        help='Unique identifier for the LLM')

    parser.add_argument('--llm-name', type=str, default=config_defaults.get('llm-name'),
                        help='Product name of the LLM')

    parser.add_argument('--api', choices=['vllm', 'openAI'], default=config_defaults.get('api', 'vllm'),
                        help='API type to use')

    parser.add_argument('--url', type=str, default=config_defaults.get('url'),
                        help='URL endpoint (required if API is openAI)')

    parser.add_argument('--key', type=str, default=config_defaults.get('key'),
                        help='API key (required if API is openAI or Gemini)')

    parser.add_argument('--interval', type=int, default=config_defaults.get('interval', 0),
                        help='Interval between API calls in seconds')

    args = parser.parse_args()

    if args.api == 'openAI' and (not args.url or not args.key):
        parser.error("--url and --key are required when --api is 'openAI'")

    if not args.results:
        args.results = f'./results/{args.llm}.json'

    return args

def load_dataset(dataset_path):
    if dataset_path.endswith('.csv'):
        df = pd.read_csv(dataset_path)
    elif dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
        df = pd.read_excel(dataset_path, sheet_name='test')
    else:
        raise ValueError("Unsupported dataset format, use CSV or Excel")
    print("Wczytane dane:")
    print(df.head())
    return df

# def get_model(api_type, llm_name, key, url=None):
#     if api_type == 'vllm':
#         return GeminiModel(llm_name, key)
#     elif api_type == 'openAI':
#         return OpenAIModel(llm_name, key, url)
#     else:
#         raise ValueError(f"Unsupported API type: {api_type}")

def load_model_and_tokenizer(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer


def get_answer(df, model, tokenizer):
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    responses = []
    correctness = []
    for index, row in df.iterrows():
        print(f"Odpowiadam na pytanie {index+1}")
        prompt = f"""Odpowiedz na pytanie: {row['Pytanie']}.
Do wyboru masz 4 odpowiedzi:
A: {row['A']},
B: {row['B']},
C: {row['C']},
D: {row['D']}.
Podaj tylko literę A, B, C lub D oznaczającą odpowiedź, którą uważasz za poprawną.
"""

        response = pipe(prompt, max_new_tokens=1)
        answer = response[0]['generated_text']
        clean_answer = answer[-1]

        print(f"Odpowiedź to: {clean_answer}")

        responses.append(clean_answer)
        if clean_answer == row['Pozycja']:
            correctness.append(1)
        else:
            correctness.append(0)

    df['Odpowiedz modelu'] = responses
    df['Poprawnosc'] = correctness
    return df


def save(df, llm_id):
    correct_answers = int(df['Poprawnosc'].sum())

    output_json = {
        "id modelu": llm_id,
        "liczba_pytan": df.shape[0],
        "podsumowanie": {
            "prawidlowe odpowiedzi": correct_answers
        }
    }

    folder = "results"
    os.makedirs(folder, exist_ok=True)
    jsonl_path = os.path.join(folder, f"{llm_id}.jsonl")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print("Zapisano")


def main():
    #load_dotenv()

    config_defaults = load_config()
    args = parse_arguments(config_defaults)

    print("Configuration and arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    df = load_dataset(args.test)
    model_id = args.llm_name
    model, tokenizer = load_model_and_tokenizer(model_id)
    df_with_answers = get_answer(df, model, tokenizer)

    save(df_with_answers, args.llm)

    print("Zrobione")

if __name__ == "__main__":
    main()
