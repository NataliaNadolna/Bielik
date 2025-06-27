import google.generativeai as genai
import pandas as pd
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from models import GeminiModel, OpenAIModel
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

def get_model(api_type, llm_name, key, url=None):
    if api_type == 'vllm':
        return GeminiModel(llm_name, key)
    elif api_type == 'openAI':
        return OpenAIModel(llm_name, key, url)
    else:
        raise ValueError(f"Unsupported API type: {api_type}")

def generate_answers(df, model, interval=0):
    responses = []
    correctness = []
    for index, row in df.iterrows():
        print(f"Odpowiadam na pytanie {index+1}")
        prompt = f"""Korzystając z wiedzy, którą masz, odpowiedz na pytanie: {row['Pytanie']}.
Do wyboru masz 4 odpowiedzi:
A: {row['A']},
B: {row['B']},
C: {row['C']},
D: {row['D']}.
W odpowiedzi podaj tylko literę A, B, C lub D odpowiedzi, którą uważasz za poprawną.
"""

        answer = model.generate_answer(prompt)

        responses.append(answer)

        if answer == row['Pozycja']:
            correctness.append(1)
        else:
            correctness.append(0)

        if interval > 0 and (index + 1) % interval == 0:
            print(f"Waiting {interval} seconds to respect rate limits...")
            time.sleep(interval)

    df['Odpowiedź modelu'] = responses
    df['Poprawność'] = correctness
    return df

def save(df, llm_id):
    correct_answers = int(df['Poprawność'].sum())

    output_json = {
        "id modelu": llm_id,
        "liczba_pytań": df.shape[0],
        "podsumowanie": {
            "prawidłowe odpowiedzi": correct_answers
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
    load_dotenv()

    config_defaults = load_config()
    args = parse_arguments(config_defaults)

    print("Configuration and arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    print("\nBenchmarking...")

    model = get_model(args.api, args.llm_name, args.key, args.url)
    df = load_dataset(args.test)
    df = generate_answers(df, model, interval=args.interval)

    save(df, args.llm)

    print("Zrobione")

if __name__ == "__main__":
    main()
