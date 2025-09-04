import pandas as pd
import argparse
import torch
from dotenv import load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from openai import OpenAI
from models import GeminiModel, OpenAIModel
from google.api_core.exceptions import ResourceExhausted
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

    parser.add_argument('--api', choices=['vllm', 'openAI', 'gemini'], default=config_defaults.get('api', 'vllm'),
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

def load_model_and_tokenizer(model_name):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    return model, tokenizer

def get_answer_local(df, model, tokenizer):
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    responses, correctness = [], []

    for index, row in df.iterrows():
        print(f"Odpowiadam na pytanie {index+1}")

        prompt = f"""
Pytanie: {row['Pytanie']}

A: {row['A']}
B: {row['B']}
C: {row['C']}
D: {row['D']}

Podaj tylko jedną literę (A, B, C lub D), która jest poprawną odpowiedzią.

Odpowiedź:
"""

        response = pipe(
            prompt,
            max_new_tokens=2,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            return_full_text=False,
            eos_token_id=tokenizer.eos_token_id
        )

        raw_answer = response[0]['generated_text'].strip().upper()

        clean_answer = "?"
        for opt in ["A", "B", "C", "D"]:
            if raw_answer.startswith(opt):
                clean_answer = opt
                break

        print(f"Odpowiedź to: {clean_answer}")

        responses.append(clean_answer)
        correctness.append(int(clean_answer == str(row['Pozycja']).strip().upper()))

    df["Odpowiedz modelu"] = responses
    df["Poprawnosc"] = correctness
    return df

def get_answer_api(df, model, interval=0):
    
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
        
        try:
            ans = model.generate_answer(prompt)
        except ResourceExhausted as e:
            print("Limit API osiągnięty, czekam 30 sekund...")
            time.sleep(interval)
            ans = model.generate_answer(prompt)

        answer = ans.upper()
        responses.append(answer)

        print(f"Odpowiedź to: {answer}")

        if answer == row['Pozycja']:
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
    load_dotenv()

    config_defaults = load_config()
    args = parse_arguments(config_defaults)

    print("Configuration and arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    df = load_dataset(args.test)

    model_id = args.llm_name

    if args.api == 'vllm':
        model, tokenizer = load_model_and_tokenizer(model_id)
        df_with_answers = get_answer_local(df, model, tokenizer)

    elif args.api == 'gemini':
        model = GeminiModel(model_id, args.key)
        df_with_answers = get_answer_api(df, model, args.interval)

    elif args.api == 'openAI':
        model = OpenAIModel(model_id, args.key, args.url)
        df_with_answers = get_answer_api(df, model, args.interval)

    else:
        raise ValueError(f"Nieobsługiwany typ API: {args.api}")


    save(df_with_answers, args.llm)

    print("Zrobione")

if __name__ == "__main__":
    main()
