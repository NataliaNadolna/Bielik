import google.generativeai as genai
import pandas as pd
import argparse
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import time

CONFIG_PATH = 'config.json'

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    return {}

def parse_arguments(config_defaults):
    parser = argparse.ArgumentParser(description='Benchmark test for LLMs')

    parser.add_argument('--test', type=str, default=config_defaults.get('test', './test.csv'),
    help='Path to the test file (CSV or URL)')

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
    help='API key (required if API is openAI)')

    parser.add_argument('--interval', type=int, default=config_defaults.get('interval'),
    help='Interval between API calls')

    args = parser.parse_args()

    if args.api == 'openAI' and (not args.url or not args.key):
        parser.error("--url and --key are required when --api is 'openAI'")

    if not args.results:
        args.results = f'./results/{args.llm}.json'

    return args

def load_dataset(dataset_path):
    df = pd.read_excel(dataset_path, sheet_name='test')
    print("Wczytane dane:")
    print(df)
    return df

def generate_answers(df, model):

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

        # # czekanie, bo limity gemini 
        # if key == "GOOGLE_API_KEY" and (index + 1) % 15 == 0:
        #     print("Waiting")
        #     time.sleep(60)

        #response = model.generate_content(prompt)
        response = model.generate_content(
            contents=prompt,
            generation_config={
                "max_output_tokens": 1
            }
        )
        answer = response.text
        responses.append(answer)

        if answer == row['Pozycja']:
            correctness.append(1)
        else:
            correctness.append(0)

    # zapis do .csv
    df['Odpowiedź modelu'] = responses
    df['Poprawność'] = correctness
    #df.to_csv(output_path, index=False)
    #df.to_excel(output_path, sheet_name='Sheet1')

    return df


def save(df, llm_id):

    # obliczenie liczby poprawnych odpowiedzi
    correct_answers = df['Poprawność'].sum()
    correct_answers = int(correct_answers)

    # zapisz odpowiedzi do plików

    # Podsumowanie - json z liczbą pytań, poprawnych, błędnych
    answers_list = df.to_dict(orient="records")
    output_json = {
        "id modelu": llm_id,
        "liczba_pytań": df.shape[0],
        "podsumowanie": {
            "prawidłowe odpowiedzi": correct_answers
        }
    }

    # dodatkowo lista jsonl z pytaniami
    folder = "results"
    filename = f"{llm_id}.jsonl"
    filepath = os.path.join(folder, filename)
    os.makedirs(folder, exist_ok=True)
    df.to_json(filepath, orient="records", lines=True, force_ascii=False)

    with open("output.json", "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)

    print("Zapisano")



def main():
    # Wczytaj zmienne środowiskowe z pliku .env
    load_dotenv()

    config_defaults = load_config()
    args = parse_arguments(config_defaults)

    print("Configuration and arguments:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

    # Placeholder for benchmark logic
    print("\nBenchmarking...")


    # Skonfiguruj model Gemini
    genai.configure(api_key=args.key)
    model = genai.GenerativeModel(args.llm_name)

    # Wczytaj dane testowe
    df = load_dataset(args.test)

    # Generuj odpowiedzi
    #df = generate_answers(df, model, interval=args.interval)
    df = generate_answers(df, model)

    # Zapisz wyniki
    save(df, args.llm)

    print("done")



if __name__ == "__main__":
    main()
