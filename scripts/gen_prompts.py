"""
Generate the prompts for individual examples.
"""
import argparse
import pandas as pd
import json
import getpass
import openai
from tqdm import tqdm

def readArgs():
    parser = argparse.ArgumentParser(description='Generate the prompts for the GPT models')
    parser.add_argument ('--template-file', type=str, required=True, help="File contains the template for the prompt")
    parser.add_argument ('--input-file', type=str, required=True, help="Path to the input TSV file")
    parser.add_argument ('--field-name', type=str, required=False, default='context_100', help="Text field")
    parser.add_argument ('--temperature', type=float, required=False, default=0.2, help="Temperature parameter")
    parser.add_argument ('--model-name', type=str, required=False, default='gpt-3.5-turbo', choices={'gpt-4', 'gpt-3.5-turbo'}, help="Model name")
    parser.add_argument ('--output-file', type=str, required=True, help="Path to the output TSV file that contains the output")
    args = parser.parse_args()
    return args

def readTemplateFile (filename):
    with open (filename) as fin:
        text = fin.read()
    
    return text

def generate_prompt (text, template):
    return template.replace ('{{TEXT}}', text)

def make_openai_call (prompt, model_name, temperature=0.2):
    response = openai.ChatCompletion.create(
            model=model_name,
            messages = [
                {"role": "user", "content": prompt}
            ],
            temperature=temperature
        )
    return response

def main (args):
    APIKEY = getpass.getpass(prompt='Enter OpenAI API key:')
    openai.api_key = APIKEY
    tqdm.pandas(desc='Progressbar')
    template = readTemplateFile (args.template_file)
    df = pd.read_csv (args.input_file, sep='\t')

    df['prompt'] = df.progress_apply (lambda x: generate_prompt (x[args.field_name], template), axis=1)
    df['response'] = df.progress_apply (lambda x: make_openai_call (x['prompt'], args.model_name, args.temperature), axis=1)

    df.to_csv (args.output_file, sep='\t', index=False, header=True)

if __name__ == '__main__':
    main (readArgs())