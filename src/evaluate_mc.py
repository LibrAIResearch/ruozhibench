import pandas as pd
import argparse
import os
import json
import sys
from utils import *

from libra_eval.llmclient import Local_Client,OpenAI_Client,Next_Client


def main():
    parser = argparse.ArgumentParser(description='Get model responses for evaluation')
    parser.add_argument('--model', type=str, default='',
                        help='Name of the model to be evaluated')
    parser.add_argument('--client', type=str, default='next',
                        help='Client to use for evaluation')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Directory to save evaluation results')
    parser.add_argument('--api_config', type=str, default='./libra_eval/config/api_config.json',
                        help='Path to the API config file')
    args = parser.parse_args()

    # Check if output file already exists
    output_file = os.path.join(args.data_dir, f"evaluation_mc", f"{args.model.split('/')[-1]}.jsonl")
    if os.path.exists(output_file):
        print(f"Results file {output_file} already exists. Skipping processing.")
        return pd.read_json(output_file, lines=True)
    
    # Create local client
    api_config = json.load(open(args.api_config))
    if args.client == 'local':
        client = Local_Client(model=args.model)
    elif args.client == 'next':
        client = Next_Client(model=args.model, api_config = api_config)
    elif args.client == 'openai':
        client = OpenAI_Client(model=args.model, api_config = api_config)
    else:
        raise ValueError(f"Invalid client: {args.client}")
    
    # Read the JSONL file
    df = pd.read_json(os.path.join(args.data_dir, f"ruozhibench_mc.jsonl"), lines=True)
    
    # Format prompts using mc_template
    formatted_prompts = []
    for _, row in df.iterrows():
        prompt = mc_eval_user_prompt.format(
            question=row['question_en'],
            answer1=row['Good Answer'],
            answer2=row['Bad Answer']
        )
        formatted_prompts.append(prompt)
    
    # Get prompts and construct message list
    messages_list = client.construct_message_list(formatted_prompts, system_role="You are a helpful assistant.")
    responses = client.multi_call(messages_list, max_tokens=1024, post_check_function=mc_post_check)
    df['good_first_response'] = responses
    df['good_first_choice'] = df['good_first_response'].apply(mc_extract)
    df['good_first_correctness'] = df['good_first_choice'] == "AnswerA"
    
    # Format prompts using mc_template
    formatted_prompts = []
    for _, row in df.iterrows():
        prompt = mc_eval_user_prompt.format(
            question=row['question_en'],
            answer1=row['Bad Answer'],
            answer2=row['Good Answer']
        )
        formatted_prompts.append(prompt)
    
    # Get prompts and construct message list
    messages_list = client.construct_message_list(formatted_prompts, system_role="You are a helpful assistant.")
    responses = client.multi_call(messages_list, max_tokens=1024, post_check_function=mc_post_check)
    df['bad_first_response'] = responses
    df['bad_first_choice'] = df['bad_first_response'].apply(mc_extract)
    df['bad_first_correctness'] = df['bad_first_choice'] == "AnswerB"

    # Drop response_scores column if it exists
    if 'response_scores' in df.columns:
        df = df.drop('response_scores', axis=1)
    
    # Save the results
    df.to_json(output_file, orient='records', force_ascii=False, lines=True)
    df.to_csv(output_file.replace('.jsonl', '.csv'), index=False)
    print(f"Results saved to {output_file}")
    
    return df

if __name__ == "__main__":
    main()