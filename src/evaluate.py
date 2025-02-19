import pandas as pd
import argparse
import os
import json
import sys
from utils import *

from libra_eval.llmclient import Local_Client,OpenAI_Client,Next_Client


def main():
    parser = argparse.ArgumentParser(description='Get model responses for evaluation')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=["gen", "normal"],
                        help='Mode to use for evaluation')
    parser.add_argument('--evaluator', type=str, required=True, 
                        choices=["gpt-4o-2024-08-06", "claude-3-5-sonnet-20241022", "meta-llama/Llama-3.1-70B-Instruct"],
                        help='Name of the model to use as evaluator')
    parser.add_argument('--client', type=str, default='next',
                        help='Client to use for evaluation')
    parser.add_argument('--data_dir', type=str, default='../data/',
                        help='Directory to save evaluation results')
    parser.add_argument('--api_config', type=str, default='./libra_eval/config/api_config.json',
                        help='Path to the API config file')
    args = parser.parse_args()

    response_dir = os.path.join(args.data_dir, f"response_{args.mode}")
    result_dir = os.path.join(args.data_dir, f"evaluation_{args.mode}", args.evaluator.split('/')[-1])

    os.makedirs(result_dir, exist_ok=True)

    # Loop through response files
    for response_file in os.listdir(response_dir)[::-1]:
        if not response_file.endswith('.jsonl'):
            continue
            
        result_file = os.path.join(result_dir, response_file)
        
        # Skip if evaluation file already exists
        if os.path.exists(result_file):
            print(f"Evaluation file {result_file} already exists. Skipping.")
            continue
            
        print(f"Processing responses for {response_file}...")
        
        # Create client for evaluation
        api_config = json.load(open(args.api_config))
        if args.client == 'next':
            client = Next_Client(model=args.evaluator, api_config = api_config)
        elif args.client == 'openai':
            client = OpenAI_Client(model=args.evaluator, api_config = api_config)
        elif args.client == 'local':
            client = Local_Client(model=args.evaluator)
        else:
            raise ValueError(f"Invalid client: {args.client}")
        
        # Read response data using pandas
        response_path = os.path.join(response_dir, response_file)
        data = pd.read_json(response_path, lines=True).to_dict('records')
        
        # Prepare evaluation messages
        eval_messages = []
        response_format = {
            "type": "json_object"
        }
        if args.mode == "gen":
            for entry in data:
                prompt = gen_eval_user_prompt.format(
                    question=entry['question_en'],
                    irrationality_analysis=entry['irrationality'],
                    answer=entry['response']
                )
                eval_messages.append([
                    {"role": "system", "content": gen_eval_system_prompt},
                    {"role": "user", "content": prompt}
                ])
        elif args.mode == "normal":
            for entry in data:
                prompt = normal_eval_user_prompt.format(
                    question=entry['pair'],
                    answer=entry['pair_response']
                )
                eval_messages.append([
                    {"role": "user", "content": prompt}
                ])

        # Get evaluation results
        eval_results = client.multi_call(eval_messages, show_progress=True, max_tokens=2048, response_format=response_format, post_check_function=rate_post_check)
        
        # Process evaluation results
        for i, (entry, eval_result) in enumerate(zip(data, eval_results)):
            rating = rate_extract(eval_result)
            entry[f'{args.mode}_evaluation'] = eval_result
            entry[f'{args.mode}_rating'] = rating
        
        # Save all results at once using pandas
        pd.DataFrame(data).to_json(result_file, orient='records', lines=True, force_ascii=False)
        pd.DataFrame(data).to_csv(result_file.replace('.jsonl', '.csv'), index=False)
        print(f"Evaluation results saved to {result_file}")
    

if __name__ == "__main__":
    main()