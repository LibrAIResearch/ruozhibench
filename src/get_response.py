import json
import pandas as pd
import argparse
from libra_eval.llmclient import Local_Client, Next_Client, OpenAI_Client
import os

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Get model responses for prompts')
    parser.add_argument('--mode', type=str, default='gen', choices=['gen', 'normal'],
                       help='Mode to use')
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       help='Name of the model to use')
    parser.add_argument('--client', type=str, default='openai',
                       help='Client to use')
    parser.add_argument('--data_dir', type=str, default='../data/',
                       help='Directory to save the results')
    parser.add_argument('--api_config', type=str, default='../libra_eval/config/api_config.json',
                       help='Path to the API config file')
    args = parser.parse_args()
    
    args.input_file = os.path.join(args.data_dir, f"ruozhibench_gen.jsonl")
    args.results_dir = os.path.join(args.data_dir, f"response_{args.mode}")

    # Create results directory if it doesn't exist
    os.makedirs(args.results_dir, exist_ok=True)

    # Check if output file already exists
    output_file = os.path.join(args.results_dir, f"{args.model.split('/')[-1]}.jsonl")
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
    df = pd.read_json(args.input_file, lines=True)
    if args.mode == "gen":
        prompts = df['question_en'].tolist()
    elif args.mode == "normal":
        df = df.dropna(subset=['pair'])
        prompts = df['pair'].tolist()

    messages_list = client.construct_message_list(prompts, system_role="You are a helpful assistant.")
    responses = client.multi_call(messages_list, max_tokens=1024)
    
    if args.mode == "gen":
        df['response'] = responses
    elif args.mode == "normal":
        df['pair_response'] = responses
    
    # Save the results
    df.to_json(output_file, orient='records', force_ascii=False, lines=True)
    df.to_csv(output_file.replace('.jsonl', '.csv'), index=False)
    print(f"Results saved to {output_file}")
    
    return df


if __name__ == "__main__":
    main()
