import json

def calculate_winrates(responses_file, sft_file):
    with open(responses_file, 'r') as f:
        response_data = [json.loads(line) for line in f]

    with open(sft_file, 'r') as f:
        sft_data = [json.loads(line) for line in f]

    sft_scores = {}
    for entry in sft_data:
        key = (entry['question_id'], tuple(entry['judge']))
        sft_scores[key] = entry['score']

    win_counts = {}
    total_counts = {}

    for entry in response_data:
        key = (entry['question_id'], tuple(entry['judge']))
        model = entry['model']
        if model not in win_counts:
            win_counts[model] = 0
            total_counts[model] = 0

        if key in sft_scores:
            total_counts[model] += 1
            if entry['score'] > sft_scores[key]:
                win_counts[model] += 1
            
    result = []
    for model in win_counts:
        win_rate = win_counts[model] / total_counts[model] if total_counts[model] > 0 else 0
        result.append({"model": model, "win_rate": win_rate})
        print(f"model: {model}, win_count: {win_counts[model]}, total_count: {total_counts[model]}, win_rate: {win_rate}")
    
    return result

responses_file = '.\data\mt_bench\model_judgment\gpt-4-turbo_single.jsonl'
sft_file = '.\data\mt_bench\model_judgment\sft_score.jsonl'

win_rates = calculate_winrates(responses_file, sft_file)

output_file = '.\data\mt_bench\model_judgment\win_rates.jsonl'
with open(output_file, 'w') as f:
    for entry in win_rates:
        f.write(json.dumps(entry) + '\n')

# 打印结果
for entry in win_rates:
    print(json.dumps(entry))