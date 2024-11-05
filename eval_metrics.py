import argparse
import re
import json
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer, util
from multiprocessing import Pool
import torch

# Argument parsing setup
parser = argparse.ArgumentParser(description="Fix JSON formatting, clean 'generated' fields, and calculate metrics.")
parser.add_argument("file_path", type=str, help="Path to the JSON file to process.")
args = parser.parse_args()
file_path = args.file_path

# Step 1: Fix JSON formatting if there are missing commas between objects
def fix_json_formatting(file_path):
    # Read the JSON file as a string
    with open(file_path, 'r') as file:
        json_string = file.read()

    # Fix missing commas between objects in the array
    fixed_json_string = re.sub(r'\}\s*\{', '},{', json_string)

    # Save the fixed JSON back to the file
    with open(file_path, 'w') as fixed_file:
        fixed_file.write(fixed_json_string)

    print(f"Fixed JSON saved to: {file_path}")

fix_json_formatting(file_path)

# Step 2: Clean the "generated" field by removing everything before "Answer:" keyword
def clean_generated_field(data):
    for item in data:
        if "generated" in item:
            generated_text = item["generated"]
            keyword = "Answer:"
            if keyword in generated_text:
                # Keep everything after "Answer:"
                cleaned_text = generated_text.split(keyword, 1)[1]
                item["generated"] = cleaned_text.strip()
    return data

# Load JSON data
with open(file_path, 'r') as file:
    json_data = json.load(file)

# Clean the "generated" fields
json_data = clean_generated_field(json_data)

# Step 3: Initialize metrics: ROUGE, BLEU, and BERT for similarity scoring
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bert_model = SentenceTransformer('bert-base-nli-mean-tokens', device='cuda' if torch.cuda.is_available() else 'cpu')
file_format = file_path.split('/')[-1].replace(".json", ".csv")

# Function to calculate ROUGE and BLEU for a single sample
def compute_metrics(sample):
    true_desc = sample["true"]
    gen_desc = sample["generated"]

    # ROUGE Scores
    rouge_scores = scorer.score(true_desc, gen_desc)

    # BLEU Score with smoothing (unigram and bigram)
    smoothing_function = SmoothingFunction().method1
    bleu_score = sentence_bleu([true_desc.split()], gen_desc.split(), weights=(0.5, 0.5), smoothing_function=smoothing_function)

    return {
        "id": sample["id"],
        "video": sample["video"],
        "rouge1": rouge_scores['rouge1'].fmeasure,
        "rouge2": rouge_scores['rouge2'].fmeasure,
        "rougeL": rouge_scores['rougeL'].fmeasure,
        "bleu": bleu_score
    }

# Batch process BERT embeddings for similarity
def compute_bert_similarity(true_descriptions, generated_descriptions):
    true_embs = bert_model.encode(true_descriptions, convert_to_tensor=True, batch_size=512)
    gen_embs = bert_model.encode(generated_descriptions, convert_to_tensor=True, batch_size=512)

    # Calculate cosine similarities
    bert_similarities = [util.pytorch_cos_sim(true_emb, gen_emb).item() for true_emb, gen_emb in zip(true_embs, gen_embs)]
    return bert_similarities

# Prepare descriptions for BERT similarity
true_descriptions = [sample['true'] for sample in json_data]
generated_descriptions = [sample['generated'] for sample in json_data]

# Compute BERT similarity in batch
bert_similarities = compute_bert_similarity(true_descriptions, generated_descriptions)

# Compute ROUGE and BLEU in parallel using multiprocessing
with Pool() as pool:
    metric_results = pool.map(compute_metrics, json_data)

# Combine results
for i, result in enumerate(metric_results):
    result['bert_similarity'] = bert_similarities[i]

# Convert results to DataFrame
df = pd.DataFrame(metric_results)

# Calculate averages for numeric columns
avg_results = df.mean(numeric_only=True)

# Add the average row to the DataFrame
avg_results_row = pd.DataFrame(avg_results).transpose()
avg_results_row['id'] = 'average'
avg_results_row['video'] = 'average'

# Append the average row to the DataFrame
df_with_avg = pd.concat([df, avg_results_row], ignore_index=True)

# Save to CSV with the average row included
df_with_avg.to_csv(file_format, index=False)

print(f"Metrics calculated and saved to {file_format}")
