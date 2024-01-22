from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
import torch
import pandas as pd

transcripts_file = 'transcripts.csv'
transcripts_df = pd.read_csv(transcripts_file)


transcripts = transcripts_df['transcript'].tolist()


model_name = 'gpt2' 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

max_position_embeddings = 4096
config = GPT2Config.from_pretrained(model_name, max_position_embeddings=max_position_embeddings)
model = GPT2LMHeadModel.from_pretrained(model_name, config=config,ignore_mismatched_sizes=True)

generated_categories = []
for transcript in transcripts:
    tokenized_transcript = tokenizer.encode(transcript, return_tensors="pt", truncation=True)
    attention_mask = torch.ones_like(tokenized_transcript)  # Set attention mask
    output = model.generate(tokenized_transcript, attention_mask=attention_mask, max_length=2048, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, do_sample=True)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_categories.append(generated_text)


transcripts_df['generated_category'] = generated_categories

output_file_path = 'transcripts_with_generated_categories.csv'
transcripts_df.to_csv(output_file_path, index=False)

print(f"Output file saved to: {output_file_path}")
