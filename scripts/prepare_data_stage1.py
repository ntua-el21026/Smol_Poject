import os
import argparse
import time
import glob
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# --- ΡΥΘΜΙΣΕΙΣ ---
TOKENIZER_ID = "meta-llama/Meta-Llama-3-8B"
BLOCK_SIZE = 2048
CHUNK_SIZE = 200000  # Σώζουμε κάθε 200.000 δείγματα (περίπου κάθε 15-20 λεπτά)

def main():
    parser = argparse.ArgumentParser()
    default_output = os.path.join(os.getenv('SCRATCH', '.'), 'data/stage1_slimpajama_packed')
    parser.add_argument("--output_dir", type=str, default=default_output)
    parser.add_argument("--skip", type=int, default=0, help="Number of samples to skip (to avoid duplicates)")
    parser.add_argument("--num_samples", type=int, default=30000000) 
    args = parser.parse_args()

    # Δημιουργία φακέλου εξόδου
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"--- Data Preparation (Chunked Mode) ---")
    print(f"Output Dir: {args.output_dir}")

    # 1. Έλεγχος για Resume (Τι έχουμε κάνει ήδη;)
    print(f"Target: {args.num_output_sequences} sequences x 2048 tokens = {args.num_output_sequences * 2048 / 1e9:.2f} Billion Tokens")
    print(f"Skipping first {args.skip_raw_docs} RAW documents...")


    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Dataset (Streaming)
    ds_stream = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    
    # 3. Explicit Skip (Το πιο σημαντικό σημείο)
    if args.skip_raw_docs > 0:
        ds_stream = ds_stream.skip(args.skip_raw_docs)

    # 4. Processing Loop
    current_buffer = []      # Tokens buffer for packing
    batch_data = []          # List of packed dicts for the current chunk
# Μετρητές
    raw_docs_processed = 0
    sequences_created = 0
    chunk_idx = 0
    
    iterator = iter(ds_stream)
    pbar = tqdm(total=args.num_output_sequences, desc="Packed Seqs Created", unit="seq")
    while sequences_created < args.num_output_sequences:
        try:
            example = next(iterator)
            raw_docs_processed += 1
        except StopIteration:
            print("Dataset finished (End of Stream)!")
            break

        # Tokenize
        tokens = tokenizer(example["text"])["input_ids"]
        tokens.append(tokenizer.eos_token_id)
        current_buffer.extend(tokens)

        # Packing Logic (κόβουμε σε BLOCK_SIZE)
        while len(current_buffer) >= BLOCK_SIZE:
            block = current_buffer[:BLOCK_SIZE]
            current_buffer = current_buffer[BLOCK_SIZE:]
            
            # Προσθήκη στο τρέχον chunk
            batch_data.append({"input_ids": block})
            sequences_created += 1
            pbar.update(1)

            # ΕΛΕΓΧΟΣ: Γέμισε το Chunk; Ώρα για Save!
            if len(batch_data) >= CHUNK_SIZE:
                save_chunk(batch_data, args.output_dir, chunk_idx)
                batch_data = [] # Καθαρισμός μνήμης
                chunk_idx += 1
        

    # Σώσε ό,τι περίσσεψε στο τέλος (αν υπάρχει)
    if len(batch_data) > 0:
        save_chunk(batch_data, args.output_dir, chunk_idx)

    pbar.close()
    print("\n" + "="*40)
    print(f"--- SUCCESS: Job Finished ---")
    print(f"Total RAW documents consumed: {raw_docs_processed}")
    print(f"Next Job (Part X) should start with --skip_raw_docs {args.skip_raw_docs + raw_docs_processed}")
    print("="*40)

def save_chunk(data, output_dir, idx):
    """Βοηθητική συνάρτηση για αποθήκευση"""
    save_path = os.path.join(output_dir, f"part_{idx:05d}") # π.χ. part_00001
    print(f"\nSaving chunk {idx} to {save_path}...")
    
    # Μετατροπή σε HuggingFace Dataset και save
    temp_ds = Dataset.from_list(data)
    temp_ds.save_to_disk(save_path)
    print("Saved!")

if __name__ == "__main__":
    main()