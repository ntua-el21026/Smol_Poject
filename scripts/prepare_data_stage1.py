import os
import argparse
import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# Ρυθμίσεις
TOKENIZER_ID = "meta-llama/Meta-Llama-3-8B"
BLOCK_SIZE = 2048

def main():
    parser = argparse.ArgumentParser()
    # Αποθήκευση στο SCRATCH (γρήγορος shared δίσκος)
    default_output = os.path.join(os.getenv('SCRATCH', '.'), 'data/stage1_slimpajama_packed')
    parser.add_argument("--output_dir", type=str, default=default_output)
    # Στόχος: ~30 Billion Tokens. 
    # Υποθέτουμε μέσο όρο ~1000 tokens ανά κείμενο στο SlimPajama -> 30M samples
    parser.add_argument("--num_samples", type=int, default=30000000) 
    args = parser.parse_args()

    print(f"--- Data Preparation Started ---")
    print(f"Output Dir: {args.output_dir}")
    print(f"Target Samples: {args.num_samples}")

    # 1. Tokenizer Setup
    print(f"Loading Tokenizer: {TOKENIZER_ID}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:
        print("\nERROR: Δεν μπόρεσα να φορτώσω τον Llama 3 Tokenizer.")
        print("Έχεις κάνει 'huggingface-cli login'; Έχεις πάρει access στο Llama 3 στο site;")
        raise e

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Dataset Loading (Streaming)
    print("Opening SlimPajama Stream...")
    ds_stream = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)

    # 3. Packing Generator Logic
    def pack_data_generator():
        current_buffer = []
        count = 0
        start_time = time.time()
        
        # tqdm progress bar
        iterator = iter(ds_stream)
        pbar = tqdm(total=args.num_samples, desc="Processing Docs")

        while count < args.num_samples:
            try:
                example = next(iterator)
            except StopIteration:
                break

            # Tokenize & Add EOS
            tokens = tokenizer(example["text"])["input_ids"]
            tokens.append(tokenizer.eos_token_id)
            current_buffer.extend(tokens)
            
            count += 1
            pbar.update(1)

            # Όσο ο buffer έχει αρκετά tokens για ένα block, κόβε και δίνε
            while len(current_buffer) >= BLOCK_SIZE:
                yield {"input_ids": current_buffer[:BLOCK_SIZE]}
                current_buffer = current_buffer[BLOCK_SIZE:]
        
        pbar.close()
        elapsed = time.time() - start_time
        print(f"\nFinished processing {count} documents in {elapsed/3600:.2f} hours.")

    # 4. Execution & Save
    print("Starting processing... (This will take hours)")
    final_ds = Dataset.from_generator(pack_data_generator)
    
    # Split Train/Val (99.5% Train, 0.5% Val)
    print("Splitting dataset...")
    splits = final_ds.train_test_split(test_size=0.005, seed=42)
    
    train_path = os.path.join(args.output_dir, "train")
    val_path = os.path.join(args.output_dir, "val")

    print(f"Saving Train to {train_path}...")
    splits["train"].save_to_disk(train_path)
    
    print(f"Saving Validation to {val_path}...")
    splits["test"].save_to_disk(val_path)
    
    print("--- SUCCESS: Data is ready for Nanotron! ---")

if __name__ == "__main__":
    main()