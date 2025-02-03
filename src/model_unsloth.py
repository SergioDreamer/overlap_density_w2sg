### Define Model and Tokenizer ###

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
from tqdm import tqdm
from datasets import DatasetDict

import torch
from unsloth import FastLanguageModel # Requires to have NVIDIA's GPU available and CUDA installed

@dataclass
class PredictorConfig(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        ...


@dataclass
class ModelConfig(PredictorConfig):
    name: str
    enable_lora: bool
    lora_modules: Optional[List[str]] = None

    def to_dict(self):
        return vars(self)


def init_model_from_unsloth(cfg: ModelConfig, max_seq_length: int = 2048):
    """
    Initializes the model from the Unsloth library. Independent of the Hugging Face Transformers library.
    * max_seq_length supports RoPE Scaling internally, so choose any!
    
    Returns the Model and Tokenizer.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    if cfg.enable_lora:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] if cfg.lora_modules is None else cfg.lora_modules,
            lora_alpha=16,
            lora_dropout=0, # Supports any, but = 0 is optimized
            bias="none", # Supports any, but = "none" is optimized
            use_gradient_checkpointing="unsloth", # True or "unsloth" for very long context
            random_state=42,
            max_seq_length=max_seq_length,
            use_rslora=False,
            loftq_config=None,
        )

    return model, tokenizer

# Example usage
# cfg = ModelConfig(name="unsloth/llama-3-8b-bnb-4bit", enable_lora=True)
# model, tokenizer = init_model_from_unsloth(cfg)


### Define model's activation collection ###

def load_model_and_save_activations(
    ds_dict: DatasetDict,
    model_cfg: str,
    acts_dir: Path,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please use a CUDA-enabled GPU for this code to run properly.")

    print(f"{torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100:.2f}% of GPU memory in use before model init")

    # Load model and tokenizer from Unsloth
    model, tokenizer = init_model_from_unsloth(cfg=model_cfg)

    # Enable faster inference
    FastLanguageModel.for_inference(model)

    print(f"{torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100:.2f}% of GPU memory in use after model init")

    def process(examples):
        return tokenizer(examples["txt"], truncation=True, padding=True, return_tensors="pt")

    # Extract train subset | Can be 'weak_' or 'strong_' train
    train_subset = [key for key in ds_dict if "train" in key][0]

    ds_dict = ds_dict.map(process, batched=True, remove_columns=ds_dict[train_subset].column_names)

    if acts_dir.exists() and all((acts_dir / f"{name}.pt").exists() for name in ds_dict.keys()):
        print("Activations already exist at", acts_dir)
    else:
        print("Saving activations to", acts_dir)
        acts_dir.mkdir(parents=True, exist_ok=True)

        

        # Gather hidden states for all splits
        def gather_hiddens(model, dataset):
            all_hidden_states = []
            for batch in tqdm(dataset, desc="Collecting activations"):
                with torch.no_grad():
                    # Convert 'input_ids' and 'attention_mask' to tensors, and move batches to the same device as the model
                    batch['input_ids'] = torch.tensor(batch['input_ids'], device=model.device).unsqueeze(0) # unsqueeze adds a batch dimension
                    batch['attention_mask'] = torch.tensor(batch['attention_mask'], device=model.device).unsqueeze(0)
                    
                    outputs = model(**batch, output_hidden_states=True) 

                # Get only the final layer's activation
                final_layer_activation = outputs.hidden_states[-1]
                # Extract final token activations for each example in batch
                final_token_activations = final_layer_activation[:, -1, :]
                all_hidden_states.append(final_token_activations.cpu())

            # Concatenate all hidden states
            return torch.cat(all_hidden_states, dim=0)

        for name, ds in ds_dict.items():
            acts = gather_hiddens(model, ds)
            acts = acts.to(torch.float32)  # Convert activations to float32 to ensure compatibility with NumPy and other operations
            torch.save(acts, acts_dir / f"{name}.pt")
            print(f"Saved activations for {name} to {acts_dir / f'{name}.pt'}")

    del model
    torch.cuda.empty_cache()
    gc.collect()