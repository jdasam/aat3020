# assignment3_pre_defined.py
import torch
import matplotlib.pyplot as plt
from datasets import Dataset


def create_gpt2_dataset(txt_path, tokenizer, block_size=128):
    """Read a text file, tokenize with GPT-2 tokenizer, return a HuggingFace Dataset for causal LM training."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', ' ')
    token_ids = tokenizer(text)['input_ids']

    examples = []
    for i in range(0, len(token_ids) - block_size, block_size // 2):
        chunk = token_ids[i: i + block_size]
        examples.append({'input_ids': chunk, 'labels': chunk})

    return Dataset.from_list(examples)


def get_lora_model(base_model):
    """Apply LoRA to a GPT-2 model and return a PeftModel."""
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        target_modules=["c_attn", "c_proj", "c_fc"],
        lora_dropout=0.05,
        bias="none",
    )
    return get_peft_model(base_model, config)


def count_trainable_params(model):
    """Print and return (trainable, total) parameter counts."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"학습 파라미터: {trainable:,} / 전체: {total:,}  ({100 * trainable / total:.2f}%)")
    return trainable, total


def plot_loss_curves(full_losses, lora_losses=None):
    """Plot Full fine-tuning and LoRA training loss on the same graph."""
    plt.figure(figsize=(10, 4))
    plt.plot(full_losses, label='Full fine-tuning', color='#3498db', alpha=0.9)
    if lora_losses is not None:
        plt.plot(lora_losses, label='LoRA', color='#e67e22', alpha=0.9)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss: Full Fine-tuning vs LoRA')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def generate_text(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0, num_samples=1, seed=None):
    """Generate text from a fine-tuned GPT-2 model (works with both Full and LoRA models).
    Returns a list of strings (length == num_samples).
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            num_return_sequences=num_samples,
            pad_token_id=tokenizer.eos_token_id,
        )

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
