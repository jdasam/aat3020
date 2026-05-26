# assignment3_pre_defined.py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datasets import Dataset


# ──────────────────────────────────────────────
# Problem 2 helpers: LSTM LM
# ──────────────────────────────────────────────

def create_lstm_lm_dataset(txt_path, tokenizer, seq_len=128):
    """Read a text file, tokenize with GPT-2 tokenizer, and return a 1D tensor for LM training."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', ' ')
    token_ids = tokenizer.encode(text)
    return torch.tensor(token_ids, dtype=torch.long)


def train_lm(model, token_ids, max_steps=500, batch_size=32, seq_len=128,
             lr=1e-3, device='cuda', log_every=100):
    """
    Training loop for LSTM LM. token_ids is the return value of create_lstm_lm_dataset.
    returns: list of loss values (length == max_steps)
    """
    model.train()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    if len(token_ids) < seq_len + 2:
        raise ValueError(f"Corpus is too short. At least {seq_len + 2} tokens required, but only {len(token_ids)} found.")
    losses = []
    n = len(token_ids)

    for step in range(max_steps):
        # Random batch sampling
        starts = torch.randint(0, n - seq_len - 1, (batch_size,))
        x = torch.stack([token_ids[s: s + seq_len] for s in starts]).to(device)
        y = torch.stack([token_ids[s + 1: s + seq_len + 1] for s in starts]).to(device)

        logits, _ = model(x)                             # [B, T, vocab_size]
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"Step {step + 1}/{max_steps}  Loss: {loss.item():.4f}")

    return losses


def generate_text_lstm(model, tokenizer, prompt, max_new_tokens=100, temperature=1.0):
    """Generate text using a trained LSTM LM."""
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        logits, hidden_and_cell = model(input_tensor)

    generated = list(input_ids)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            last_logit = logits[:, -1, :] / max(temperature, 1e-6)
            probs = torch.softmax(last_logit, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            next_input = torch.tensor([[next_token]], device=device)
            logits, hidden_and_cell = model(next_input, hidden_and_cell)

    return tokenizer.decode(generated)


def visualize_gates(model, tokenizer, text, max_tokens=40):
    """
    Visualize gate activations of a trained LSTM LM as a bar chart.
    MyLSTM weight_ih, weight_hh order: [i | f | g | o]
    """
    tokens = tokenizer.encode(text)[:max_tokens]
    labels = [tokenizer.decode([t]).replace(' ', '·') for t in tokens]

    model.eval()
    device = next(model.parameters()).device
    H = model.lstm.hidden_size

    forget_vals, input_vals, output_vals = [], [], []

    with torch.no_grad():
        x = model.embedding(torch.tensor([tokens], device=device))  # [1, T, H]
        h = torch.zeros(1, H, device=device)
        c = torch.zeros(1, H, device=device)

        for t in range(x.shape[1]):
            x_t = x[:, t, :]
            gates = model.lstm.weight_ih(x_t) + model.lstm.weight_hh(h)  # [1, 4H]
            i_g = torch.sigmoid(gates[:, :H])
            f_g = torch.sigmoid(gates[:, H:2 * H])
            g   = torch.tanh(gates[:, 2 * H:3 * H])
            o_g = torch.sigmoid(gates[:, 3 * H:])
            c = f_g * c + i_g * g
            h = o_g * torch.tanh(c)

            forget_vals.append(f_g.mean().item())
            input_vals.append(i_g.mean().item())
            output_vals.append(o_g.mean().item())

    pos = range(len(labels))
    fig, axes = plt.subplots(3, 1, figsize=(max(12, len(labels) * 0.5), 8))
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    names  = ['Forget Gate', 'Input Gate', 'Output Gate']
    for ax, vals, name, color in zip(axes, [forget_vals, input_vals, output_vals], names, colors):
        ax.bar(pos, vals, color=color, alpha=0.8)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels, rotation=60, ha='right', fontsize=9)
        ax.set_ylabel(name)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    plt.suptitle('LSTM Gate Activations (mean over hidden units)', fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_loss_curves(lstm_losses, gpt2_losses=None):
    """Plot loss curves for Problem 2 and 3 on the same graph."""
    plt.figure(figsize=(10, 4))
    plt.plot(lstm_losses, label='LSTM LM', color='#3498db', alpha=0.9)
    if gpt2_losses is not None:
        plt.plot(gpt2_losses, label='GPT-2 fine-tuned', color='#e67e22', alpha=0.9)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────
# Problem 3 helpers: GPT-2 fine-tuning
# ──────────────────────────────────────────────

def create_gpt2_lm_dataset(txt_path, tokenizer, block_size=128):
    """Create a HuggingFace Dataset for GPT-2 fine-tuning. Uses the same corpus as Problem 2."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        text = f.read().replace('\n', ' ')
    token_ids = tokenizer(text)['input_ids']

    examples = []
    for i in range(0, len(token_ids) - block_size, block_size // 2):
        chunk = token_ids[i: i + block_size]
        examples.append({'input_ids': chunk})

    return Dataset.from_list(examples)
