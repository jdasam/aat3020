import torch
import torch.nn as nn
from pathlib import Path
from tqdm.auto import tqdm

from torch.nn.utils.rnn import PackedSequence, pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence
from torch.utils.data import DataLoader


class TranslationSet:
  def __init__(self, df, src_tokenizer, tgt_tokenizer):
    self.data = df
    self.src_tokenizer = src_tokenizer
    self.tgt_tokenizer = tgt_tokenizer

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    selected_row = self.data.iloc[idx]
    source = selected_row['원문']
    target = selected_row['번역문']

    source_enc = self.src_tokenizer(source)['input_ids']
    target_enc = self.tgt_tokenizer(target)['input_ids']

    return torch.LongTensor(source_enc), torch.LongTensor(target_enc[:-1]), torch.LongTensor(target_enc[1:])


def pack_collate(raw_batch):
  source, target, shifted_target = zip(*raw_batch)
  return pack_sequence(source, enforce_sorted=False), pack_sequence(target, enforce_sorted=False), pack_sequence(shifted_target, enforce_sorted=False)


class Trainer:
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device, save_dir='./'):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    self.train_loader = train_loader
    self.valid_loader = valid_loader

    self.model.to(device)

    self.best_valid_accuracy = 0
    self.device = device

    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []
    self.save_dir = Path(save_dir)
    self.save_dir.mkdir(exist_ok=True)

  def save_model(self, path='kor_eng_translator_attention_model.pt'):
    save_path = self.save_dir / path
    torch.save({'model': self.model.state_dict(), 'optim': self.optimizer.state_dict()}, save_path)

  def train_by_num_epoch(self, num_epochs):
    for epoch in tqdm(range(num_epochs)):
      self.model.train()
      with tqdm(self.train_loader, leave=False) as pbar:
        for batch in pbar:
          loss_value = self._train_by_single_batch(batch)
          self.training_loss.append(loss_value)
          pbar.set_description(f"Epoch {epoch+1}, Loss {loss_value:.4f}")
      self.model.eval()
      validation_loss, validation_acc = self.validate()
      self.validation_loss.append(validation_loss)
      self.validation_acc.append(validation_acc)

      if validation_acc > self.best_valid_accuracy:
        print(f"Saving the model with best validation accuracy: Epoch {epoch+1}, Acc: {validation_acc:.4f} ")
        self.save_model('kor_eng_translator_attention_model_best.pt')
      else:
        self.save_model('kor_eng_translator_attention_model_last.pt')
      self.best_valid_accuracy = max(validation_acc, self.best_valid_accuracy)

  def _train_by_single_batch(self, batch):
    src, tgt, shifted_tgt = batch
    src = src.to(self.device)
    tgt = tgt.to(self.device)
    shifted_tgt = shifted_tgt.to(self.device)

    prob = self.model(src, tgt)

    if isinstance(prob, PackedSequence):
      loss = self.loss_fn(prob.data, shifted_tgt.data)
    else:
      loss = self.loss_fn(prob, shifted_tgt)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()

    return loss.item()

  def validate(self, external_loader=None):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader

    self.model.eval()

    validation_loss = 0
    num_correct_guess = 0
    num_data = 0
    with torch.inference_mode():
      for batch in tqdm(loader, leave=False):
        src, tgt, shifted_tgt = batch
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        shifted_tgt = shifted_tgt.to(self.device)

        prob = self.model(src, tgt)

        if isinstance(prob, PackedSequence):
          loss = self.loss_fn(prob.data, shifted_tgt.data)
        else:
          loss = self.loss_fn(prob, shifted_tgt)

        validation_loss += loss.item() * len(prob.data)
        if isinstance(prob, PackedSequence):
          num_correct_guess += (prob.data.argmax(dim=-1) == shifted_tgt.data).sum().item()
        else:
          num_correct_guess += (prob.argmax(dim=-1) == shifted_tgt.data).sum().item()
        num_data += len(prob.data)
    return validation_loss / num_data, num_correct_guess / num_data


def nll_loss(pred, target, eps=1e-8):
  if pred.ndim == 3:
    pred = pred.flatten(0, 1)
  if target.ndim == 2:
    target = target.flatten(0, 1)
  assert pred.ndim == 2
  assert target.ndim == 1
  return -torch.log(pred[torch.arange(len(target)), target] + eps).mean()


class TranslatorBi(nn.Module):
  def __init__(self, src_tokenizer, tgt_tokenizer, hidden_size=256, num_layers=3):
    super().__init__()
    self.src_tokenizer = src_tokenizer
    self.tgt_tokenizer = tgt_tokenizer

    self.src_vocab_size = self.src_tokenizer.vocab_size
    self.tgt_vocab_size = self.tgt_tokenizer.vocab_size

    self.src_embedder = nn.Embedding(self.src_vocab_size, hidden_size)
    self.tgt_embedder = nn.Embedding(self.tgt_vocab_size, hidden_size)

    self.encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
    self.decoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    self.decoder_proj = nn.Linear(hidden_size, self.tgt_vocab_size)

  def run_encoder(self, x):
    if isinstance(x, PackedSequence):
      emb_x = PackedSequence(self.src_embedder(x.data), batch_sizes=x.batch_sizes, sorted_indices=x.sorted_indices, unsorted_indices=x.unsorted_indices)
    else:
      emb_x = self.src_embedder(x)

    enc_hidden_state_by_t, last_hidden = self.encoder(emb_x)

    last_hidden_sum = last_hidden.reshape(self.encoder.num_layers, 2, last_hidden.shape[1], -1).mean(dim=1)
    if isinstance(x, PackedSequence):
      hidden_mean = enc_hidden_state_by_t.data.reshape(-1, 2, last_hidden_sum.shape[-1]).mean(1)
      enc_hidden_state_by_t = PackedSequence(hidden_mean, x[1], x[2], x[3])
    else:
      enc_hidden_state_by_t = enc_hidden_state_by_t.reshape(x.shape[0], x.shape[1], 2, -1).mean(dim=2)

    return enc_hidden_state_by_t, last_hidden_sum

  def run_decoder(self, y, last_hidden_state):
    if isinstance(y, PackedSequence):
      emb_y = PackedSequence(self.tgt_embedder(y.data), batch_sizes=y.batch_sizes, sorted_indices=y.sorted_indices, unsorted_indices=y.unsorted_indices)
    else:
      emb_y = self.tgt_embedder(y)
    out, decoder_last_hidden = self.decoder(emb_y, last_hidden_state)
    return out, decoder_last_hidden

  def forward(self, x, y):
    enc_hidden_state_by_t, last_hidden_sum = self.run_encoder(x)
    out, decoder_last_hidden = self.run_decoder(y, last_hidden_sum)

    if isinstance(out, PackedSequence):
      logits = self.decoder_proj(out.data)
      probs = torch.softmax(logits, dim=-1)
      probs = PackedSequence(probs, batch_sizes=y.batch_sizes, sorted_indices=y.sorted_indices, unsorted_indices=y.unsorted_indices)
    else:
      logits = self.decoder_proj(out)
      probs = torch.softmax(logits, dim=-1)
    return probs


class MLP(nn.Module):
  def __init__(self, in_size, hidden_size):
    super().__init__()
    self.input_size = in_size
    self.layer = nn.Sequential(nn.Linear(in_size, hidden_size),
                               nn.ReLU(),
                               nn.Linear(hidden_size, in_size))

  def forward(self, x):
    return self.layer(x)


class PosEncoding(nn.Module):
  def __init__(self, size, max_t):
    super().__init__()
    self.size = size
    self.max_t = max_t
    self.register_buffer('encoding', self._prepare_emb())

  def _prepare_emb(self):
    dim_axis = 10000 ** (torch.arange(self.size // 2) * 2 / self.size)
    timesteps = torch.arange(self.max_t)
    pos_enc_in = timesteps.unsqueeze(1) / dim_axis.unsqueeze(0)
    pos_enc_sin = torch.sin(pos_enc_in)
    pos_enc_cos = torch.cos(pos_enc_in)
    pos_enc = torch.stack([pos_enc_sin, pos_enc_cos], dim=-1).reshape([self.max_t, 512])
    return pos_enc

  def forward(self, x):
    return self.encoding[x]


class ResidualLayerNormModule(nn.Module):
  def __init__(self, submodule):
    super().__init__()
    self.submodule = submodule
    self.layer_norm = nn.LayerNorm(self.submodule.input_size)

  def forward(self, x, mask=None, y=None):
    if y is not None:
      res_x = self.submodule(x, y, mask)
    elif mask is not None:
      res_x = self.submodule(x, mask)
    else:
      res_x = self.submodule(x)
    x = x + res_x
    return self.layer_norm(x)


class TransformerTrainer(Trainer):
  def __init__(self, model, optimizer, loss_fn, train_loader, valid_loader, device):
    super().__init__(model, optimizer, loss_fn, train_loader, valid_loader, device)
    self.num_iter = 0
    self._adjust_optim()

  def _adjust_optim(self):
    self.num_iter += 1
    self.optimizer.param_groups[0]['lr'] = 512 ** (-0.5) * min(self.num_iter ** (-0.5), self.num_iter * 4000 ** (-1.5))

  def _train_by_single_batch(self, batch):
    src, tgt_i, tgt_o = batch
    pred = self.model(src.to(self.device), tgt_i.to(self.device))
    pred = pack_padded_sequence(pred, pad_packed_sequence(tgt_o)[1], batch_first=True, enforce_sorted=False)
    loss = self.loss_fn(pred.data, tgt_o.data)
    loss.backward()
    self.optimizer.step()
    self.optimizer.zero_grad()
    self._adjust_optim()
    return loss.item()

  def validate(self, external_loader=None):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader

    self.model.eval()

    validation_loss = 0
    num_correct_guess = 0
    num_data = 0
    with torch.inference_mode():
      for batch in tqdm(loader, leave=False):
        src, tgt_i, tgt_o = batch
        tgt_o = tgt_o.to(self.device)
        pred = self.model(src.to(self.device), tgt_i.to(self.device))
        pred = pack_padded_sequence(pred, pad_packed_sequence(tgt_o)[1], batch_first=True, enforce_sorted=False)

        if isinstance(pred, PackedSequence):
          loss = self.loss_fn(pred.data, tgt_o.data)
        else:
          loss = self.loss_fn(pred, tgt_o)

        validation_loss += loss.item() * len(pred.data)
        if isinstance(pred, PackedSequence):
          num_correct_guess += (pred.data.argmax(dim=-1) == tgt_o.data).sum().item()
        else:
          num_correct_guess += (pred.argmax(dim=-1) == tgt_o.data).sum().item()
        num_data += len(pred.data)
    return validation_loss / num_data, num_correct_guess / num_data


def pad_collate(raw_batch):
  srcs = [x[0] for x in raw_batch]
  tgts_i = [x[1][:-1] for x in raw_batch]
  tgts_o = [x[1][1:] for x in raw_batch]

  srcs = pad_sequence(srcs, batch_first=True)
  tgts_i = pad_sequence(tgts_i, batch_first=True)
  tgts_o = pack_sequence(tgts_o, enforce_sorted=False)
  return srcs, tgts_i, tgts_o
