import torch
from transformers import AutoTokenizer
from bible import BibleLLM

tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

ckpt_path = "ckpt/epoch=1-step=4693.ckpt"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = BibleLLM.load_from_checkpoint(ckpt_path, tokenizer=tokenizer).to(device)
model.eval()


def generate(prompt: str, max_tokens: int = 200) -> None:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    for _ in range(max_tokens):
        x = ids[:, -model.hparams.seq_len :]
        pad = x != tokenizer.pad_token_id
        L = x.size(1)
        causal = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        mask = pad.unsqueeze(1) & causal.unsqueeze(0)
        logits = model(x, mask)
        next_id = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_id], dim=1)
        if next_id.item() == tokenizer.eos_token_id:
            break
    print(tokenizer.decode(ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    while True:
        prompt = input(">>> ")
        if not prompt.strip():
            break
        generate(prompt)
