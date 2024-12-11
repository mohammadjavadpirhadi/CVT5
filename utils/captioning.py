import torch
from transformers import AutoTokenizer

umt5_tokenizer = AutoTokenizer.from_pretrained('google/umt5-base')


def caption_logits_to_string(caption_logits):
    return umt5_tokenizer.batch_decode(
        torch.argmax(caption_logits, dim=-1), skip_special_tokens=True
    )


def caption_ids_to_string(caption_ids):
    return umt5_tokenizer.batch_decode(
        caption_ids, skip_special_tokens=True
    )
