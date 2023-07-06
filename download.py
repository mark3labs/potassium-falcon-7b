# This file runs during container build time to get model weights built into the container

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = AutoTokenizer.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        use_fast=False,
        padding_side="left",
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        "tiiuae/falcon-7b-instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )


if __name__ == "__main__":
    download_model()
