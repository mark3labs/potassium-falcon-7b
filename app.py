from potassium import Potassium, Request, Response

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = Potassium("potassium-falcon-7b")

# @app.init runs at startup, and loads models into the app's context


@app.init
def init():

    tokenizer = AutoTokenizer.from_pretrained(
        "h2oai/h2ogpt-gm-oasst1-multilang-2048-falcon-7b",
        use_fast=False,
        padding_side="left",
        trust_remote_code=True,
    )

    generate_text = pipeline(
        model="h2oai/h2ogpt-gm-oasst1-multilang-2048-falcon-7b",
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_fast=False,
        device_map={"": "cuda:0"},
    )
    context = {
        "model": generate_text
    }

    return context

# @app.handler runs for every call


@app.handler()
def handler(context: dict, request: Request) -> Response:
    prompt = request.json.get("prompt")
    model = context.get("model")
    outputs = model(
        prompt,
        min_new_tokens=2,
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1,
        temperature=float(0.3),
        repetition_penalty=float(1.2),
        renormalize_logits=True
    )

    return Response(
        json={"outputs": outputs[0]},
        status=200
    )


if __name__ == "__main__":
    app.serve()
