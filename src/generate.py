"""
generate.py - The grounded generation step.
Loads Qwen2.5-7B-Instruct in 4-bit (needs a GPU), and exposes answer(question),
which: retrieves context (via retriever.py) -> builds a constrained prompt ->
generates an answer grounded ONLY in that context -> guarantees a citation line.

NOTE: This needs a GPU (e.g. Google Colab T4). It will be very slow / may fail on CPU.
Run:  python src/generate.py   (self-test)
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

from retriever import retrieve

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# 4-bit quantization so a 7B model fits a 16 GB GPU (~5 GB instead of ~14 GB).
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading LLM tokenizer...")
llm_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading Qwen2.5-7B (this takes a few minutes the first time)...")
llm = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
print("LLM loaded.")


SYSTEM_PROMPT = """You are a helpful PyTorch documentation assistant. Answer the user's question using ONLY the provided context from the PyTorch documentation.

Rules:
- Base your answer strictly on the context below. Do not add facts, recommendations, or opinions that are not supported by the context.
- If the user asks which option to choose (e.g. "which optimizer should I use"), this IS answerable: do not refuse and do not say you lack information. Instead, explain the relevant options that appear in the context and what each does, and let the user decide. Do not pick one for them.
- Only if the context is entirely unrelated to the question (the documentation contains nothing on the topic) should you respond exactly: "I don't have information on that in the provided documentation." — and in that case, write nothing else.
- Be concise and include code examples from the context when relevant.

IMPORTANT: You MUST end every answer with a citation line listing the sources you used, in exactly this format:
Sources: [torch.nn.Linear], [torch.optim.Adam]

Only cite sources you actually used. Always include this citation line."""


def build_context(hits):
    """Format retrieved chunks into a labeled context block."""
    blocks = [f"[Source: {h['source']}]\n{h['text']}" for h in hits]
    return "\n\n---\n\n".join(blocks)


def answer(question, final_k=8):
    hits = retrieve(question, final_k=final_k)
    context = build_context(hits)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    prompt = llm_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = llm_tokenizer(prompt, return_tensors="pt").to(llm.device)

    outputs = llm.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.1,
        do_sample=True,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=llm_tokenizer.eos_token_id,
    )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    response = llm_tokenizer.decode(generated, skip_special_tokens=True).strip()

    sources = [h["source"] for h in hits]

    # Safety net: if the model refused, don't append citations.
    refusal = "I don't have information on that in the provided documentation" in response
    if not refusal and "Sources:" not in response and "Source:" not in response:
        response = response + "\n\nSources: " + ", ".join(f"[{s}]" for s in sources)

    return response, sources


if __name__ == "__main__":
    for q in [
        "How do I create a linear layer?",
        "How do I use the Adam optimizer?",
        "What optimizer should I use for training?",
        "How do I bake a chocolate cake?",
    ]:
        resp, srcs = answer(q)
        print("Q:", q)
        print("A:", resp)
        print("=" * 70)