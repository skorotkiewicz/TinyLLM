# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["torch>=2.6,<3", "transformers==4.57.3", "datasets>=4,<5", "accelerate>=1.2,<2"]
# ///
"""Random-weight 1B Llama: pretrain, chat-tune, then chat. Nothing runs on import."""

import argparse
from pathlib import Path

TOKENIZER = "HuggingFaceTB/SmolLM2-135M-Instruct"  # Tokenizer only, never its weights.
SYSTEM = "You are a helpful assistant."
ARCH = dict(vocab_size=49152, hidden_size=2048, intermediate_size=5632,
            num_hidden_layers=20, num_attention_heads=32, num_key_value_heads=8,
            max_position_embeddings=2048, tie_word_embeddings=True)


def with_system(messages, system=SYSTEM):
    return messages if messages and messages[0]["role"] == "system" else [
        {"role": "system", "content": system}, *messages]


def examples(rows, tokenizer, stage, context):
    pending = []
    for row in rows:
        if stage == "sft":
            messages = with_system(row["messages"])
            ids = tokenizer.apply_chat_template(messages, tokenize=True)
            # ponytail: skip long chats instead of splitting turns; add turn-aware splitting if needed.
            if len(ids) > context or not any(m["role"] == "assistant" for m in messages):
                continue
            yield dict(input_ids=ids, attention_mask=[1] * len(ids), labels=ids.copy())
        else:
            pending.extend(tokenizer(row["text"], add_special_tokens=False)["input_ids"])
            pending.append(tokenizer.eos_token_id)
            end = len(pending) // context * context
            for start in range(0, end, context):
                ids = pending[start:start + context]
                yield dict(input_ids=ids, attention_mask=[1] * context, labels=ids.copy())
            pending = pending[end:]  # Only the final incomplete block is discarded.


def fit_history(messages, tokenizer, limit):
    messages = messages.copy()
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    while len(ids) > limit and len(messages) > 2:
        del messages[1:3]  # Keep the system prompt and newest user turn.
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if len(ids) > limit:
        raise ValueError("System prompt and message exceed the context window; shorten them.")
    return messages, ids


def train(args):
    import torch
    from datasets import IterableDataset, load_dataset
    from transformers import (AutoTokenizer, LlamaConfig, LlamaForCausalLM,
                              Trainer, TrainingArguments, default_data_collator, set_seed)

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Training requires a CUDA GPU with BF16 support. Use a larger training machine.")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError("Output is not empty. Choose another --output or use --resume.")
    if args.resume and (not Path(args.resume).is_dir()
                        or Path(args.resume).resolve().parent != output.resolve()):
        raise ValueError("--resume must name a checkpoint directory inside --output.")
    set_seed(42)
    source = args.resume or args.model
    tokenizer = AutoTokenizer.from_pretrained(source or TOKENIZER, local_files_only=bool(source))
    if source:
        model = LlamaForCausalLM.from_pretrained(source, local_files_only=True, torch_dtype=torch.float32)
    else:
        if len(tokenizer) != ARCH["vocab_size"]:
            raise ValueError("Tokenizer vocabulary changed; expected 49152 tokens.")
        model = LlamaForCausalLM(LlamaConfig(
            **ARCH, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id))
    model.config.use_cache = False
    context = model.config.max_position_embeddings
    tokenizer.model_max_length = context
    pretrain = args.stage == "pretrain"
    dataset = ("HuggingFaceFW/fineweb-edu", "sample-100BT") if pretrain else (
        "HuggingFaceTB/smol-smoltalk", "default")
    rows = load_dataset(*dataset, split="train", streaming=True).shuffle(seed=42, buffer_size=10000)
    data = IterableDataset.from_generator(lambda: examples(rows, tokenizer, args.stage, context))
    print(f"{model.num_parameters():,} parameters; {dataset[0]}; context {context}")
    # ponytail: one GPU, batch size 1; use sharded training when this no longer fits.
    trainer = Trainer(
        model=model, processing_class=tokenizer, train_dataset=data,
        data_collator=default_data_collator,
        args=TrainingArguments(
            output_dir=str(output), max_steps=args.steps or (300000 if pretrain else 3000),
            per_device_train_batch_size=1, gradient_accumulation_steps=32,
            learning_rate=args.lr or (3e-4 if pretrain else 2e-5),
            lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.1,
            max_grad_norm=1.0, bf16=True, gradient_checkpointing=True,
            logging_steps=10, save_steps=1000, save_total_limit=2, report_to="none",
            seed=42, dataloader_num_workers=0,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume)
    model.config.use_cache = True
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))


def chat(args):
    import torch
    from transformers import AutoTokenizer, LlamaForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = LlamaForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=dtype).to(device).eval()
    model.config.use_cache = True
    limit = model.config.max_position_embeddings - args.max_new_tokens
    if limit < 1:
        raise ValueError("--max-new-tokens must be smaller than the model context window.")
    history = with_system([], args.system)
    print("/quit to exit. Oldest exchanges are dropped when the context fills.")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if text == "/quit":
            break
        if not text:
            continue
        try:
            messages, ids = fit_history(history + [{"role": "user", "content": text}], tokenizer, limit)
        except ValueError as error:
            print(error)
            continue
        inputs = torch.tensor([ids], device=device)
        with torch.inference_mode():
            output = model.generate(
                inputs, attention_mask=torch.ones_like(inputs), max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        answer = tokenizer.decode(output[0, len(ids):], skip_special_tokens=True)
        print(f"Assistant: {answer}")
        history = messages + [{"role": "assistant", "content": answer}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["pretrain", "sft", "chat"])
    parser.add_argument("--model", help="Local trained model directory, required for sft/chat")
    parser.add_argument("--output", help="Training output directory, must be empty unless resuming")
    parser.add_argument("--resume", help="Local Trainer checkpoint inside --output")
    parser.add_argument("--steps", type=int, help="Optimizer steps, default: pretrain 300000, sft 3000")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--system", default=SYSTEM, help="Chat system prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.stage in ("sft", "chat") and not args.model:
        parser.error("sft/chat require --model pointing to your locally trained weights")
    if args.model and not Path(args.model).is_dir():
        parser.error("--model must be a local directory")
    if args.stage == "pretrain" and args.model:
        parser.error("pretrain starts from random weights; use --resume to continue a checkpoint")
    if args.stage != "chat" and not args.output:
        parser.error("training requires --output")
    if (args.steps is not None and args.steps < 1) or (args.lr is not None and not 0 < args.lr < 1):
        parser.error("--steps must be positive; --lr must be between 0 and 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    try:
        (chat if args.stage == "chat" else train)(args)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
