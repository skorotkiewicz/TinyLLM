## From-scratch 100M and 1B chat models

`TinyLLM1B.py` is separate from the toy demos. By default it initializes **1,002,522,624
parameters randomly**, using Hugging Face's Llama implementation: 20 layers,
2048 hidden width, grouped-query attention, tied embeddings, and 2048-token context.
Only the 49,152-token tokenizer comes from `HuggingFaceTB/SmolLM2-135M-Instruct`.
No pretrained model weights are used.

Two training stages are needed. Chat examples alone will not teach a random 1B
model enough language or knowledge:

- **Pretrain:** `HuggingFaceFW/fineweb-edu`, `sample-100BT`. Filtered English
  educational web text is a reasonable starting point for general English chat.
- **Chat-tune:** `HuggingFaceTB/smol-smoltalk`. Short conversations selected for
  small models, with system/user/assistant formatting. Existing system prompts
  are preserved; missing ones become "You are a helpful assistant."

### Train on a separate GPU machine

**Do not run these training commands on this machine.** Use a BF16-capable CUDA
GPU. For 1B, budget roughly 40 to 80 GB VRAM for full-parameter AdamW training; actual
usage depends on kernels and sequence length. This is not LoRA or quantized
training. Your 2 GB GPU is not suitable. Install a CUDA-compatible PyTorch build
if the environment does not already provide one.

With [uv](https://docs.astral.sh/uv/) installed, the script metadata supplies its
Python dependencies. These commands download the tokenizer and stream datasets
only when you explicitly run them. Run all commands below from `TinyLLM1B/`:

```sh
# On the training machine only. Large compute job, not a quick demo.
export CUDA_VISIBLE_DEVICES=0  # This minimal script targets one visible GPU.
uv run --python 3.12 TinyLLM1B.py pretrain --output runs/base
uv run --python 3.12 TinyLLM1B.py sft --model runs/base --output runs/chat

# Resume an interrupted stage, using the same options and its own checkpoint.
uv run --python 3.12 TinyLLM1B.py pretrain --output runs/base --resume runs/base/checkpoint-1000
```

Defaults: batch size 1, 32 gradient accumulation steps, BF16, gradient
checkpointing, cosine learning rate. Pretraining uses 300,000 optimizer steps,
about 19.7 billion tokens on one GPU; chat training uses 3,000 steps. Override
with `--steps` and `--lr`. These are starting budgets, not a claim of chat quality.
Pretraining packs documents with EOS separators and drops the final partial
block. Chat training uses full-conversation loss, not assistant-only loss, and
skips conversations longer than the context window rather than cutting turns.
The stream can repeat if a requested step budget exceeds one dataset pass.

Checkpoints include optimizer state every 1,000 steps; only the latest two are
retained. Final model/tokenizer files go into `--output`. A nonempty output is
refused unless resuming. Stream replay during resume can be slow. Allow tens of
GB of disk for checkpoints, plus dataset caches. No automatic held-out evaluation
is included; evaluate saved models before committing to a long run or deployment.

### Smaller 100M preset

`--size 100m` initializes **100,291,200 random parameters**: 16 layers, 640 hidden
width, 1728 intermediate width, 10 attention heads and 2 KV heads. It reuses the
same tokenizer, datasets, 2048-token context, training code, and system-prompt chat.
No second script or extra dependencies.

Run training only on the separate BF16-capable GPU machine, from `TinyLLM1B/`:

```sh
export CUDA_VISIBLE_DEVICES=0
uv run --python 3.12 TinyLLM1B.py pretrain --size 100m --steps 30000 --output runs/base-100m
uv run --python 3.12 TinyLLM1B.py sft --model runs/base-100m --output runs/chat-100m

# Resume with the same total step budget. Size comes from the checkpoint.
uv run --python 3.12 TinyLLM1B.py pretrain --steps 30000 --output runs/base-100m --resume runs/base-100m/checkpoint-1000
```

30,000 steps process about 1.97 billion pretraining tokens. `--size` changes only
fresh model initialization, not the default step budget. SFT, resume, and chat
load the saved architecture automatically. This smaller model is for experiments;
do not expect the knowledge or instruction-following ability of a larger model.

After copying the trained model back, CPU chat uses the same command:

```sh
CUDA_VISIBLE_DEVICES="" uv run --python 3.12 TinyLLM1B.py chat --model runs/chat-100m --system "Answer briefly."
```

100M weights alone occupy about 0.4 GB in FP32; inference and training need extra
memory. GPU memory usage and output quality have not been measured.

### Chat with your trained weights

Copy `runs/chat` from the training machine first. CPU inference needs several GB
of free RAM and will be slow. Do not expect the 1B model to fit a 2 GB GPU:

```sh
CUDA_VISIBLE_DEVICES="" uv run --python 3.12 TinyLLM1B.py chat --model runs/chat --system "Answer briefly and honestly."
```

Use `/quit` to exit. Chat retains the system prompt and removes the oldest
exchanges when context fills. Oversized individual prompts are rejected.
System-prompt formatting does not guarantee instruction following or safety.

### Offline checks, no training

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_tinyllm1b.py
python3 TinyLLM1B.py --help
```

These checks cover parameter arithmetic, data packing, chat formatting, context
limits, and CLI guards. They do not load an ML framework, allocate a model, run
an optimizer, or contact Hugging Face. GPU execution and trained output quality
must be checked on the training machine.

Dataset licenses: FineWeb-Edu is ODC-By; smol-smoltalk is Apache-2.0. Review their
model/dataset cards and source-data terms before redistribution or deployment.
