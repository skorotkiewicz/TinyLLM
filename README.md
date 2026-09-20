# TinyLLM — the world's smallest language model family 🦡

The same bigram Markov chain, ported to three languages (original: JS by hand,
Rust + Odin: with love). Training corpus: "mama dad mama cat dad mama dog".

## Run

**Rust** (needs cargo):
```sh
python3 TinyLLM.py && node TinyLLM.js
cd rust && cargo run
```

**Odin** (needs odin compiler):
```sh
odin run odin/
```

## Ladder to god (next rungs)
- v1: uniform random chars        ← GPT's joke version
- v2: bigram Markov (THIS)        ← learns real transitions
- v3: trigram / n-gram            ← more context, spookier output
- v4: attention + training        ← congratulations, you're building an LLM
