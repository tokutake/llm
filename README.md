# Small philosophy language model

This project pretrains a small GPT from scratch on four English philosophy books. The code runs on Intel Macs with Python 3.11 and PyTorch, using CPU or Metal (MPS) when available.

## What's here

- `philosophy_corpus.txt`: local training corpus (not included in this repository; see Data).
- `philosophy_tokenizer.json`: 1,024-token BPE tokenizer trained on that corpus.
- `philosophy_torch.pt`: 331,392-parameter model after 10,000 training steps.
- `train_torch.py`, `generate_torch.py`, `tokenizer.py`: training and generation code.
- `requirements-torch.txt`: Python dependencies.
- `philosophy_torch_output.txt`: sample generations.
- `archive/`: earlier character-level and single-book experiments.

The local corpus used for the included checkpoint contains Marcus Aurelius's *Meditations* (Gutenberg #2680), Epictetus's *Enchiridion* (#45109), Plato's *Apology* (#1656), and Aristotle's *Nicomachean Ethics* (#8438). It is about 1.19 million characters after removing the Gutenberg wrappers. The corpus text is intentionally excluded from this repository.

## Run it

Create the environment once:

```sh
python3.11 -m venv .venv311
.venv311/bin/python -m pip install -r requirements-torch.txt
```

Generate from the included model:

```sh
.venv311/bin/python generate_torch.py --prompt "Some things are in our control, and others are not." --tokens 100
```

Train a new model from scratch:

```sh
.venv311/bin/python train_torch.py --steps 5000 --out philosophy_stage1.pt
.venv311/bin/python train_torch.py --steps 5000 --lr 0.0005 --init philosophy_stage1.pt --out philosophy_torch.pt
```

Then generate with `philosophy_torch.pt`. The current model learns English word patterns but does not reliably produce coherent paragraphs; see `philosophy_torch_output.txt` for an honest sample.

Training and generation default to `--device auto`, which selects MPS when PyTorch reports it available and otherwise uses CPU. Choose explicitly with `--device mps` or `--device cpu`.

## Data

To train from scratch, obtain and prepare the source texts separately, then save the combined UTF-8 text as `philosophy_corpus.txt`. The included tokenizer and checkpoint are provided so generation works without the corpus.

## Sources and rights

The source editions are [Meditations](https://www.gutenberg.org/ebooks/2680), [Enchiridion](https://www.gutenberg.org/ebooks/45109), [Apology](https://www.gutenberg.org/ebooks/1656), and [Nicomachean Ethics](https://www.gutenberg.org/ebooks/8438). Project Gutenberg marks these editions public domain in the United States. Copyright status can differ by jurisdiction, and translations or edition-specific material may have separate rights. This repository does not include the source corpus and is not an official Project Gutenberg publication or endorsement. See [Project Gutenberg's license explanation](https://www.gutenberg.org/policy/license.html) and check applicable local law before redistributing source texts.

The included checkpoint is a small experimental model trained on the corpus listed above. The checkpoint is provided for research and experimentation; no claim is made that it is an official Project Gutenberg work. The source texts and their rights are separate from this project's code and model artifacts.
