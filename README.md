# Please refuse to answer me! Mitigating Over-Refusal in Large Language Models via Adaptive Contrastive Decoding

## 👉 Quick Start

**Step 0. Environment Setup.**

Depending on your CUDA version (currently configured for CUDA 12.4).

```bash
pip install -r requirements.txt
```

**Step 1. Run the AdaCD.**

```bash
python AdaCD.py --model /path/to/your/model
```

This runs AdaCD on a user query:

> *"How can I kill a Python process?"*

and prints the model's response to the terminal.

**Step 2. (Optional) Adjust Hyperparameters.**

```bash
python adacd_demo.py \
    --model /path/to/your/model \
    --ratio 4.5 \
    --lmd 0.9 \
    --step 10 \
    --beta 0.01 \
    --max_new_tokens 512 \
    --chat_type llama3 \
    --gpu 0
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | Local path or Hugging Face model ID |
| `--ratio` | `4.5` | Contrastive penalty coefficient α |
| `--lmd` | `0.9` | Agreement ratio threshold λ |
| `--step` | `10` | Number of steps to apply contrastive decoding |
| `--beta` | `0.01` | Adaptive plausibility constraint β |
| `--max_new_tokens` | `512` | Maximum number of generated tokens |
| `--chat_type` | `llama3` | Chat template type (`llama3`, `qwen3` or `gemma2`) |
| `--gpu` | `0` | GPU index to use |

## 🌟 Note

- All model paths should be either a local path or a Hugging Face model ID.
