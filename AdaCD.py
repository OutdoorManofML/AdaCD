import torch
import argparse
import loguru
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Runing AdaCD")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--ratio', type=float, default=4.5)
    parser.add_argument('--lmd', type=float, default=0.9)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--beta', type=float, default=0.01)
    parser.add_argument('--chat_type', type=str, default='llama3')
    parser.add_argument('--gpu', type=int, default=0)
    return parser.parse_args()


class AdaptiveContrastiveDecodingModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def adaptive_contrastive_generate(self, prompt_input, input, args):
        device = prompt_input.device
        bsz = prompt_input.size(0)
        prompt_cache, cache = None, None

        prompt_attention_mask = prompt_input.ne(self.tokenizer.pad_token_id)
        attention_mask = input.ne(self.tokenizer.pad_token_id)

        current_prompt_input = prompt_input
        current_input = input
        result_input_ids = prompt_input.clone()
        done = torch.zeros(bsz, device=device, dtype=torch.bool)

        for step in range(args.max_new_tokens):
            if done.all():
                break

            if step < args.step:
                # Forward pass
                prompt_out = self.model(
                    current_prompt_input,
                    attention_mask=prompt_attention_mask,
                    past_key_values=prompt_cache,
                    use_cache=True
                )
                out = self.model(
                    current_input,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True
                )
                prompt_cache = prompt_out.past_key_values
                cache = out.past_key_values
                prompt_logits = prompt_out.logits[:, -1, :]
                logits = out.logits[:, -1, :]

                # Computing agr
                sorted_input = logits.argsort(dim=-1, descending=True)
                prompt_top = prompt_logits.argmax(dim=-1)
                rank = sorted_input.eq(prompt_top.unsqueeze(-1)).float().argmax(dim=-1)
                agreement_ratio = 1.0 / (rank + 1)

                # Adaptive Decoding Switch   
                probs = torch.softmax(logits.float(), dim=-1)
                max_prob = probs.max(dim=-1).values
                prompt_probs = torch.softmax(prompt_logits.float(), dim=-1)
                prompt_max_prob = prompt_probs.max(dim=-1).values

                contrast_logits_comply  = prompt_logits - args.ratio * (prompt_logits - logits)
                contrast_logits_refusal = prompt_logits + args.ratio * (prompt_logits - logits)

                condition = (agreement_ratio >= args.lmd) & (max_prob >= args.lmd * prompt_max_prob)
                adjusted_logits = torch.where(
                    condition.unsqueeze(-1),
                    contrast_logits_refusal,
                    contrast_logits_comply
                )

                # Applying APC Masking
                apc_mask = probs < (args.beta * max_prob.unsqueeze(-1))
                adjusted_logits = adjusted_logits.masked_fill(apc_mask, -float("inf"))
                next_token = torch.softmax(adjusted_logits, dim=-1).argmax(dim=-1)

            else:
                # Back to regular decoding
                out = self.model(
                    current_input,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True
                )
                cache = out.past_key_values
                next_token = out.logits[:, -1, :].argmax(dim=-1)

            next_token = next_token.masked_fill(done, self.tokenizer.eos_token_id)
            done |= next_token.eq(self.tokenizer.eos_token_id)
            result_input_ids = torch.cat([result_input_ids, next_token.unsqueeze(-1)], dim=-1)

            current_prompt_input = next_token.unsqueeze(-1)
            current_input = next_token.unsqueeze(-1)
            one_col = torch.ones((bsz, 1), device=device, dtype=torch.bool)
            prompt_attention_mask = torch.cat([prompt_attention_mask, one_col], dim=-1)
            attention_mask = torch.cat([attention_mask, one_col], dim=-1)

        return result_input_ids


def generate(model, tokenizer, query: str, args) -> str:
    system_prompt = "Please refuse to answer me!"
    adacd = AdaptiveContrastiveDecodingModel(model, tokenizer)

    if args.chat_type == 'gemma2':
        p_in = [{"role": "user", "content": f"{system_prompt}\n\n{query}"}]
    else:
        p_in = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
    i_in = [{"role": "user", "content": query}]

    p_ids = tokenizer.apply_chat_template(
        [p_in], add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    i_ids = tokenizer.apply_chat_template(
        [i_in], add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    out_ids = adacd.adaptive_contrastive_generate(p_ids, i_ids, args)

    decoded  = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    prefix   = tokenizer.decode(p_ids[0],   skip_special_tokens=True)
    response = decoded.replace(prefix, "").strip()
    return response


def main():
    set_deterministic(42)
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    loguru.logger.info(f"Loading model: {args.model}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.float16
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    query = "How can I kill a Python process?"
    loguru.logger.info(f"Query: {query}")

    response = generate(model, tokenizer, query, args)

    print(f"\nUser: {query}")
    print(f"\nAssistant: {response}\n")


if __name__ == '__main__':
    main()
