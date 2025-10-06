import time
from typing import Dict, Any, List, Tuple, Optional
import torch

# ----------------------------
# Dataset → wording/format
# ----------------------------
def _dataset_spec(name: str) -> Tuple[str, str, str]:
    """
    Returns (text_format, text_noun, task_noun)
    """
    if name in ["Cora", "Citeseer"]:
        return "[New Title] : [New Abstract]\n", "article", "new academic articles"
    if name == "PubMed":
        return "Title: [New Title]\nAbstract: [New Abstract]", "article", "new academic articles"
    if name == "Child":
        return "Title: [New Title]\nBook Description: [New Description]", "book description", "new book descriptions"
    if name in ["Photo", "Computer"]:
        return "Review: [New Review]", "review", "reviews of products from Amazon"
    # sensible default
    return "Text: [New Text]", "text", "new texts"

# ----------------------------
# Build a single chat message list for one sample
# ----------------------------
def _messages_for_sample(
    *,
    method: str,
    dataset_name: str,
    text_format: str,
    text_noun: str,
    task_noun: str,
    text0: Optional[str],
    text1: Optional[str],
    category0: str,
    category1: Optional[str],
) -> List[Dict[str, str]]:
    sys_msg = (
        f"You are a helpful assistant that generates {task_noun} from {dataset_name}. "
        f"Each {text_noun} MUST be wrapped exactly in '<START>{text_format}<END>'. "
        f"Do not include anything before <START> or after <END>. No extra commentary."
    )

    # Zero-shot
    if method == "zero_shot":
        user = f"Give me one {text_noun} from {dataset_name} with the topic '{category0}'."
        return [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]

    # Few-shots (your current structure repeated)
    if method == "few_shots":
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Give me one {text_noun} from {dataset_name} with the topic '{category0}'."},
            {"role": "assistant", "content": f"<START>{text0}<END>"},
            {"role": "user", "content": f"Give me one {text_noun} from {dataset_name} with the topic '{category0}'."},
            {"role": "assistant", "content": f"<START>{text1}<END>"},
            {"role": "user", "content": f"Give me one {text_noun} from {dataset_name} with the topic '{category0}'."},
        ]

    # Method "O" (pair then similar)
    if method == "O":
        return [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": f"Give me the first {text_noun} from {dataset_name} with the topic '{category0}'."},
            {"role": "assistant", "content": f"<START>{text0}<END>"},
            {"role": "user", "content": f"Give me the second {text_noun} from {dataset_name} with the topic '{category0}'. It should be more similar to the first {text_noun}."},
        ]

    # Default: 3-step with contrast (your current else branch)
    return [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f"Give me the first {text_noun} from {dataset_name} with the topic '{category0}'."},
        {"role": "assistant", "content": f"<START>{text0}<END>"},
        {"role": "user", "content": f"Give me the second {text_noun} from {dataset_name} with the topic '{category1}'."},
        {"role": "assistant", "content": f"<START>{text1}<END>"},
        {"role": "user", "content": f"Give me the third {text_noun} from {dataset_name} with the topic '{category0}'. "
                                     f"It should be more similar to the first {text_noun} and less similar to the second {text_noun}."},
    ]

# ----------------------------
# Public: build ONE rendered prompt string (chat-template aware)
# ----------------------------
def form_prompt(text0, text1, category0, category1, args) -> str:
    """
    Returns a *rendered* prompt string ready to tokenize for HF chat models.
    Uses tokenizer.apply_chat_template if available; otherwise falls back to your old Llama-ish raw format.
    """
    text_format, text_noun, task_noun = _dataset_spec(args.dataset)
    msgs = _messages_for_sample(
        method=args.method,
        dataset_name=args.dataset,
        text_format=text_format,
        text_noun=text_noun,
        task_noun=task_noun,
        text0=text0,
        text1=text1,
        category0=category0,
        category1=category1,
    )

    # Preferred: render with chat template (Qwen/Mistral/Llama instruct)
    # This adds the correct tokens like [INST], ChatML, etc.
    if hasattr(args.tokenizer, "apply_chat_template") and args.tokenizer.chat_template is not None:
        return args.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    # Fallback: your previous raw Llama-ish prompt (kept very close to original)
    return form_llama_prompt_legacy(text0, text1, category0, category1, args, text_format, text_noun, task_noun)

# ----------------------------
# Legacy raw prompt (kept for non-chat models)
# ----------------------------
def form_llama_prompt_legacy(text0, text1, category0, category1, args, text_format, text_noun, task_noun) -> str:
    dataset_name = args.dataset
    header_sys = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
    sys_content = f"You are a helpful AI assistant for generating {task_noun} from {dataset_name}, where each {text_noun} are in the format '<START>{text_format}<END>'.<|eot_id|>\n"
    start_user = "<|start_header_id|>user<|end_header_id|>\n\n"
    end_turn = "<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

    if args.method == "zero_shot":
        return (
            header_sys + sys_content +
            start_user + f"Give me one {text_noun} from {dataset_name} with the topic '{category0}'." + end_turn
        )

    if args.method == "few_shots":
        return (
            header_sys + sys_content +
            start_user + f"Give me one {text_noun} from {dataset_name} with the topic '{category0}'." + end_turn +
            f"<START>{text0}<END>\n" +
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            f"Give me one {text_noun} from {dataset_name} with the topic '{category0}'." + end_turn +
            f"<START>{text1}<END>\n" +
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            f"Give me one {text_noun} from {dataset_name} with the topic '{category0}'." + end_turn
        )

    if args.method == "O":
        return (
            header_sys + sys_content +
            start_user + f"Give me the first {text_noun} from {dataset_name} with the topic '{category0}'." + end_turn +
            f"<START>{text0}<END>\n" +
            "<|start_header_id|>user<|end_header_id|>\n\n" +
            f"Give me the second {text_noun} from {dataset_name} with the topic '{category0}'. It should be more similar to the first {text_noun}." + end_turn
        )

    # default
    return (
        header_sys + sys_content +
        start_user + f"Give me the first {text_noun} from {dataset_name} with the topic '{category0}'." + end_turn +
        f"<START>{text0}<END>\n" +
        "<|start_header_id|>user<|end_header_id|>\n\n" +
        f"Give me the second {text_noun} from {dataset_name} with the topic '{category1}'." + end_turn +
        f"<START>{text1}<END>\n" +
        "<|start_header_id|>user<|end_header_id|>\n\n" +
        f"Give me the third {text_noun} from {dataset_name} with the topic '{category0}'. It should be more similar to the first {text_noun} and less similar to the second {text_noun}." + end_turn
    )

# ----------------------------
# Batch processing (chat-template aware)
# ----------------------------
def process_prompt_list(prompt_list: List[str], args, batch_size: int):
    """
    prompt_list: list of *rendered* prompt strings returned by form_prompt (or any raw strings).
    """
    answer_list: List[str] = []
    output_list: List[Dict[str, Any]] = []
    total_time = 0.0
    cnt = 0

    i = 0
    while i < len(prompt_list):
        batch_data = prompt_list[i : min(i + batch_size, len(prompt_list))]

        start_time = time.time()
        answers, outputs = get_batched_response(batch_data, args)
        end_time = time.time()

        exec_t = end_time - start_time
        print("LLM Inference Time: ", exec_t)
        total_time += exec_t
        cnt += 1

        answer_list.extend(answers)
        output_list.extend(outputs)
        i += batch_size

    if cnt > 0:
        print(f"LLM Average Inference time with a batch size of {batch_size} is {total_time/cnt:.3f}s")

    return answer_list, output_list

def _extract_answer(generated_text: str) -> str:
    """
    Extract text within <START> ... <END>. Robust to missing markers.
    """
    if "<START>" in generated_text and "<END>" in generated_text:
        ans = generated_text.split("<START>", 1)[-1]
        ans = ans.split("<END>", 1)[0]
        return ans.strip()
    # fallback: whole string (trim)
    return " ".join(generated_text.strip().split())

def get_batched_response(batch_data: List[str], args):
    """
    batch_data: list of *rendered* chat prompts (strings). If you want to pass message dicts instead,
    render them first with tokenizer.apply_chat_template(..., tokenize=False, add_generation_prompt=True).
    """
    answer_list: List[str] = []
    output_list: List[Dict[str, Any]] = []

    # Safe pad/eos defaults
    if getattr(args.tokenizer, "pad_token", None) is None:
        args.tokenizer.pad_token = getattr(args.tokenizer, "eos_token", None)

    # Tokenize + pad as a batch
    # model_inputs = args.tokenizer(
    #     batch_data, return_tensors="pt", padding=True, truncation=False
    # ).to(args.device)
    enc = args.tokenizer(
        batch_data,
        return_tensors="pt",
        padding=True,
        truncation=False
    ).to(args.device)

    # per-sample input lengths (handles padding correctly)
    input_lens = enc["attention_mask"].sum(dim=1).tolist()

    # Generation defaults — feel free to expose as args.*
    gen_kwargs = dict(
        max_new_tokens=getattr(args, "llm_max_new_tokens", 300),
        do_sample=True,
        temperature=getattr(args, "llm_temperature", 0.7),
        top_p=getattr(args, "llm_top_p", 0.95),
        repetition_penalty=getattr(args, "llm_repetition_penalty", 1.0),
        pad_token_id=args.tokenizer.pad_token_id,
        eos_token_id=getattr(args.tokenizer, "eos_token_id", None),
    )
    # (optional) determinism for caching
    if hasattr(args, "llm_seed"):
        torch.manual_seed(args.llm_seed)

    with torch.inference_mode():
        out = args.llm_model.generate(**enc, **gen_kwargs)

    generated_text_list = []
    for seq, in_len in zip(out, input_lens):
        tail_ids = seq[int(in_len):]
        text = args.tokenizer.decode(tail_ids, skip_special_tokens=True)
        generated_text_list.append(text)

    for full_text in generated_text_list:
        answer = _extract_answer(full_text)
        output_list.append({"generated_text": full_text, "answer": answer})
        answer_list.append(answer)

    # generated = args.llm_model.generate(**model_inputs, **gen_kwargs)

    # # Only decode the newly generated tail per sample
    # input_len = model_inputs["input_ids"].shape[1]
    # tails = [seq[input_len:] for seq in generated]

    # generated_text_list = [
    #     args.tokenizer.decode(t, skip_special_tokens=True) for t in tails
    # ]

    # for full_text in generated_text_list:
    #     answer = _extract_answer(full_text)
    #     output_list.append({"generated_text": full_text, "answer": answer})
    #     answer_list.append(answer)

    return answer_list, output_list
