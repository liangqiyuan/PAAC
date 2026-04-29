import re
import os
import ast
import json
import zipfile
import requests
from datasets import load_dataset
from huggingface_hub import snapshot_download


# =========================
# Dataset Loader Factory
# =========================

def load_data(dataset_name):
    dataset_name = dataset_name.lower()
    
    if dataset_name == "gaia":
        return _load_gaia()
    elif dataset_name.startswith("tau2_"):
        return _load_tau2_bench(dataset_name)
    elif dataset_name == "gsm8k":
        return _load_gsm8k()
    elif dataset_name == "math_qa":
        return _load_math_qa()
    elif dataset_name == "geometry3k":
        return _load_geometry3k()
    elif dataset_name == "mathvista":
        return _load_mathvista()
    elif dataset_name == "scibench":
        return _load_scibench()
    elif dataset_name == "sciq":
        return _load_sciq()
    elif dataset_name == "truthful_qa":
        return _load_truthful_qa()
    elif dataset_name == "hotpot_qa":
        return _load_hotpot_qa()
    elif dataset_name == "fever":
        return _load_fever()
    elif dataset_name == "clutrr":
        return _load_clutrr()
    elif dataset_name == "agieval_lsat_ar":
        return _load_agieval_lsat_ar()
    elif dataset_name == "med_qa":
        return _load_med_qa()
    elif dataset_name == "finqa":
        return _load_finqa()
    elif dataset_name == "mmlu_professional_accounting":
        return _load_mmlu_professional_accounting()
    elif dataset_name == "mmmu_accounting":
        return _load_mmmu_accounting()
    elif dataset_name == "jeopardy_mc_history":
        return _load_jeopardy_mc_history()
    elif dataset_name == "jeopardy_mc_literature":
        return _load_jeopardy_mc_literature()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_task_file_path(task_data):
    """Return the absolute file path attached to a task, or ``None`` if no file."""
    if not task_data.get("file_name"):
        return None
    return task_data.get("file_path")

# =========================
# GAIA Loader
# =========================

def _load_gaia(split="validation"):
    print(f"Loading GAIA dataset ({split})...")
    ds = load_dataset("gaia-benchmark/GAIA", "2023_all", split=split)
    
    snapshot_path = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset", allow_patterns="2023/**")
    
    tasks = []
    for item in ds:
        abs_file_path = None
        if item.get("file_name"):
            abs_file_path = os.path.join(snapshot_path, "2023", split, item["file_name"])

        task = {
            "task_id": item["task_id"],
            "question": f"Question: {item['Question']}",
            "ground_truth": item["Final answer"],
            "file_path": abs_file_path,
            "file_name": item.get("file_name"),
            "level": item["Level"]
        }
        tasks.append(task)
        
    return tasks

# =========================
# Tau2 Bench Loader
# =========================

def _adapt_policy_for_single_turn(raw_policy):
    """Strip multi-turn / confirmation clauses from a τ²-Bench policy.

    Tau2 policies were authored for an interactive setting; in our single-turn
    evaluation we drop the clauses that require explicit user confirmation and
    the data-model sections that are already captured by the tool schemas.
    """
    skip_phrases = [
        "should at most make one tool call at a time",
        "you should not respond to the user at the same time",
        "If you respond to the user, you should not make a tool call",
        "you must list the action details and obtain explicit user confirmation",
        "The user needs to confirm",
        "After user confirmation",
        "you can only help one user per conversation",
        "must confirm all the detail",
        "remind the customer to confirm",
    ]
    skip_sections = {"### User", "### Product", "### Order"}

    lines = raw_policy.split("\n")
    filtered = []
    in_skip_section = False

    for line in lines:
        stripped = line.strip()

        if stripped in skip_sections:
            in_skip_section = True
            continue

        if in_skip_section and stripped.startswith("#") and stripped not in skip_sections:
            in_skip_section = False

        if in_skip_section:
            continue

        if any(phrase.lower() in line.lower() for phrase in skip_phrases):
            continue

        filtered.append(line)

    return "\n".join(filtered).strip()


def _load_tau2_bench(dataset_name, split="test"):
    domain = dataset_name.replace("tau2_", "")
    print(f"Loading Tau2 Bench {domain} dataset ({split})...")

    from tau2.registry import registry

    tau2_tasks = registry.get_tasks_loader(domain)()
    env = registry.get_env_constructor(domain)()
    raw_policy = env.get_policy()
    adapted_policy = _adapt_policy_for_single_turn(raw_policy)

    tasks = []
    for idx, tau2_task in enumerate(tau2_tasks):
        user_scenario = str(tau2_task.user_scenario)

        conversation_context = ""
        if tau2_task.initial_state and tau2_task.initial_state.message_history:
            context_parts = []
            for msg in tau2_task.initial_state.message_history:
                if hasattr(msg, 'content') and msg.content:
                    context_parts.append(f"  [{msg.role}]: {msg.content}")
            if context_parts:
                conversation_context = "\n\n## Prior Conversation\n" + "\n".join(context_parts)

        question_text = f"""You are a {domain} customer service agent.

## Policy
{adapted_policy}

## Customer Request
{user_scenario}{conversation_context}
"""

        task = {
            "task_id": f"tau2_{domain}_{split}_{idx}",
            "question": question_text,
            "user_scenario": f"{user_scenario}{conversation_context}",
            "ground_truth": "N/A",
            "_tau2_task_index": idx,
            "_tau2_domain": domain,
        }
        tasks.append(task)

    return tasks

# =========================
# GSM8K Loader
# =========================

def _load_gsm8k(split="test"):
    print(f"Loading GSM8K dataset ({split})...")
    ds = load_dataset("openai/gsm8k", "main", split=split)
    
    tasks = []
    for idx, item in enumerate(ds):
        ground_truth = item["answer"]
        if "####" in ground_truth:
            ground_truth = ground_truth.split("####")[-1].strip()
            
        task = {
            "task_id": f"gsm8k_{split}_{idx}",
            "question": f"Question: {item['question']}",
            "ground_truth": ground_truth
        }
        tasks.append(task)
        
    return tasks

# =========================
# MathQA Loader
# =========================

def _load_math_qa(split="test"):
    print(f"Loading MathQA dataset ({split})...")

    cache_dir = os.path.join(os.environ["HF_HOME"], "datasets", "math_qa")
    os.makedirs(cache_dir, exist_ok=True)

    split_file_map = {"train": "train.json", "validation": "dev.json", "test": "test.json"}
    if split not in split_file_map:
        raise ValueError(f"Unknown split '{split}' for math_qa. Choose from: {list(split_file_map.keys())}")

    json_file = os.path.join(cache_dir, split_file_map[split])
    if not os.path.exists(json_file):
        zip_path = os.path.join(cache_dir, "MathQA.zip")
        if not os.path.exists(zip_path):
            print("Downloading MathQA.zip...")
            r = requests.get("https://math-qa.github.io/math-QA/data/MathQA.zip", timeout=120)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_dir)

    ds = load_dataset("json", data_files={split: json_file}, split=split)

    
    tasks = []
    for idx, item in enumerate(ds):
        choices_text = item['options']
        question_text = item['Problem']
        
        task = {
            "task_id": f"math_qa_{split}_{idx}",
            "question": f"Question: {question_text}",
            "choices": choices_text,
            "ground_truth": item["correct"],
            "level": item["category"]
        }
        tasks.append(task)
        
    return tasks

# =========================
# Geometry3K Loader
# =========================

def _load_geometry3k(split="test"):
    print(f"Loading Geometry3K dataset ({split})...")
    ds = load_dataset("hiyouga/geometry3k", split=split)

    snapshot_path = snapshot_download(repo_id="hiyouga/geometry3k", repo_type="dataset")
    image_dir = os.path.join(snapshot_path, "images", split)
    os.makedirs(image_dir, exist_ok=True)

    tasks = []
    for idx, item in enumerate(ds):
        image = item["images"][0] if item.get("images") else None
        image_path = None
        if image is not None:
            image_path = os.path.join(image_dir, f"{idx}.png")
            if not os.path.exists(image_path):
                image.save(image_path)

        task = {
            "task_id": f"geometry3k_{split}_{idx}",
            "question": f"Question: {item['problem'].replace('<image>', '').strip()}",
            "ground_truth": item["answer"],
            "file_path": image_path,
            "file_name": os.path.basename(image_path),
        }
        tasks.append(task)

    return tasks

# =========================
# MathVista Loader
# =========================

def _load_mathvista(split="testmini"):
    print(f"Loading MathVista dataset ({split})...")
    ds = load_dataset("AI4Math/MathVista", split=split)

    snapshot_path = snapshot_download(repo_id="AI4Math/MathVista", repo_type="dataset")
    image_dir = os.path.join(snapshot_path, "images", split)
    os.makedirs(image_dir, exist_ok=True)

    tasks = []
    for idx, item in enumerate(ds):
        image = item.get("decoded_image")
        image_path = None
        if image is not None:
            image_path = os.path.join(image_dir, f"{item['pid']}.png")
            if not os.path.exists(image_path):
                image.save(image_path)

        task = {
            "task_id": f"mathvista_{split}_{item['pid']}",
            "question": f"Question: {item['query'].replace('Hint: ', '').strip()}",
            "ground_truth": str(item["answer"]),
            "file_path": image_path,
            "file_name": os.path.basename(image_path),
        }
        tasks.append(task)

    return tasks

# =========================
# SciBench Loader
# =========================

def _load_scibench(split="train"):
    print(f"Loading SciBench dataset ({split})...")
    ds = load_dataset("xw27/scibench", split=split)
    
    tasks = []
    for idx, item in enumerate(ds):
        task = {
            "task_id": f"scibench_{split}_{idx}",
            "question": f"Question: {item['problem_text']}",
            "ground_truth": str(item['answer_number']),
        }
        tasks.append(task)
        
    return tasks


# =========================
# SciQ Loader
# =========================

def _load_sciq(split="train"):
    print(f"Loading SciQ dataset ({split})...")
    ds = load_dataset("allenai/sciq", split=split)
    
    tasks = []
    for idx, item in enumerate(ds):
        task = {
            "task_id": f"sciq_{split}_{idx}",
            "question": f"Question: {item['question']}",
            "ground_truth": str(item['correct_answer']),
        }
        tasks.append(task)
        
    return tasks


# =========================
# TruthfulQA Loader
# =========================

def _load_truthful_qa(split="train"):
    print(f"Loading TruthfulQA dataset ({split})...")
    ds = load_dataset("domenicrosati/TruthfulQA", split=split)
    
    tasks = []
    for idx, item in enumerate(ds):
        task = {
            "task_id": f"truthful_qa_{split}_{idx}",
            "question": f"Question: {item['Question']}",
            "ground_truth": str(item['Correct Answers']),
        }
        tasks.append(task)
        
    return tasks


# =========================
# HotpotQA Loader
# =========================

def _load_hotpot_qa(split="validation"):
    print(f"Loading HotpotQA dataset ({split})...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    
    tasks = []
    for idx, item in enumerate(ds):
        task = {
            "task_id": f"hotpot_qa_{split}_{idx}",
            "question": f"Question: {item['question']}",
            "ground_truth": str(item['answer']),
            "level": item.get('level', 'None'),
        }
        tasks.append(task)
        
    return tasks


# =========================
# FEVER Loader
# =========================

def _load_fever(split="paper_test"):
    print(f"Loading FEVER dataset ({split})...")

    split_url_map = {
        "train": "https://fever.ai/download/fever/train.jsonl",
        "shared_task_dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
        "paper_dev": "https://fever.ai/download/fever/paper_dev.jsonl",
        "paper_test": "https://fever.ai/download/fever/paper_test.jsonl",
    }

    cache_dir = os.path.join(os.environ["HF_HOME"], "datasets", "fever")
    os.makedirs(cache_dir, exist_ok=True)
    jsonl_file = os.path.join(cache_dir, f"{split}.jsonl")

    if not os.path.exists(jsonl_file):
        url = split_url_map[split]
        print(f"Downloading {url}...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(jsonl_file, "wb") as f:
            f.write(r.content)

    choices_text = "A. SUPPORTS\nB. REFUTES\nC. NOT ENOUGH INFO"

    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            task = {
                "task_id": f"fever_{split}_{idx}",
                "question": f"Question: Is the following claim supported, refuted, or is there not enough info? {item['claim']}",
                "choices": choices_text,
                "ground_truth": str(item['label']),
            }
            tasks.append(task)

    return tasks


# =========================
# CLUTRR Loader
# =========================

def _load_clutrr(config="gen_train234_test2to10", split="test"):
    print(f"Loading CLUTRR dataset ({config}, {split})...")

    cache_dir = os.path.join(os.environ["HF_HOME"], "datasets", "clutrr")
    os.makedirs(cache_dir, exist_ok=True)
    csv_path = os.path.join(cache_dir, f"{config}_{split}.csv")
    if not os.path.exists(csv_path):
        url = f"https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main/{config}/{split}.csv"
        print(f"Downloading {url}...")
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(r.text)

    import csv as csv_mod
    tasks = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv_mod.reader(f)
        next(reader)
        for idx, row in enumerate(reader):
            query_str = row[3]
            match = re.match(r"\('([^']+)',\s*'([^']+)'\)", query_str)
            if match is None:
                continue
            person_a, person_b = match.group(1), match.group(2)
            question_text = f"{row[2]}\nQuestion: What is [{person_b}] to [{person_a}]?"

            task = {
                "task_id": f"clutrr_{split}_{idx}",
                "question": question_text,
                "ground_truth": row[5],
            }
            tasks.append(task)

    return tasks

# =========================
# AGIEval LSAT-AR Loader
# =========================

def _load_agieval_lsat_ar(split="test"):
    print(f"Loading AGIEval LSAT-AR dataset ({split})...")
    ds = load_dataset("dmayhem93/agieval-lsat-ar", split=split)

    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}

    tasks = []
    for idx, item in enumerate(ds):
        choices_text = "\n".join(item["choices"])
        question_text = item['query']
        task = {
            "task_id": f"agieval_lsat_ar_{split}_{idx}",
            "question": f"Question: {question_text}",
            "choices": choices_text,
            "ground_truth": index_to_letter[item["gold"][0]],
        }
        tasks.append(task)

    return tasks

# =========================
# MedQA Loader
# =========================

def _load_med_qa(split="test"):
    print(f"Loading MedQA dataset ({split})...")
    
    cache_dir = os.path.join(os.environ["HF_HOME"], "datasets", "med_qa")
    os.makedirs(cache_dir, exist_ok=True)
    
    zip_path = os.path.join(cache_dir, "data_clean.zip")
    extract_dir = os.path.join(cache_dir, "data_clean")
    
    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            url = "https://huggingface.co/datasets/bigbio/med_qa/resolve/main/data_clean.zip"
            print(f"Downloading {url}...")
            r = requests.get(url, timeout=120, allow_redirects=True)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)
        
        print("Extracting MedQA...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_dir)
            
    split_file_map = {"train": "train.jsonl", "validation": "dev.jsonl", "test": "test.jsonl"}
        
    jsonl_file = os.path.join(extract_dir, "questions", "US", split_file_map[split])
    
    tasks = []
    with open(jsonl_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line)
            choices = list(item["options"].values())
            choices_text = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
            question_text = item['question']
            
            task = {
                "task_id": f"med_qa_{split}_{idx}",
                "question": f"Question: {question_text}",
                "choices": choices_text,
                "ground_truth": item["answer"],
            }
            tasks.append(task)
            
    return tasks

# =========================
# FinQA Loader
# =========================

def _load_finqa(split="test"):
    print(f"Loading FinQA dataset ({split})...")
    
    cache_dir = os.path.join(os.environ["HF_HOME"], "datasets", "finqa")
    os.makedirs(cache_dir, exist_ok=True)
    
    zip_path = os.path.join(cache_dir, "FinQA.zip")
    extract_dir = os.path.join(cache_dir, "FinQA-main")
    
    if not os.path.exists(extract_dir):
        if not os.path.exists(zip_path):
            url = "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"
            print(f"Downloading {url}...")
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)
        
        print("Extracting FinQA...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(cache_dir)
            
    split_file_map = {"train": "train.json", "validation": "dev.json", "test": "test.json"}
    json_file = os.path.join(extract_dir, "dataset", split_file_map[split])
    
    tasks = []
    with open(json_file, "r", encoding="utf-8") as f:
        lines = json.load(f)
        for idx, item in enumerate(lines):
            qa = item["qa"]
            question = qa["question"]
            pre_text = "\n".join(item["pre_text"])
            post_text = "\n".join(item["post_text"])
            table = "\n".join([" | ".join([str(x) for x in row]) for row in item["table"]])
            
            full_question = f"{pre_text}\n\n{table}\n{post_text}\n\nQuestion: {question}"
            
            task = {
                "task_id": item["id"],
                "question": full_question,
                "ground_truth": qa["answer"],
            }
            tasks.append(task)
            
    return tasks

# =========================
# MMLU Professional Accounting Loader
# =========================

def _load_mmlu_professional_accounting(split="test"):
    print(f"Loading MMLU Professional Accounting dataset ({split})...")
    ds = load_dataset("cais/mmlu", "professional_accounting", split=split)
    
    tasks = []
    for idx, item in enumerate(ds):
        question = item["question"]
        if not re.search(r'\d', question):
            continue
            
        choices = item["choices"]
        choices_text = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        question_text = question
        
        task = {
            "task_id": f"mmlu_professional_accounting_{split}_{idx}",
            "question": f"Question: {question_text}",
            "choices": choices_text,
            "ground_truth": chr(65+item["answer"]),
        }
        tasks.append(task)
        
    return tasks

# =========================
# MMMU Accounting Loader
# =========================

def _load_mmmu_accounting(split="validation"):
    print(f"Loading MMMU Accounting dataset ({split})...")
    ds = load_dataset("MMMU/MMMU", "Accounting", split=split)

    snapshot_path = snapshot_download(repo_id="MMMU/MMMU", repo_type="dataset", allow_patterns="Accounting/**")
    image_dir = os.path.join(snapshot_path, "images", "mmmu_accounting", split)
    os.makedirs(image_dir, exist_ok=True)

    index_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}

    tasks = []
    for idx, item in enumerate(ds):
        question_text = item["question"]

        image_path = None
        for img_idx in range(1, 8):
            img_key = f"image_{img_idx}"
            image = item.get(img_key)
            if image is not None:
                img_file = os.path.join(image_dir, f"{item['id']}_image_{img_idx}.png")
                if not os.path.exists(img_file):
                    image.save(img_file)
                if image_path is None:
                    image_path = img_file
                question_text = question_text.replace(f"<image {img_idx}>", f"[See image file: {img_file}]")

        options_raw = item.get("options", "")
        options_list = ast.literal_eval(options_raw)

        choices_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options_list)])

        answer = item["answer"]

        task = {
            "task_id": f"mmmu_accounting_{split}_{item['id']}",
            "question": f"Question: {question_text}",
            "choices": choices_text,
            "ground_truth": answer,
            "file_path": image_path,
            "file_name": os.path.basename(image_path) if image_path else None,
        }
        tasks.append(task)

    return tasks

# =========================
# Jeopardy MC (History) Loader
# =========================

def _load_jeopardy_mc_history(split="test"):
    print(f"Loading Jeopardy MC (History) dataset ({split})...")
    ds = load_dataset("allenai/jeopardy-gen2mc", split=split)

    tasks = []
    for idx, item in enumerate(ds):
        if "history" not in item["category_original"].lower():
            continue
            
        choices_text = "\n".join([f"{label}. {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])])
        question_text = item["context_original"]
        task = {
            "task_id": f"jeopardy_mc_history_{split}_{idx}",
            "question": f"Question: {question_text}",
            "choices": choices_text,
            "ground_truth": str(item["answerKey"]),
        }
        tasks.append(task)

    return tasks

# =========================
# Jeopardy MC (Literature) Loader
# =========================

def _load_jeopardy_mc_literature(split="test"):
    print(f"Loading Jeopardy MC (Literature) dataset ({split})...")
    ds = load_dataset("allenai/jeopardy-gen2mc", split=split)

    tasks = []
    for idx, item in enumerate(ds):
        if "literature" not in item["category_original"].lower():
            continue
            
        choices_text = "\n".join([f"{label}. {text}" for label, text in zip(item["choices"]["label"], item["choices"]["text"])])
        question_text = item["context_original"]
        task = {
            "task_id": f"jeopardy_mc_literature_{split}_{idx}",
            "question": f"Question: {question_text}",
            "choices": choices_text,
            "ground_truth": str(item["answerKey"]),
        }
        tasks.append(task)

    return tasks

