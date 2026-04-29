import os
import sys
import math
import re
import random
import datetime
import time
import json
import requests
import subprocess
import threading
import copy

from google import genai
from google.genai import types

import tiktoken
import json_repair
import prompts


_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
if not _GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Export it before running, e.g. "
        "`export GEMINI_API_KEY=your_key_here`."
    )
gemini_client = genai.Client(api_key=_GEMINI_API_KEY)


# =========================
# Execution Environment
# =========================
def get_exec_globals():
    """Standard library globals exposed to the Python sandbox."""
    return {
        "math": math,
        "os": os,
        "sys": sys,
        "re": re,
        "random": random,
        "datetime": datetime,
        "time": time,
        "json": json,
        "requests": requests,
        "subprocess": subprocess,
    }


# =========================
# Text Truncation Helper
# =========================
def truncate_by_tokens(text, max_tokens=20000):
    if not isinstance(text, str):
        text = str(text)

    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    if len(tokens) > max_tokens:
        half = max_tokens // 2
        truncated_tokens = (
            tokens[:half]
            + enc.encode(f"\n\n...[TRUNCATED TO {max_tokens} TOKENS]...\n\n")
            + tokens[-half:]
        )
        text = enc.decode(truncated_tokens)
        print(f"[WARNING] Truncated input from {len(tokens)} to {max_tokens} tokens.")

    return text


# =========================
# History Sanitizer for Final Answer
# =========================
def process_history_for_final_answer(history, max_words=2000):
    if not history:
        return "[]"

    sanitized_history = []
    for i, item in enumerate(history):
        device_judgment = item.get("device_judgment", {})
        key_findings = device_judgment.get("key_findings", "")
        if isinstance(key_findings, list):
            key_findings = "\n".join(str(x) for x in key_findings)
        key_findings = truncate_by_tokens(key_findings, max_words)

        step_str = f"Step {item.get('step', i)}"
        step_str += f" Key Findings: {key_findings}"

        for log in item.get("device_execution", []):
            if isinstance(log, dict) and "action" in log and isinstance(log["action"], dict):
                if log["action"].get("tool") == "final_answer":
                    cloud_args = log["action"].get("args", {})
                    if isinstance(cloud_args, dict) and "answer" in cloud_args:
                        cloud_final_answer = cloud_args["answer"]
                    else:
                        cloud_final_answer = cloud_args

                    step_str += (
                        f"\n[REFERENCE ONLY] Cloud Agent Submission: {cloud_final_answer}"
                        "\n(Note: The above final answer is the raw output from the agent and "
                        "might be incorrectly formatted. You MUST extract the verified facts "
                        "from Key Findings instead of trusting it blindly.)"
                    )

        sanitized_history.append(step_str)

    return json.dumps(sanitized_history, indent=4, ensure_ascii=False)


# =========================
# Step Execution Formatting
# =========================
def build_execution_summaries(step_results, max_error_tokens=300, max_output_tokens=1500):
    summaries = []
    for i, sr in enumerate(step_results):
        action = sr.get("action", {})
        result = sr.get("result", {})

        entry = {
            "action": i + 1,
            "tool": action.get("tool", "?"),
            "args": action.get("args", {}),
        }

        if isinstance(result, dict) and "error" in result:
            entry["status"] = "failed"
            entry["error"] = truncate_by_tokens(str(result.get("error")), max_error_tokens)
        elif isinstance(result, dict):
            entry["status"] = "success"
            content = (
                result.get("content")
                or result.get("output")
                or result.get("results")
                or result.get("summary")
                or result.get("observation")
                or result.get("answer")
            )
            if content is not None and content != "":
                entry["output"] = truncate_by_tokens(str(content), max_output_tokens)
            else:
                entry["output"] = truncate_by_tokens(str(result), max_output_tokens)
        else:
            entry["status"] = "completed"
            entry["output"] = truncate_by_tokens(str(result), max_output_tokens)

        summaries.append(entry)

    return summaries


# =========================
# LLM Response Parsing
# =========================
def parse_llm_json_response(response_text):
    """Parse an LLM JSON response, returning a dict.

    Uses json_repair to tolerate truncated JSON, Markdown fences, and minor
    syntax errors. Lists are unwrapped to their first dict element. A non-dict
    response is wrapped under the key ``raw_response``.
    """
    if response_text is None or response_text == "":
        return {}

    response_text = response_text.strip()

    try:
        response = json_repair.loads(response_text)
    except Exception as e:
        raise Exception(f"Failed to decode JSON: {e}")

    if isinstance(response, list):
        for item in response:
            if isinstance(item, dict):
                return item
        if len(response) == 0:
            return {}

    if not isinstance(response, dict):
        return {"raw_response": response_text}

    return response


# =========================
# LLM Backends
# =========================
def call_vllm_api(contents, model, temperature=1.0, max_tokens=8192, port=8000):
    if isinstance(contents, list):
        truncated = []
        for item in contents:
            if isinstance(item, dict) and item.get("type") == "text":
                truncated.append({"type": "text", "text": truncate_by_tokens(item.get("text", ""))})
            else:
                truncated.append(item)
        messages = [{"role": "user", "content": truncated}]
    else:
        messages = [{"role": "user", "content": truncate_by_tokens(str(contents))}]

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    response = requests.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=600,
    )
    data = response.json()
    response_text = data["choices"][0]["message"]["content"]
    parsed = parse_llm_json_response(response_text)
    if isinstance(parsed, dict) and "usage" in data:
        parsed["token_usage"] = {
            "input_tokens": data["usage"]["prompt_tokens"],
            "output_tokens": data["usage"]["completion_tokens"],
        }
    return parsed


def call_gemini(contents, model, temperature=1.0):
    contents = truncate_by_tokens(contents)

    response = gemini_client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=temperature,
        ),
    )
    parsed = parse_llm_json_response(response.text)
    if isinstance(parsed, dict) and getattr(response, "usage_metadata", None):
        parsed["token_usage"] = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
        }
    return parsed


def call_gemini_vision(image_bytes, image_mime, prompt, temperature=1.0):
    image_part = types.Part.from_bytes(data=image_bytes, mime_type=image_mime)
    text_part = types.Part.from_text(text=prompt)

    response = gemini_client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=[types.Content(role="user", parts=[image_part, text_part])],
        config=types.GenerateContentConfig(temperature=temperature),
    )
    return response.text


def call_gemini_search(query):
    prompt = f"""You are a Google Search engine.
User Query: "{query}"

Please rely on Google Search to answer.
Based on the search results, return a JSON object with a single key "results", which is a list of search result objects.
Each object must have:
- "title": Title of the page/result.
- "link": The URL.
- "snippet": A summary of the content.

Output JSON:
{{
  "results": [
      {{
        "title": "...",
        "link": "...",
        "snippet": "..."
      }}
  ]
}}
"""

    grounding_tool = types.Tool(google_search=types.GoogleSearch())
    config = types.GenerateContentConfig(tools=[grounding_tool])

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=config,
    )

    return parse_llm_json_response(response.text)


def call_gemini_fallback_search(query):
    fallback_prompt = f"""You are simulating a Google Search engine.
User Query: "{query}"

Please simulate a Google Search based on your knowledge.
Based on your knowledge, return a JSON object with a single key "results", which is a list of search result objects.
Each object must have:
- "title": Title of the page/result.
- "snippet": A summary of the content.

Output JSON:
{{
  "results": [
      {{
        "title": "...",
        "snippet": "..."
      }}
  ]
}}
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=fallback_prompt,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        if response.text:
            parsed = parse_llm_json_response(response.text)
            if isinstance(parsed, dict) and "results" in parsed:
                return parsed
    except Exception as e:
        print(f"[Gemini Fallback Search] Error: {e}")

    return {"results": [], "error": "Gemini fallback search failed"}


# =========================
# Unified LLM Call with Optional Reflection
# =========================
def call_llm(args, prompt, role, temperature=1.0, max_tokens=8192, reflection_fn=None):
    if role == "cloud":
        model = args.cloud_model
    elif role == "device":
        model = args.device_model
    else:
        model = args.cloud_model

    use_gemini = "gemini" in model.lower()

    if use_gemini:
        result = call_gemini(prompt, model=model, temperature=temperature)
    else:
        port = getattr(args, "vllm_port", int(os.environ.get("VLLM_PORT", 8000)))
        result = call_vllm_api(prompt, model=model, temperature=temperature, max_tokens=max_tokens, port=port)

    if args.tier == "pro" and reflection_fn is not None:
        try:
            reflection_prompt = reflection_fn(result)

            if use_gemini:
                reflection_res = call_gemini(reflection_prompt, model=model, temperature=temperature)
            else:
                port = getattr(args, "vllm_port", int(os.environ.get("VLLM_PORT", 8000)))
                reflection_res = call_vllm_api(
                    reflection_prompt, model=model, temperature=temperature,
                    max_tokens=max_tokens, port=port,
                )

            if "is_correct" not in reflection_res:
                return result

            reflection_payload = reflection_res.copy()
            if reflection_res.get("is_correct") is False and "correction" in reflection_res:
                correction = reflection_res["correction"]
                if isinstance(correction, str):
                    correction = parse_llm_json_response(correction)

                final_res = copy.deepcopy(result)
                if isinstance(correction, dict):
                    for key, value in correction.items():
                        if value is not None and value != "" and value != []:
                            final_res[key] = value

                reflection_payload["correction"] = reflection_payload.pop("correction", None)
                reflection_payload["original_response"] = copy.deepcopy(result)
            else:
                final_res = result
                if reflection_res.get("is_correct") is True:
                    reflection_payload.pop("correction", None)

            final_res["reflection"] = reflection_payload

            if isinstance(final_res, dict):
                orig_usage = result.get("token_usage", {"input_tokens": 0, "output_tokens": 0})
                ref_usage = reflection_res.get("token_usage", {"input_tokens": 0, "output_tokens": 0})
                final_res["token_usage"] = {
                    "input_tokens": orig_usage.get("input_tokens", 0) + ref_usage.get("input_tokens", 0),
                    "output_tokens": orig_usage.get("output_tokens", 0) + ref_usage.get("output_tokens", 0),
                }

            return final_res
        except Exception:
            return result

    return result


# =========================
# Evaluation
# =========================
def evaluate_result(args, prompt, temperature=0.2):
    result = call_gemini(prompt, model="gemini-3-flash-preview", temperature=temperature)
    return result.get("is_correct", False), result.get("reasoning", "")


# =========================
# Final Answer Generation
# =========================
def generate_final_answer(args, question, history, sanitizer):
    history_str = process_history_for_final_answer(history, max_words=2000)
    history_str = sanitizer.desanitize(history_str)
    prompt = prompts.get_final_answer_extraction_prompt(question, history_str)
    result = call_llm(
        args, prompt, role="device",
        reflection_fn=lambda x: prompts.get_reflection_final_answer_prompt(x, question, history_str),
    )
    return result["answer"], history_str, result["token_usage"]


def process_final_answer(args, question, ground_truth, history, sanitizer, task_id, toolbox):
    if args.dataset.startswith("tau2_"):
        return _process_tau2_final_answer(args, history, task_id, toolbox)

    final_answer, history_str, token_usage = generate_final_answer(args, question, history, sanitizer=sanitizer)
    history.append({"final_answer_input": history_str, "token_usage": token_usage})
    prompt_eval = prompts.get_evaluation_prompt(question, ground_truth, final_answer)
    is_correct, _reasoning = evaluate_result(args, prompt_eval)
    return final_answer, is_correct, history


def _process_tau2_final_answer(args, history, task_id, toolbox):
    is_correct, db_match, user_db_match, assertion_match = toolbox.evaluate_tau2_task(task_id)

    if not is_correct:
        env_data = toolbox._get_or_create_env(task_id)
        task = env_data["task"]

        gt_assistant_actions = []
        gt_user_actions = []
        if task.evaluation_criteria and task.evaluation_criteria.actions:
            for a in task.evaluation_criteria.actions:
                if a.requestor == "assistant":
                    gt_assistant_actions.append(f"{a.name}: {a.arguments}")
                elif a.requestor == "user":
                    gt_user_actions.append(f"{a.name}: {a.arguments}")

        user_scenario = str(task.user_scenario)

        simplified_history = []
        for step in history:
            simplified_step = {"step": step.get("step")}
            if "device_execution" in step:
                simplified_step["device_execution"] = step["device_execution"]
            simplified_history.append(simplified_step)
        history_str = json.dumps(simplified_history, indent=2, ensure_ascii=False)

        if not db_match:
            gt_assistant_str = "\n".join(gt_assistant_actions) if gt_assistant_actions else "No specific expected actions."
            prompt_db = prompts.get_tau2_db_evaluation_prompt(user_scenario, gt_assistant_str, history_str)
            override_db, _ = evaluate_result(args, prompt_db)
            if override_db:
                db_match = True

        if not user_db_match:
            gt_user_str = "\n".join(gt_user_actions) if gt_user_actions else "No specific expected user device actions."
            prompt_user_db = prompts.get_tau2_user_db_evaluation_prompt(user_scenario, gt_user_str, history_str)
            override_user_db, _ = evaluate_result(args, prompt_user_db)
            if override_user_db:
                user_db_match = True

        is_correct = bool(db_match and user_db_match and assertion_match)

    final_answer = (
        f"[Tau2 Eval: Correct={is_correct}, DB_match={db_match}, "
        f"User_DB_match={user_db_match}, Assertion_match={assertion_match}]"
    )
    return final_answer, is_correct, history


# =========================
# Incremental Saver
# =========================
class IncrementalSaver:
    """Persists per-task traces incrementally so a run can resume after a crash."""

    def __init__(self, output_file):
        self.output_file = output_file
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        if os.path.exists(output_file):
            with open(output_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            self.traces = [t for t in loaded if "error" not in str(t.get("final_prediction", "")).lower()]
            dropped = len(loaded) - len(self.traces)
            self.completed_ids = {t["task_id"] for t in self.traces}
            msg = f"[Resume] Loaded {len(self.completed_ids)} completed tasks from {output_file}"
            if dropped:
                msg += f" ({dropped} errored tasks will be re-run)"
            print(msg)
        else:
            self.traces = []
            self.completed_ids = set()

    def filter_pending(self, ds):
        """Return ``(orig_index, item)`` pairs for tasks not yet completed."""
        return [(i, item) for i, item in enumerate(ds) if item["task_id"] not in self.completed_ids]

    def save(self, trace):
        """Append one trace and flush atomically (thread-safe)."""
        with self._lock:
            self.traces.append(trace)
            with open(self.output_file, "w", encoding="utf-8", errors="backslashreplace") as f:
                json.dump(self.traces, f, indent=4, ensure_ascii=False)
