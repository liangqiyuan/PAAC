import re
import json
import traceback
import argparse
import concurrent.futures
import shutil
from tools import ToolBox, Tau2ToolBox
import multiprocessing
import sys
import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import warnings
warnings.filterwarnings("ignore")

from utils import (
    call_llm, parse_llm_json_response, truncate_by_tokens,
    evaluate_result, process_final_answer,
    IncrementalSaver, build_execution_summaries,
)
import prompts
import data_loader
import config

# =========================
# Privacy Sanitizer
# =========================
class PrivacySanitizer:
    def __init__(self, privacy_level, dataset_name):
        self.privacy_level = privacy_level
        self.sensitive_registry = {}
        self.reverse_registry = {}
        self._prefix_value_to_token = {}

        dataset_config = config.get_dataset_config(dataset_name)
        self.privacy_definitions = dataset_config["privacy_options"]
        self.checked_categories = dataset_config["privacy_levels"][privacy_level]
        self.step_token_usage = {"input_tokens": 0, "output_tokens": 0}

        self.token_counters = {}

    def _get_next_token(self, prefix):
        if prefix not in self.token_counters:
            self.token_counters[prefix] = 0
        self.token_counters[prefix] += 1
        return f"{prefix}_{self.token_counters[prefix]}"
    
    def _find_existing_token(self, prefix, real_value):
        key = (prefix, str(real_value))
        return self._prefix_value_to_token.get(key)
    
    def _register_token(self, prefix, real_value, canonical_token):
        self.sensitive_registry[canonical_token] = real_value
        self.reverse_registry[str(real_value)] = canonical_token
        self._prefix_value_to_token[(prefix, str(real_value))] = canonical_token

    def update_mapping(self, mapping):
        if isinstance(mapping, str):
            mapping = parse_llm_json_response(mapping)
            
        if not isinstance(mapping, dict):
            return

        for llm_key, real_value in mapping.items():
            if not str(real_value).strip():
                continue
            prefix = llm_key
            match = re.match(r"^(.*)_\d+$", llm_key)
            if match:
                prefix = match.group(1)
            
            existing = self._find_existing_token(prefix, real_value)
            if existing:
                continue
            
            new_token = self._get_next_token(prefix)
            self._register_token(prefix, real_value, new_token)
    
    def get_mapping(self):
        return self.sensitive_registry.copy()
    
    # =====================
    # Core Sanitize/Desanitize
    # =====================
    
    def _safe_replace(self, text, old, new):
        start_bound = r"(?<![\w])" if re.match(r'^\w', old) else ""
        end_bound = r"(?![\w])" if re.search(r'\w$', old) else ""
        pattern = start_bound + re.escape(old) + end_bound
        return re.sub(pattern, lambda _: new, text)
    
    def sanitize(self, text):
        if self.privacy_level == 0:
            return text
            
        sorted_items = sorted(
            self.sensitive_registry.items(),
            key=lambda x: len(str(x[1])),
            reverse=True
        )
        
        for placeholder, real_value in sorted_items:
            val_str = str(real_value)
            if val_str and val_str in text:
                text = self._safe_replace(text, val_str, placeholder)
        
        return text
    
    def get_real_val(self, ph, raw_val, is_code=False):
        v_str = str(raw_val)
        if is_code:
            word_map = {
                'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
                'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
                'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
                'eighteen': '18', 'nineteen': '19', 'twenty': '20',
            }
            v_lower = v_str.lower().strip()
            if v_lower in word_map:
                return word_map[v_lower]

            if any(c.isdigit() for c in v_str):
                return v_str.replace(',', '')

        return v_str

    def desanitize(self, text, is_code=False):
        if not isinstance(text, str):
            return text

        if text in self.sensitive_registry:
            return self.get_real_val(text, self.sensitive_registry[text], is_code=is_code)
        
        clean_text = text.strip("{}")
        if clean_text in self.sensitive_registry:
            return self.get_real_val(clean_text, self.sensitive_registry[clean_text], is_code=is_code)
        
        sorted_keys = sorted(self.sensitive_registry.keys(), key=len, reverse=True)

        for placeholder in sorted_keys:
            real_value = self.get_real_val(placeholder, self.sensitive_registry[placeholder], is_code=is_code)
            
            if placeholder.startswith("{") and placeholder.endswith("}"):
                text = text.replace(placeholder, real_value)
            else:
                text = text.replace("{" + placeholder + "}", real_value)
                text = text.replace(placeholder, real_value)
        
        return text
    
    def desanitize_data(self, data, is_code=False):
        if isinstance(data, str):
            return self.desanitize(data, is_code)
        elif isinstance(data, list):
            return [self.desanitize_data(item, is_code) for item in data]
        elif isinstance(data, dict):
            return {k: self.desanitize_data(v, is_code) for k, v in data.items()}
        else:
            return data
    
    # =====================
    # Cloud Agent I/O Processing
    # =====================

    def prepare_actions_for_execution(self, actions):
        desanitized_actions = []
        for action in actions:
            action_copy = action.copy() if isinstance(action, dict) else action
            if isinstance(action_copy, dict) and "args" in action_copy:
                is_code = (action_copy.get("tool") == "python_exec")
                action_copy["args"] = self.desanitize_data(action_copy.get("args", {}), is_code=is_code)
            desanitized_actions.append(action_copy)
        
        return desanitized_actions
    
    # =====================
    # Data Retrieval Sanitization
    # =====================

    def data_retrieval_extract(self, data, prefix="VALUE"):
        cats = set(self.checked_categories)

        if isinstance(data, dict):
            for k, v in data.items():
                k_lower = k.lower()
                pfx = config.KEY_TO_PREFIX.get(k_lower, prefix)
                if k_lower.endswith("_id") and k_lower not in config.KEY_TO_PREFIX:
                    pfx = k.upper()

                cat = config.KEY_TO_CATEGORY.get(k_lower)

                if cat in cats:
                    self.extract_all(v, pfx)
                elif isinstance(v, (dict, list)):
                    self.data_retrieval_extract(v, pfx)

        elif isinstance(data, list):
            for item in data:
                self.data_retrieval_extract(item, prefix)

    def extract_all(self, data, prefix="VALUE"):
        if isinstance(data, dict):
            for k, v in data.items():
                k_lower = str(k).lower()
                pfx = config.KEY_TO_PREFIX.get(k_lower)
                if not pfx:
                    pfx = str(k).upper()
                self.extract_all(v, pfx)
        elif isinstance(data, list):
            for item in data:
                self.extract_all(item, prefix)
        elif isinstance(data, (str, int, float)):
            val_str = str(data).strip()
            if not val_str or str(data).lower() in {"true", "false", "none", "null"}:
                return
            if len(val_str) < 2 and not ("ID" in prefix or "PRICE" in prefix):
                return
            if val_str.lower() in {"pending", "delivered", "shipped", "cancelled", "success", "error", "processing", "active", "inactive"}:
                return
            if val_str not in self.reverse_registry:
                new_token = self._get_next_token(prefix)
                self._register_token(prefix, val_str, new_token)
    
    # =====================
    # LLM-based Sanitization
    # =====================
    
    def call_sanitizer_llm(self, args, content):
        prompt = prompts.get_sanitizer_prompt(content, self.checked_categories, self.privacy_definitions)
        response = call_llm(args, prompt, role="local", reflection_fn=lambda x: prompts.get_reflection_sanitizer_prompt(x, content, self.checked_categories, self.privacy_definitions))
        if response is None:
            response = {}
            
        self.step_token_usage["input_tokens"] += response["token_usage"]["input_tokens"]
        self.step_token_usage["output_tokens"] += response["token_usage"]["output_tokens"]
        
        new_mappings = response.get("extracted_mapping", {})
        if not isinstance(new_mappings, dict):
            print(f"[WARN] Sanitizer LLM returned non-dict mapping: {new_mappings}")
            new_mappings = {}
        
        llm_sanitized_text = response.get("processed_content", "")
            
        return self._repair_llm_output(content, llm_sanitized_text, new_mappings)

    def _repair_llm_output(self, content, llm_sanitized_text, new_mappings):
        content_str = str(content)
        if content_str.startswith("Question: ") and not llm_sanitized_text.startswith("Question: "):
            llm_sanitized_text = "Question: " + llm_sanitized_text

        fixed_mappings = {}
        for k, v in new_mappings.items():
            if not str(v).strip():
                fixed_mappings[k] = v
                continue
                
            val_str = str(v)
            
            is_present = False
            if val_str.isdigit():
                num_boundary_behind = r"(?<!\d)(?<!\d[.,])"
                num_boundary_ahead = r"(?![.,]?\d)"
                is_present = bool(re.search(num_boundary_behind + re.escape(val_str) + num_boundary_ahead, content_str))
            else:
                is_present = val_str in content_str

            if not is_present:
                norm_val = self.get_real_val(k, val_str, is_code=True)
                norm_present = False
                if norm_val.isdigit():
                    num_boundary_behind = r"(?<!\d)(?<!\d[.,])"
                    num_boundary_ahead = r"(?![.,]?\d)"
                    norm_present = bool(re.search(num_boundary_behind + re.escape(norm_val) + num_boundary_ahead, content_str))
                else:
                    norm_present = norm_val in content_str

                if norm_present:
                    val_str = norm_val
                else:
                    lower_idx = content_str.lower().find(val_str.lower())
                    if not val_str.isdigit() and lower_idx != -1:
                        val_str = content_str[lower_idx:lower_idx+len(val_str)]
                    else:
                        rev_word_map = {
                            '0': ['zero'], '1': ['one', 'once'], '2': ['two', 'twice', 'half', 'double'], 
                            '3': ['three', 'thrice', 'triple', 'third'], '4': ['four', 'quarter'],
                            '5': ['five', 'fifth'], '6': ['six', 'sixth'], '7': ['seven'], '8': ['eight'], '9': ['nine'],
                            '10': ['ten'], '11': ['eleven'], '12': ['twelve', 'dozen'], '13': ['thirteen'],
                            '14': ['fourteen'], '15': ['fifteen'], '16': ['sixteen'], '17': ['seventeen'],
                            '18': ['eighteen'], '19': ['nineteen'], '20': ['twenty']
                        }
                        if val_str in rev_word_map:
                            found_w = None
                            for w in rev_word_map[val_str]:
                                if w in content_str.lower():
                                    found_w = w
                                    break
                            if found_w:
                                idx_w = content_str.lower().find(found_w)
                                val_str = content_str[idx_w:idx_w+len(found_w)]
                        elif val_str.isdigit() and len(val_str) > 3:
                            with_comma = f"{int(val_str):,}"
                            if with_comma in content_str:
                                val_str = with_comma

            fixed_mappings[k] = val_str

            # Strip duplicated unit tokens that the LLM left adjacent to the placeholder
            # (e.g. value "4 hours" mapped against text "DURATION hours").
            tokens = val_str.split()
            if len(tokens) > 1:
                for i in range(1, len(tokens)):
                    tail = " ".join(tokens[i:])
                    if f"{k} {tail}" in llm_sanitized_text:
                        llm_sanitized_text = llm_sanitized_text.replace(f"{k} {tail}", k)
                    elif f"{k}{tail}" in llm_sanitized_text:
                        llm_sanitized_text = llm_sanitized_text.replace(f"{k}{tail}", k)
                for i in range(1, len(tokens)):
                    head = " ".join(tokens[:-i])
                    if f"{head} {k}" in llm_sanitized_text:
                        llm_sanitized_text = llm_sanitized_text.replace(f"{head} {k}", k)
                    elif f"{head}{k}" in llm_sanitized_text:
                        llm_sanitized_text = llm_sanitized_text.replace(f"{head}{k}", k)

            # Restore currency / percent symbols that the LLM may have stripped.
            if val_str.isdigit():
                num_boundary_behind = r"(?<!\d)(?<!\d[.,])"
                num_boundary_ahead = r"(?![.,]?\d)"
                val_exact_regex = num_boundary_behind + re.escape(val_str) + num_boundary_ahead
                dollar_pattern = r"\$" + re.escape(val_str) + num_boundary_ahead
                pct_pattern = num_boundary_behind + re.escape(val_str) + r"%"
            else:
                val_exact_regex = re.escape(val_str)
                dollar_pattern = r"\$" + re.escape(val_str)
                pct_pattern = re.escape(val_str) + r"%"

            matches_dollar = len(re.findall(dollar_pattern, content_str))
            matches_exact = len(re.findall(val_exact_regex, content_str)) if val_str.isdigit() else content_str.count(val_str)
            
            is_money_token = any(w in k.upper() for w in ["MONEY", "PRICE", "COST", "DOLLAR"])
            should_have_dollar = matches_dollar > 0 and (is_money_token or matches_dollar == matches_exact)
            
            if should_have_dollar:
                if f"${k}" not in llm_sanitized_text and k in llm_sanitized_text:
                    llm_sanitized_text = re.sub(r"(?<!\$)\b" + re.escape(k) + r"\b", f"${k}", llm_sanitized_text)
            else:
                if f"${k}" in llm_sanitized_text:
                    llm_sanitized_text = llm_sanitized_text.replace(f"${k}", k)
                    
            matches_pct = len(re.findall(pct_pattern, content_str))
            
            is_pct_token = any(w in k.upper() for w in ["PERCENT", "RATIO"])
            should_have_pct = matches_pct > 0 and (is_pct_token or matches_pct == matches_exact)
            
            if should_have_pct:
                if f"{k}%" not in llm_sanitized_text and k in llm_sanitized_text:
                    llm_sanitized_text = re.sub(r"\b" + re.escape(k) + r"\b(?!%)", f"{k}%", llm_sanitized_text)
            else:
                if f"{k}%" in llm_sanitized_text:
                    llm_sanitized_text = llm_sanitized_text.replace(f"{k}%", k)

        # Down-case spelled-out numbers when the LLM upper-cased them without tokenising.
        for raw_word in ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN"]:
            if raw_word.lower() in content_str and raw_word not in content_str:
                llm_sanitized_text = re.sub(r"\b" + raw_word + r"\b", raw_word.lower(), llm_sanitized_text)
                        
        return fixed_mappings, llm_sanitized_text

    def _verify_alignment(self, content, llm_sanitized_text, new_mappings):
        recon_text = llm_sanitized_text
        for k, v in new_mappings.items():
            if not str(v).strip():
                continue
            recon_text = recon_text.replace(f"{{{k}}}", str(v))
            recon_text = recon_text.replace(f"[{k}]", str(v))
            recon_text = recon_text.replace(f"<{k}>", str(v))
            recon_text = recon_text.replace(k, str(v))

        is_perfect = (recon_text.strip() == str(content).strip())

        # Reject the trivial case where the LLM echoed the input verbatim while
        # claiming to have produced replacements.
        if is_perfect and new_mappings and llm_sanitized_text.strip() == str(content).strip():
            return False

        return is_perfect

    def _apply_mappings(self, content, llm_sanitized_text, new_mappings, is_perfect):
        final_text = llm_sanitized_text if is_perfect else str(content)

        sorted_llm_items = sorted(new_mappings.items(), key=lambda x: len(str(x[1])), reverse=True)
        
        for llm_key, real_value in sorted_llm_items:
            val_str = str(real_value)
            
            if not val_str.strip() or val_str not in content:
                continue
                
            prefix = llm_key
            match = re.match(r"^(.*)_\d+$", llm_key)
            if match:
                prefix = match.group(1)
            
            existing_token = self._find_existing_token(prefix, real_value)
            if existing_token:
                canonical_token = existing_token
            else:
                canonical_token = self._get_next_token(prefix)
                self._register_token(prefix, real_value, canonical_token)
            
            if is_perfect:
                final_text = final_text.replace(f"{{{llm_key}}}", canonical_token)
                final_text = final_text.replace(f"[{llm_key}]", canonical_token)
                final_text = final_text.replace(f"<{llm_key}>", canonical_token)
                final_text = final_text.replace(llm_key, canonical_token)
            else:
                final_text = self._safe_replace(final_text, val_str, canonical_token)

        return final_text
    
    def generate_sanitized_content(self, args, content):
        if not self.checked_categories or not str(content).strip():
            return content
            
        new_mappings, llm_sanitized_text = self.call_sanitizer_llm(args, content)

        is_perfect = self._verify_alignment(content, llm_sanitized_text, new_mappings)

        return self._apply_mappings(content, llm_sanitized_text, new_mappings, is_perfect)


# =========================
# Task Context
# =========================
class TaskContext:
    def __init__(self, task_data, privacy_level=0):
        self.task_id = task_data["task_id"]
        self.level = task_data.get("level", "None")
        self.ground_truth = task_data["ground_truth"]
        self.file_name = task_data.get("file_name")
        self.privacy_level = privacy_level
        self.choices = task_data.get("choices", "")

        self.original_question = task_data["question"]
        self.sanitized_question = self.original_question

        self.private_mapping = {}

        self.step_history = []
        
    def update_sanitized_question(self, sanitized_question, new_mappings=None):
        self.sanitized_question = sanitized_question
        if new_mappings:
            self.private_mapping.update(new_mappings)
    
    def add_step(self, step_num, cloud_output=None, cloud_output_sanitized=None,
                local_execution=None, local_execution_sanitized=None,
                local_judgment=None, local_judgment_sanitized=None):
        step_record = {
            "step": step_num,
            "cloud_output": cloud_output,
            "cloud_output_sanitized": cloud_output_sanitized,
            "local_execution": local_execution,
            "local_execution_sanitized": local_execution_sanitized,
            "local_judgment": local_judgment,
            "local_judgment_sanitized": local_judgment_sanitized,
        }
        self.step_history.append(step_record)
        return step_record
    
    def get_last_step(self):
        return self.step_history[-1] if self.step_history else None
    
    def update_last_step(self, **kwargs):
        if self.step_history:
            self.step_history[-1].update(kwargs)
    
    def get_real_history(self):
        real_history = []
        for step in self.step_history:
            entry = {
                "step": step["step"],
                "cloud_output": step.get("cloud_output"),
                "local_execution": step.get("local_execution"),
                "local_judgment": step.get("local_judgment"),
            }
            real_history.append(entry)
        return real_history
    
    def get_sanitized_findings_history(self):
        sanitized_findings_history = []
        for step in self.step_history:
            entry = {
                "step": step["step"],
                "sanitized_findings": step.get("local_judgment_sanitized", {}).get("key_findings", ""),
            }
            sanitized_findings_history.append(entry)
        return sanitized_findings_history


# =========================
# Strategies
# =========================

class AgentStrategy:
    def __init__(self, args, task_data):
        self.args = args
        self.task_data = task_data

        self.ctx = TaskContext(task_data, privacy_level=args.privacy_level)

        self.task_id = self.ctx.task_id
        self.level = self.ctx.level
        self.ground_truth = self.ctx.ground_truth
        self.file_name = self.ctx.file_name

        self.original_question = self.ctx.original_question
        self.question = self.ctx.sanitized_question

        self.local_file = data_loader.get_task_file_path(task_data)

        self.scratch_dir = os.path.abspath(f"scratch/{args.run_name}/{self.task_id}")
        os.makedirs(self.scratch_dir, exist_ok=True)

        if self.local_file and os.path.exists(self.local_file):
            file_basename = os.path.basename(self.local_file)
            scratch_file_path = os.path.join(self.scratch_dir, file_basename)
            shutil.copy2(self.local_file, scratch_file_path)
            self.local_file = scratch_file_path

        self.sanitizer = PrivacySanitizer(privacy_level=args.privacy_level, dataset_name=args.dataset)
        self.sanitized_findings_history = []
        self.real_history = []

        self._cloud_extra_instructions = self._build_cloud_extra_instructions()
        self._judge_extra_instructions = ""

        if args.dataset.startswith("tau2_"):
            self.tools_list = list(toolbox.registry)
        else:
            dataset_config = config.get_dataset_config(args.dataset)
            included_tools = dataset_config["tools"]
            self.tools_list = [t for t in toolbox.registry if t["name"] in included_tools]

    def _build_cloud_extra_instructions(self):
        if self.args.dataset.startswith("tau2_"):
            return prompts.get_tau2_cloud_instructions(self.args.privacy_level)
        elif self.args.privacy_level > 0:
            return prompts.get_cloud_common_instructions(self.args.privacy_level)
        return ""

    def sanitize_question(self):
        if "user_scenario" in self.task_data:
            user_scenario = self.task_data["user_scenario"]
            sanitized_scenario = self.local_sanitize(user_scenario)
            self.question = self.original_question.replace(user_scenario, sanitized_scenario)
        else:
            self.question = self.local_sanitize(self.original_question)

        self.ctx.update_sanitized_question(self.question)

    def _format_current_step(self, current_step):
        executions = build_execution_summaries(current_step.get("local_execution", []))
        formatted = {
            "step": current_step.get("step", 0),
            "executions": executions,
        }
        return json.dumps(formatted, indent=2, ensure_ascii=False)

    def _is_tau2_user_data_retrieval(self, step_results):
        if not self.args.dataset.startswith("tau2_") or not step_results:
            return False
        for log in step_results:
            tool = log["action"].get("tool", "")
            if not (tool.startswith("get_") or tool.startswith("find_") or tool.startswith("search_") or tool.startswith("list_")):
                return False
        return True
        
    def tau2_local_judgment(self, current_step):
        feedback_parts = []
        key_findings = []

        for i, log in enumerate(current_step["local_execution"]):
            tool = log["action"]["tool"]
            result = log["result"]
            
            if result["success"]:
                feedback_parts.append(f"Action {i+1} ({tool}) succeeded.")
            else:
                feedback_parts.append(f"Action {i+1} ({tool}) failed: {result['error']}")
            
            if not result["success"]:
                key_findings.append(f"{tool} error: {result['error']}")
                continue

            root_prefix = "VALUE"
            if tool.startswith("find_user_id") or tool.startswith("get_customer_by") or tool == "get_user_details":
                root_prefix = "USER_ID"
            elif tool.startswith("search_direct_flight") or tool.startswith("search_onestop_flight"):
                root_prefix = "FLIGHT"
                
            obs = result["observation"]
            try:
                data = json.loads(obs)
            except (json.JSONDecodeError, TypeError):
                data = obs
                
            if root_prefix == "USER_ID" and "user_internal" in self.sanitizer.checked_categories:
                self.sanitizer.extract_all(data, prefix=root_prefix)
            elif root_prefix == "USER_ID" and not isinstance(data, dict):
                self.sanitizer.extract_all(data, prefix=root_prefix)
            else:
                self.sanitizer.data_retrieval_extract(data, prefix=root_prefix)
            key_findings.append(f"{tool} result:\n{result['observation']}")

        return {
            "ready_for_final_answer": False,
            "feedback": " | ".join(feedback_parts) + ". Check key_findings for extracted data.",
            "key_findings": "\n\n".join(key_findings),
            "token_usage": {"input_tokens": 0, "output_tokens": 0}
        }

    def local_judgment(self, question, current_step):
        step_info = self._format_current_step(current_step)

        if hasattr(self.ctx, "choices") and self.ctx.choices:
            question = f"{question}\nChoices:\n{self.ctx.choices}"

        disable_ready = (self.args.decision_making == "cloud")
        prompt = prompts.get_local_judge_react_prompt(question=question, step_info=step_info, disable_ready=disable_ready)
        reflection_fn = (
            (lambda x: prompts.get_reflection_judge_prompt(
                x, question, step_info, extra_instructions=self._judge_extra_instructions
            ))
            if self.args.tier == "pro" else None
        )
        result = call_llm(self.args, prompt, role="local", reflection_fn=reflection_fn)
        if result is None:
            result = {}

        _val = result.get("ready_for_final_answer", False)
        if isinstance(_val, bool):
            result["ready_for_final_answer"] = _val
        elif str(_val).lower() in ["true", "yes"]:
            result["ready_for_final_answer"] = True
        else:
            result["ready_for_final_answer"] = False

        return result

    def local_sanitize(self, content, is_data_retrieval_extract=False):
        is_tau2_privacy_1 = self.args.dataset.startswith("tau2_") and self.sanitizer.privacy_level == 1
        if is_data_retrieval_extract or is_tau2_privacy_1:
            sanitized = self.sanitizer.sanitize(content)
        else:
            sanitized = self.sanitizer.generate_sanitized_content(self.args, content)
        self.ctx.private_mapping.update(self.sanitizer.get_mapping())
        return sanitized

    def run(self):
        self.sanitize_question()

        local_reasoning = None
        previous_reasoning = None
        final_prediction = ""
        is_correct = False
        self.current_plan = None

        for step in range(self.args.max_steps):
            # --- Cloud reasoning ---
            try:
                reasoning, actions = self.propose_actions(local_reasoning, previous_reasoning, step=step)
                previous_reasoning = reasoning

                if isinstance(reasoning, dict) and "plan" in reasoning:
                    self.current_plan = self.sanitizer.desanitize_data(reasoning["plan"])

                if not isinstance(actions, list):
                    actions = [actions]
                # Tolerate the LLM occasionally returning ``{"action": [...]}`` or
                # ``{"actions": [[...]]}`` by flattening one level of nesting.
                flat_actions = []
                for a in actions:
                    if isinstance(a, list):
                        flat_actions.extend(x for x in a if isinstance(x, dict))
                    elif isinstance(a, dict):
                        flat_actions.append(a)
                actions = flat_actions
            except Exception as e:
                final_prediction = f"Error: {str(e)}"
                break

            # --- Desanitise & execute ---
            sanitized_actions = json.dumps(actions, ensure_ascii=False)
            actions = self.sanitizer.prepare_actions_for_execution(actions)

            def execute_one(idx_action):
                idx, action = idx_action
                action_work_dir = os.path.join(self.scratch_dir, f"action_{idx}")
                os.makedirs(action_work_dir, exist_ok=True)
                res = toolbox.execute(action, file_path_var=self.local_file, work_dir=action_work_dir, task_id=self.ctx.task_id)
                return {"action": action, "result": res}

            if len(actions) == 0:
                step_results = []
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(actions)) as executor:
                    step_results = list(executor.map(execute_one, enumerate(actions)))

            # --- Real history (raw, unsanitised) ---
            cloud_output = reasoning if isinstance(reasoning, dict) else {"reasoning": reasoning}

            cloud_input_str = f"{self.question}\n"
            if self.ctx.step_history:
                cloud_input_str += "\nFeedback History:\n"
                for past_step in self.ctx.step_history:
                    fb = past_step["local_judgment_sanitized"]["feedback"]
                    cloud_input_str += f"Step {past_step['step']}: {fb}\n"

            history_entry = {
                "step": step,
                "cloud_input": cloud_input_str.strip(),
                "cloud_output": cloud_output,
                "local_execution": step_results,
            }
            self.real_history.append(history_entry)

            # --- Local judgment over the current step only ---
            is_tau2_user_data_retrieval = self._is_tau2_user_data_retrieval(step_results)
            try:
                if is_tau2_user_data_retrieval:
                    judgment = self.tau2_local_judgment(history_entry)
                else:
                    judgment = self.local_judgment(self.original_question, history_entry)
                history_entry["local_judgment"] = judgment
            except Exception as e:
                judgment = {
                    "ready_for_final_answer": False,
                    "feedback": f"System Error in Local Judgment: {str(e)}",
                    "key_findings": "System Error: Local Judgment Failed",
                }
                history_entry["local_judgment"] = judgment

            # --- Sanitise findings/feedback for the next cloud round ---
            _kf = judgment.get("key_findings", "")
            if isinstance(_kf, list):
                _kf = "\n".join(str(x) for x in _kf)
            _fb = judgment.get("feedback", "")
            if isinstance(_fb, list):
                _fb = "\n".join(str(x) for x in _fb)
            sanitized_findings = self.local_sanitize(_kf, is_data_retrieval_extract=is_tau2_user_data_retrieval)
            sanitized_feedback = self.local_sanitize(_fb, is_data_retrieval_extract=is_tau2_user_data_retrieval)
            local_reasoning = {"feedback": sanitized_feedback, "key_findings": sanitized_findings}

            self.sanitizer.step_token_usage = {"input_tokens": 0, "output_tokens": 0}

            self.ctx.add_step(
                step_num=step,
                cloud_output=cloud_output,
                cloud_output_sanitized=sanitized_actions,
                local_execution=step_results,
                local_execution_sanitized=None,
                local_judgment=judgment,
                local_judgment_sanitized={"feedback": sanitized_feedback, "key_findings": sanitized_findings},
            )
            self.ctx.private_mapping = self.sanitizer.get_mapping()

            self.sanitized_findings_history.append({
                "step": step,
                "sanitized_findings": sanitized_findings,
            })

            has_final_tool = any(
                (sr.get("action", {}).get("tool") == "final_answer")
                for sr in step_results
            )
            has_transfer_tool = any(
                (sr.get("action", {}).get("tool") == "transfer_to_human_agents")
                for sr in step_results
            )

            is_ready = judgment["ready_for_final_answer"]
            should_terminate = False

            if has_transfer_tool:
                should_terminate = True
            elif self.args.decision_making == "cloud":
                should_terminate = has_final_tool
            elif self.args.decision_making == "local":
                should_terminate = is_ready
            elif self.args.decision_making == "joint":
                should_terminate = is_ready and has_final_tool
            else:
                should_terminate = is_ready

            if should_terminate:
                full_orig_question = self.original_question
                if hasattr(self.ctx, "choices") and self.ctx.choices:
                    full_orig_question = f"{full_orig_question}\nChoices:\n{self.ctx.choices}"

                final_prediction, is_correct, self.real_history = process_final_answer(
                    args=self.args,
                    question=full_orig_question,
                    ground_truth=self.ground_truth,
                    history=self.real_history,
                    sanitizer=self.sanitizer,
                    task_id=self.ctx.task_id,
                    toolbox=toolbox,
                )

                break

        else:
            full_orig_question = self.original_question
            if hasattr(self.ctx, "choices") and self.ctx.choices:
                full_orig_question = f"{full_orig_question}\nChoices:\n{self.ctx.choices}"

            final_prediction, is_correct, self.real_history = process_final_answer(
                args=self.args,
                question=full_orig_question,
                ground_truth=self.ground_truth,
                history=self.real_history,
                sanitizer=self.sanitizer,
                task_id=self.ctx.task_id,
                toolbox=toolbox,
            )

        result = {
            "task_id": self.task_id,
            "question": self.original_question,
            "level": self.level,
            "ground_truth": self.ground_truth,
            "final_prediction": final_prediction,
            "is_correct": is_correct,
            "privacy": {
                "level": self.args.privacy_level,
                "mapping": self.ctx.private_mapping,
            },
            "history": self.real_history,
        }

        if os.path.exists(self.scratch_dir):
            shutil.rmtree(self.scratch_dir)

        return is_correct, result
        
    def get_file_context(self):
        if self.file_name and self.local_file:
            return f"""
RELEVANT FILE:
The file name is '{self.file_name}'.
The absolute path is: {self.local_file}
IMPORTANT: In Python code, always open the file using the absolute path above. Do NOT assume the current working directory contains the file.
"""
        return ""

    def _get_compact_tools(self):
        tools = []
        for tool in self.tools_list:
            entry = {"name": tool["name"], "desc": tool["description"]}
            if "args" in tool and tool["args"]:
                entry["args"] = tool["args"]
            tools.append(entry)
        if self.args.decision_making == "local":
            tools = [t for t in tools if t["name"] != "final_answer"]
        return tools


class ReActStrategy(AgentStrategy):

    def _get_compact_history(self, max_words=2000):
        compact = []
        for step_record in getattr(self.ctx, "step_history", []):
            step = step_record.get("step")

            cloud_output = step_record.get("cloud_output", {})
            if isinstance(cloud_output, str):
                try:
                    cloud_output = json.loads(cloud_output)
                except Exception:
                    cloud_output = {}
            if not isinstance(cloud_output, dict):
                cloud_output = {}

            thought = cloud_output.get("reasoning", "")

            action = cloud_output.get("action", {})
            if not action and "actions" in cloud_output and cloud_output["actions"]:
                action = cloud_output["actions"][0]

            local_judg_san = step_record.get("local_judgment_sanitized", {})
            obs = str(local_judg_san.get("key_findings", ""))

            words = obs.split()
            if len(words) > max_words:
                obs = " ".join(words[:max_words]) + "...[TRUNCATED]"

            compact.append({"step": step, "thought": thought, "action": action, "obs": obs})

        return json.dumps(compact, indent=4, ensure_ascii=False)
    
    def cloud_reasoning(self, local_reasoning=None, step=None):
        prompt = prompts.get_cloud_react_prompt(
            question=self.question,
            step=step+1,
            max_steps=self.args.max_steps,
            file_context=self.get_file_context(),
            tools_json=json.dumps(self._get_compact_tools(), indent=4, ensure_ascii=False),
            history_json=self._get_compact_history(),
            feedback=local_reasoning if local_reasoning else "None",
            privacy_level=self.args.privacy_level
        )

        tools_json = json.dumps(self._get_compact_tools(), ensure_ascii=False)
        return call_llm(self.args, prompt, role="cloud", reflection_fn=lambda x: prompts.get_reflection_cloud_prompt(self.args.strategy, x, question=self.question, tools_json=tools_json, history_json=self._get_compact_history(), feedback=local_reasoning if local_reasoning else "None", extra_instructions=self._cloud_extra_instructions))

    def propose_actions(self, local_reasoning, previous_reasoning=None, step=None):
        response = self.cloud_reasoning(local_reasoning, step=step)
        reasoning = response.get("reasoning", "")

        if "reflection" in response or "token_usage" in response:
            if isinstance(reasoning, str):
                reasoning = {"reasoning": reasoning}
            if "reflection" in response:
                reasoning["reflection"] = response["reflection"]
            if "token_usage" in response:
                reasoning["token_usage"] = response["token_usage"]

        action = response.get("action", {})
        actions = []
        if action:
            actions.append(action)
        elif "actions" in response and response.get("actions"):
            actions = response.get("actions")

        return reasoning, actions


class RecurrentGPTStrategy(AgentStrategy):
    def _get_compact_history(self, max_words=2000):
        compact = []
        for item in self.sanitized_findings_history:
            obs = str(item.get("sanitized_findings", ""))
            words = obs.split()
            if len(words) > max_words:
                obs = " ".join(words[:max_words]) + "...[TRUNCATED]"
            compact.append({"step": item.get("step"), "obs": obs})
        return json.dumps(compact, indent=4, ensure_ascii=False)

    def cloud_reasoning(self, local_reasoning=None, previous_reasoning=None, step=None):
        prev_reasoning_str = ""
        if previous_reasoning:
            if isinstance(previous_reasoning, dict):
                prev_reasoning_str = previous_reasoning.get("reasoning", "")
            else:
                prev_reasoning_str = str(previous_reasoning)

        skip_ctx = f"\nPrevious Step Reasoning Sequence (Working Memory):\n{prev_reasoning_str}\n" if prev_reasoning_str else ""

        prompt = prompts.get_cloud_recurrentgpt_prompt(
            question=self.question,
            step=step+1,
            max_steps=self.args.max_steps,
            file_context=self.get_file_context(),
            skip_ctx=skip_ctx,
            tools_json=json.dumps(self._get_compact_tools(), ensure_ascii=False),
            history_json=self._get_compact_history(),
            feedback=local_reasoning if local_reasoning else "None",
            privacy_level=self.args.privacy_level
        )

        tools_json = json.dumps(self._get_compact_tools(), ensure_ascii=False)
        return call_llm(self.args, prompt, role="cloud", reflection_fn=lambda x: prompts.get_reflection_cloud_prompt(self.args.strategy, x, question=self.question, tools_json=tools_json, history_json=self._get_compact_history(), feedback=local_reasoning if local_reasoning else "None", extra_instructions=self._cloud_extra_instructions))

    def propose_actions(self, local_reasoning, previous_reasoning=None, step=None):
        response = self.cloud_reasoning(local_reasoning, previous_reasoning, step=step)
        reasoning = response.get("reasoning", "")

        if "reflection" in response or "token_usage" in response:
            if isinstance(reasoning, str):
                reasoning = {"reasoning": reasoning}
            if "reflection" in response:
                reasoning["reflection"] = response["reflection"]
            if "token_usage" in response:
                reasoning["token_usage"] = response["token_usage"]

        action = response.get("action", {})
        actions = [action] if action else []
        return reasoning, actions


class ParallelPlanAndSolveStrategy(AgentStrategy):
    def _get_compact_history(self, max_words=2000):
        compact = []
        for item in self.sanitized_findings_history:
            obs = str(item.get("sanitized_findings", ""))
            words = obs.split()
            if len(words) > max_words:
                obs = " ".join(words[:max_words]) + "...[TRUNCATED]"
            compact.append({"step": item.get("step"), "obs": obs})
        return json.dumps(compact, indent=4, ensure_ascii=False)

    def cloud_reasoning(self, feedback, previous_plan_context=None, step=None):
        plan_context = ""
        if previous_plan_context and isinstance(previous_plan_context, list):
            plan_context = str(json.dumps(previous_plan_context, indent=4, ensure_ascii=False))

        tools_json = json.dumps(self._get_compact_tools(), ensure_ascii=False)

        if self.args.dataset.startswith("tau2_"):
            prompt = prompts.get_cloud_tau2_prompt(
                question=self.question,
                step=step+1,
                max_steps=self.args.max_steps,
                plan_context=plan_context,
                tools_json=tools_json,
                history_json=self._get_compact_history(),
                feedback=feedback,
                privacy_level=self.args.privacy_level,
            )
        else:
            prompt = prompts.get_cloud_parallel_plan_and_solve_prompt(
                question=self.question,
                step=step+1,
                max_steps=self.args.max_steps,
                file_context=self.get_file_context(),
                plan_context=plan_context,
                tools_json=tools_json,
                history_json=self._get_compact_history(),
                feedback=feedback,
                privacy_level=self.args.privacy_level
            )

        return call_llm(self.args, prompt, role="cloud", reflection_fn=lambda x: prompts.get_reflection_cloud_prompt(self.args.strategy, x, question=self.question, tools_json=tools_json, history_json=self._get_compact_history(), feedback=feedback, extra_instructions=self._cloud_extra_instructions))

    def propose_actions(self, local_reasoning, previous_reasoning=None, step=None):
        feedback = local_reasoning

        if isinstance(local_reasoning, dict):
            feedback = local_reasoning["feedback"]

        if isinstance(local_reasoning, dict) and local_reasoning.get("ready_for_final_answer"):
            feedback = f"[IMPORTANT: LOCAL JUDGE SAYS TASK IS DONE. PLEASE SUBMIT FINAL ANSWER.]\n{feedback}"

        response = self.cloud_reasoning(feedback, previous_reasoning, step=step)

        current_reasoning = response.get("reasoning", "")
        current_plan = response.get("plan", [])
        actions = response.get("actions", [])

        output = {"reasoning": current_reasoning, "plan": current_plan}
        if "reflection" in response:
            output["reflection"] = response["reflection"]
        if "token_usage" in response:
            output["token_usage"] = response["token_usage"]

        return output, actions

    def local_judgment(self, question, current_step):
        if self._is_tau2_user_data_retrieval(current_step["local_execution"]):
            return self.tau2_local_judgment(current_step)

        step_info = self._format_current_step(current_step)
        plan_info = f"\n\nCurrent Plan:\n{json.dumps(self.current_plan, indent=2, ensure_ascii=False)}" if self.current_plan else ""

        full_question = question
        if hasattr(self.ctx, "choices") and self.ctx.choices:
            full_question = f"{question}\nChoices:\n{self.ctx.choices}"

        disable_ready = (self.args.decision_making == "cloud")
        prompt = prompts.get_local_judge_plan_and_solve_prompt(
            question=full_question,
            step_info=step_info,
            plan_info=plan_info,
            disable_ready=disable_ready,
        )

        reflection_fn = (
            (lambda x: prompts.get_reflection_judge_prompt(
                x, full_question, step_info, plan_info, extra_instructions=self._judge_extra_instructions
            ))
            if self.args.tier == "pro" else None
        )
        result = call_llm(self.args, prompt, role="local", reflection_fn=reflection_fn)
        if result is None:
            result = {}

        _val = result.get("ready_for_final_answer", False)
        if isinstance(_val, bool):
            result["ready_for_final_answer"] = _val
        elif str(_val).lower() in ["true", "yes"]:
            result["ready_for_final_answer"] = True
        else:
            result["ready_for_final_answer"] = False

        if "key_findings" not in result:
            result["key_findings"] = ""
        if "feedback" not in result:
            result["feedback"] = ""

        return result


class PlanAndSolveStrategy(ParallelPlanAndSolveStrategy):
    def cloud_reasoning(self, feedback=None, previous_plan_context=None, step=None):
        plan_context = ""
        if previous_plan_context and isinstance(previous_plan_context, list):
            plan_context = f"\nPlan: {json.dumps(previous_plan_context, indent=4, ensure_ascii=False)}"
            
        prompt = prompts.get_cloud_plan_and_solve_prompt(
            question=self.question,
            step=step+1,
            max_steps=self.args.max_steps,
            file_context=self.get_file_context(),
            plan_context=plan_context,
            tools_json=json.dumps(self._get_compact_tools(), indent=4, ensure_ascii=False),
            history_json=self._get_compact_history(),
            feedback=feedback,
            privacy_level=self.args.privacy_level
        )

        tools_json = json.dumps(self._get_compact_tools(), ensure_ascii=False)
        return call_llm(self.args, prompt, role="cloud", reflection_fn=lambda x: prompts.get_reflection_cloud_prompt(self.args.strategy, x, question=self.question, tools_json=tools_json, history_json=self._get_compact_history(), feedback=feedback, extra_instructions=self._cloud_extra_instructions))

    def propose_actions(self, local_reasoning, previous_reasoning=None, step=None):
        feedback = local_reasoning

        if isinstance(local_reasoning, dict):
            feedback = local_reasoning.get("feedback", "")

        if isinstance(local_reasoning, dict) and local_reasoning.get("ready_for_final_answer"):
            feedback = f"[IMPORTANT: LOCAL JUDGE SAYS TASK IS DONE. PLEASE SUBMIT FINAL ANSWER.]\n{feedback}"

        response = self.cloud_reasoning(feedback, previous_reasoning, step=step)

        current_reasoning = response.get("reasoning", "")
        current_plan = response.get("plan", [])

        action = response.get("action", None)
        actions = [action] if action else []

        output = {"reasoning": current_reasoning, "plan": current_plan}
        if "reflection" in response:
            output["reflection"] = response["reflection"]
        if "token_usage" in response:
            output["token_usage"] = response["token_usage"]

        return output, actions

# =========================
# Agent Run
# =========================
def run_agent(args, task_data):
    if args.strategy == "react":
        strategy = ReActStrategy(args, task_data)
    elif args.strategy == "recurrent_gpt":
        strategy = RecurrentGPTStrategy(args, task_data)
    elif args.strategy == "parallel_plan_and_solve":
        strategy = ParallelPlanAndSolveStrategy(args, task_data)
    elif args.strategy == "plan_and_solve":
        strategy = PlanAndSolveStrategy(args, task_data)
    else:
        raise ValueError(f"Unknown strategy: {args.strategy}")

    try:
        return strategy.run()
    except Exception as e:
        print(f"[ERROR] Fatal error in strategy {args.strategy}: {e}\n{traceback.format_exc()}")
        captured_history = getattr(strategy, "real_history", [])

        fallback_answer = None
        for step in reversed(captured_history):
            for exec_log in reversed(step.get("local_execution", [])):
                action = exec_log.get("action", {})
                result = exec_log.get("result", {})
                if (isinstance(action, dict) and action.get("tool") == "final_answer"
                        and isinstance(result, dict) and result.get("success")):
                    fallback_answer = result.get("answer", "")
                    break
            if fallback_answer is not None:
                break

        ground_truth = task_data["ground_truth"]
        if args.dataset.startswith("tau2_"):
            try:
                final_prediction, is_correct, captured_history = process_final_answer(
                    args=args,
                    question=task_data["question"],
                    ground_truth=ground_truth,
                    history=captured_history,
                    sanitizer=strategy.sanitizer,
                    task_id=task_data["task_id"],
                    toolbox=toolbox,
                )
                print(f"[INFO] Tau2 fallback evaluation: correct={is_correct}")
            except Exception:
                final_prediction = f"[Tau2 Eval: Correct=False, Error={e}]"
                is_correct = False
        elif fallback_answer is not None:
            final_prediction = fallback_answer
            prompt = prompts.get_evaluation_prompt(task_data["question"], ground_truth, fallback_answer)
            is_correct, _ = evaluate_result(args, prompt)
            print(f"[INFO] Fallback to final_answer tool result: {fallback_answer} (correct={is_correct})")
        else:
            final_prediction = f"[Tau2 Eval: Correct=False, Error={e}]" if args.dataset.startswith("tau2_") else f"Error: {e}"
            is_correct = False

        return is_correct, {
            "task_id": task_data["task_id"],
            "question": task_data["question"],
            "level": task_data.get("level", ""),
            "ground_truth": ground_truth,
            "final_prediction": final_prediction,
            "is_correct": is_correct,
            "history": captured_history,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Agent")
    parser.add_argument("--max_steps", type=int, default=10, help="Max steps")
    parser.add_argument("--num_samples", type=int, default=20, help="Number of samples to run (default: all)")

    parser.add_argument("--workers", type=int, default=20, help="Number of workers")
    parser.add_argument("--run_name", type=str, default="default", help="Unique name for this run to isolate scratch directories")
    parser.add_argument("--strategy", type=str, default="parallel_plan_and_solve", choices=["react", "recurrent_gpt", "plan_and_solve", "parallel_plan_and_solve"], help="Agent Strategy")
    parser.add_argument("--privacy_level", type=int, default=0, choices=[0, 1, 2, 3], help="Privacy Level: 0 = No Privacy; 1 = Sanitize Local Exec; 2 = Privacy 1 + Sanitize Prompt; 3 = Privacy 2 + Sanitize Search")

    parser.add_argument("--local_model", type=str, default="Qwen/Qwen3-4B-Instruct-2507", help="Local model name")
    parser.add_argument("--cloud_model", type=str, default="gemini-3-flash-preview", help="Cloud model name")
    
    parser.add_argument("--tier", type=str, default="base", choices=["base", "pro"], help="Tier: base (standard), pro (1-step reflection)")
    parser.add_argument("--decision_making", type=str, default="joint", choices=["original", "local", "cloud", "joint"], help="Termination policy: original (judge-only baseline), local (judge decides), cloud (agent decides), joint (both must agree)")
    parser.add_argument("--run_idx", type=int, default=None, help="Run index for repeated experiments (e.g., 0, 1, 2). When set, appended to output filename.")
    parser.add_argument("--vllm_port", type=int, default=8000, help="Port for the vLLM API server")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save the final results (default is results/)")
    parser.add_argument("--output_filename", type=str, default="", help="Custom output filename (without .json). If empty, auto-generated from args.")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="gaia", 
        choices=[
            "tau2_airline", "tau2_retail", # Agent
            "gaia", # Agent
            "gsm8k", "math_qa",  # Math
            "geometry3k", "mathvista",  # Multimodal Math
            "scibench", "sciq", # Science
            "truthful_qa", "hotpot_qa", "fever", # Factual Reasoning
            "clutrr", "agieval_lsat_ar", # Logic
            "med_qa", # Medical
            "finqa", # Finance
            "mmlu_professional_accounting", # Accounting
            "mmmu_accounting", # Multimodal Accounting
            "jeopardy_mc_history", # History
            "jeopardy_mc_literature", # Literature
        ], 
        help="Dataset to evaluating on"
    )
    args = parser.parse_args()

    if os.path.exists("__pycache__"):
        shutil.rmtree("__pycache__", ignore_errors=True)

    run_scratch = f"scratch/{args.run_name}"
    if os.path.exists(run_scratch):
        shutil.rmtree(run_scratch, ignore_errors=True)
    os.makedirs(run_scratch, exist_ok=True)

    multiprocessing.set_start_method("spawn", force=True)

    if not config.has_privacy_level(args.dataset, args.privacy_level):
        sys.exit(0)

    if args.dataset.startswith("tau2_"):
        toolbox = Tau2ToolBox(args)
    else:
        toolbox = ToolBox(args)

    ds = data_loader.load_data(args.dataset)

    if args.num_samples is not None and args.num_samples > 0:
        ds = ds[:min(len(ds), args.num_samples)]

    output_dir = os.path.join(args.results_dir, args.dataset)
    run_suffix = f"_run_{args.run_idx}" if args.run_idx is not None else ""
    if args.output_filename:
        output_file = f"{output_dir}/{args.output_filename}{run_suffix}.json"
    else:
        output_file = (
            f"{output_dir}/strategy_{args.strategy}_privacy_{args.privacy_level}"
            f"_tier_{args.tier}_decision_{args.decision_making}{run_suffix}.json"
        )

    saver = IncrementalSaver(output_file)
    pending_ds = saver.filter_pending(ds)
    correct_count = sum(1 for t in saver.traces if t.get("is_correct"))

    if not pending_ds:
        final_acc = (correct_count / len(ds)) * 100 if ds else 0
        print(f"[INFO] All tasks already completed. Final Accuracy: {final_acc:.2f}%")
        sys.exit(0)

    print(f"[Resume] Running {len(pending_ds)}/{len(ds)} pending tasks.")

    def worker(item):
        return run_agent(args, item)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_task = {executor.submit(worker, item): orig_idx for orig_idx, item in pending_ds}
        for future in concurrent.futures.as_completed(future_to_task):
            idx = future_to_task[future]
            try:
                is_correct, trace = future.result()
                saver.save(trace)

                completed_count = len(saver.traces)
                if is_correct:
                    correct_count += 1

                acc = (correct_count / completed_count) * 100
                print(f"[Progress] Task {idx} Finished. Correct: {is_correct}. Current Acc: {acc:.2f}% ({correct_count}/{completed_count})")
            except Exception as e:
                print(f"[ERROR] Task {idx} failed: {e}")
                traceback.print_exc()

            sys.stdout.flush()

    final_acc = (correct_count / len(ds)) * 100 if ds else 0
    print(f"[INFO] Final Accuracy: {final_acc:.2f}%")
    print(f"[INFO] Results saved to {output_file}")

