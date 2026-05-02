import io
import json
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from urllib.parse import parse_qs, unquote, urlparse

import arxiv
import pandas as pd
import pypdf
import requests
import speech_recognition as sr
import wikipedia
from bs4 import BeautifulSoup
from ddgs import DDGS
from docx import Document
from googlesearch import search as google_search
from pptx import Presentation
from pydub import AudioSegment
from tenacity import retry, stop_after_attempt, wait_random

from utils import call_gemini_search, call_gemini_vision


# =========================
# Web search shared state (rate limit + cache)
# =========================
_WEB_SEARCH_RATE_LOCK = threading.Lock()
_WEB_SEARCH_NEXT_TS = {}

_WEB_SEARCH_CACHE_LOCK = threading.Lock()
_WEB_SEARCH_CACHE = {}


# =========================
# Standalone Helper Functions
# =========================

_PYTHON_EXEC_HELPER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "tools_python_exec.py",
)


# =========================
# ToolBox Class
# =========================

class ToolBox:
    """Registry and dispatcher for the cloud-side execution tools."""

    def __init__(self, args):
        self.args = args
        self._registry = [
            {
                "name": "web_search",
                "description": "Search the web for information using a search engine.",
                "args": {
                    "query": "string"
                }
            },
            {
                "name": "visit_website",
                "description": "Visit a URL and extract its text content. Use this to read full pages found by search.",
                "args": {
                    "url": "string"
                }
            },
            {
                "name": "arxiv_search",
                "description": "Search specifically for academic papers on arXiv. Returns the title, authors, summary, and PDF URL. Use this when the user asks for scientific papers.",
                "args": {
                    "query": "string"
                }
            },
            {
                "name": "wikipedia_lookup",
                "description": "Lookup an entity on Wikipedia",
                "args": {
                    "entity": "string"
                }
            },
            {
                "name": "python_exec",
                "description": "Execute python code for calculation, process files, etc. (Headless: do NOT open GUI viewers or call Image.show()/matplotlib.show().)",
                "args": {
                    "code": "string"
                }
            },
            {
                "name": "read_file",
                "description": "Read content from a file (support txt, csv, json, pdf, docx, xlsx, pptx, zip, images, audio). Uses specialized libraries to extract text/data. For images, uses Vision LLM to describe the image; for audio, performs transcription. Returns the full content.",
            },
            {
                "name": "final_answer",
                "description": "Signal that you have solved the problem. Provide the answer you found (including placeholders/tokens if that is what you see) and your reasoning. This stops the loop and triggers verification by the Device Agent.",
                "args": {
                    "answer": "string",
                    "reasoning": "string"
                }
            },
        ]

    @property
    def registry(self):
        return self._registry

    def web_search(self, query, max_results=5):
        """Search the web with a cascading fallback over multiple providers."""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_8 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Mobile/15E148 Safari/604.1"
        ]

        def get_headers(referer="https://www.google.com/"):
            return {
                "User-Agent": random.choice(user_agents),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": referer,
                "Connection": "close",
            }

        errors = []

        # --- Strategy 0: Gemini Search with grounding (preferred path) ---
        try:
            response = call_gemini_search(query)
            results = response.get("results", [])
            if results:
                payload = {
                    "tool": "web_search", 
                    "success": True, 
                    "results": []
                }
                for result in results:
                    if isinstance(result, dict):
                        link = result.get("link", "")
                        if len(link) > 100:
                            link = link[:100] + "..."
                        payload["results"].append({
                            "title": result.get("title", ""),
                            "link": link,
                            "snippet": result.get("snippet", ""),
                            "source": "gemini_grounding"
                        })
                    else:
                        payload["results"].append({
                            "snippet": str(result),
                            "source": "gemini_grounding"
                        })
                return payload
            else:
                error_msg = response.get("error", "Gemini grounding returned no results")
                errors.append(f"Gemini Grounding: {error_msg[:100]}")
        except Exception as e:
            errors.append(f"Gemini Grounding Error: {e}")

        cache_key = (query, int(max_results))
        with _WEB_SEARCH_CACHE_LOCK:
            cached = _WEB_SEARCH_CACHE.get(cache_key)
        if cached:
            return cached

        def _rate_limit(domain, min_interval_s):
            now = time.time()
            with _WEB_SEARCH_RATE_LOCK:
                next_ts = _WEB_SEARCH_NEXT_TS.get(domain, 0.0)
                wait_s = max(0.0, next_ts - now)
                reserved = max(next_ts, now) + min_interval_s
                _WEB_SEARCH_NEXT_TS[domain] = reserved
            if wait_s > 0:
                time.sleep(wait_s + random.uniform(0.05, 0.25))
            else:
                time.sleep(random.uniform(0.05, 0.25))

        def _ddg_decode_link(href):
            # DDG Lite often returns /l/?uddg=<encoded>.
            parsed = urlparse(href)
            if parsed.netloc:
                return href
            qs = parse_qs(parsed.query or "")
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
            return href

        def _finish(results):
            payload = {"tool": "web_search", "success": True, "results": results[:max_results]}
            with _WEB_SEARCH_CACHE_LOCK:
                _WEB_SEARCH_CACHE[cache_key] = payload
            return payload

        # --- Strategy 1: googlesearch-python ---
        try:
            _rate_limit("google.com", 2.0)
            google_results = list(google_search(query, region="us", num_results=max_results, advanced=True))
            if google_results:
                results = []
                for r in google_results:
                    results.append({
                        "title": r.title,
                        "link": r.url,
                        "snippet": r.description,
                        "source": "googlesearch_lib"
                    })
                return _finish(results)
            else:
                errors.append("Google Lib: No results found.")
        except Exception as e:
            errors.append(f"Google Lib Error: {e}")

        # --- Strategy 2: DuckDuckGo (ddgs library) ---
        try:
            _rate_limit("duckduckgo.com", 0.7)
            with DDGS() as ddgs:
                ddgs_results = list(ddgs.text(query, region='us-en', max_results=max_results, backend="auto"))
            
            if ddgs_results:
                results = []
                for r in ddgs_results[:max_results]:
                    results.append({
                        "title": r.get("title", ""),
                        "link": r.get("href", ""),
                        "snippet": r.get("body", ""),
                        "source": "ddgs_lib",
                    })
                return _finish(results)
            else:
                errors.append("DDGS Lib: No results found.")
        except Exception as e:
            errors.append(f"DDGS Lib Error: {e}")

        # --- Strategy 3: DuckDuckGo HTML endpoint ---
        try:
            url = "https://html.duckduckgo.com/html/"
            headers = get_headers("https://html.duckduckgo.com/")
            _rate_limit("duckduckgo.com", 0.7)
            response = requests.get(url, params={"q": query}, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                for result in soup.find_all('div', class_='result'):
                    if len(results) >= max_results:
                        break
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = (
                        result.find('a', class_='result__snippet')
                        or result.find('div', class_='result__snippet')
                        or result.find('span', class_='result__snippet')
                    )
                    if title_elem:
                        results.append({
                            "title": title_elem.get_text(strip=True),
                            "link": title_elem.get('href'),
                            "snippet": snippet_elem.get_text(strip=True) if snippet_elem else "",
                            "source": "ddg_html"
                        })
                if results:
                    return _finish(results)
                errors.append("DDG HTML: 200 OK but no results via parser.")
            else:
                errors.append(f"DDG HTML: Status {response.status_code}")
        except Exception as e:
            errors.append(f"DDG HTML Error: {e}")

        # --- Strategy 4: DuckDuckGo Lite endpoint ---
        try:
            url = "https://lite.duckduckgo.com/lite/"
            headers = get_headers("https://lite.duckduckgo.com/")
            _rate_limit("duckduckgo.com", 0.8)
            response = requests.post(url, data={"q": query}, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                results = []
                for link_tag in soup.select("a.result-link"):
                    if len(results) >= max_results:
                        break
                    title = link_tag.get_text(strip=True)
                    href = _ddg_decode_link(link_tag.get("href") or "")
                    row = link_tag.find_parent("tr")
                    nxt = row.find_next_sibling("tr") if row else None
                    snippet_td = nxt.select_one("td.result-snippet") if nxt else None
                    snippet = snippet_td.get_text(strip=True) if snippet_td else ""
                    if title and href:
                        results.append({"title": title, "link": href, "snippet": snippet, "source": "ddg_lite"})
                if results:
                    return _finish(results)
                errors.append("DDG Lite: 200 OK but no results via parser.")
            else:
                errors.append(f"DDG Lite: Status {response.status_code}")
        except Exception as e:
            errors.append(f"DDG Lite Error: {e}")
        
        # --- Strategy 5: Bing search ---
        try:
            url = "https://www.bing.com/search"
            headers = get_headers("https://www.bing.com/")
            headers["Cookie"] = "SRCHHPGUSR=SRCHLANG=en"
            _rate_limit("bing.com", 0.9)
            response = requests.get(url, params={"q": query}, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                for li in soup.select("li.b_algo"):
                    if len(results) >= max_results:
                        break
                    a = li.select_one("h2 a")
                    if not a:
                        continue
                    href = a.get("href", "")
                    if not href.startswith("http"):
                        continue
                    snippet_tag = li.select_one("div.b_caption p") or li.select_one("p")
                    snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                    results.append({
                        "title": a.get_text(strip=True),
                        "link": href,
                        "snippet": snippet,
                        "source": "bing",
                    })
                if results:
                    return _finish(results)
                errors.append("Bing: 200 OK but no results via parser.")
            else:
                errors.append(f"Bing: Status {response.status_code}")
        except Exception as e:
            errors.append(f"Bing Error: {e}")

        # --- Strategy 6 (last resort): ungrounded Gemini ---
        # The output is generated without grounding and may hallucinate; only
        # invoked after every previous strategy has failed.
        try:
            from utils import call_gemini_fallback_search
            response = call_gemini_fallback_search(query)
            results = response.get("results", [])
            if results:
                payload = {
                    "tool": "web_search", 
                    "success": True, 
                    "results": []
                }
                for result in results:
                    if isinstance(result, dict):
                        payload["results"].append({
                            "title": result.get("title", ""),
                            "link": result.get("link", ""),
                            "snippet": result.get("snippet", ""),
                            "source": "gemini_ungrounded_fallback",
                        })
                return payload
            else:
                errors.append("Gemini Fallback: No results")
        except Exception as e:
            errors.append(f"Gemini Fallback Error: {e}")

        return {"tool": "web_search", "success": False, "error": f"All strategies failed. Details: {'; '.join(errors)}"}

    @retry(stop=stop_after_attempt(3), wait=wait_random(min=1, max=5))
    def visit_website(self, url):
        """Fetch a URL and return its text content (HTML or PDF)."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
        except Exception:
            return {
                "tool": "visit_website",
                "success": False,
                "error": f"Invalid URL, HTTP Error, or Connection Failed: '{url}'.",
            }

        content_type = response.headers.get('Content-Type', '').lower()

        if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
            pdf_file = io.BytesIO(response.content)
            reader = pypdf.PdfReader(pdf_file)
            text_content = []
            for i, page in enumerate(reader.pages):
                text_content.append(f"--- Page {i+1} ---\n{page.extract_text()}")
            return {
                "tool": "visit_website",
                "success": True,
                "title": "PDF Document",
                "content": " ".join(("\n".join(text_content)).split()[:2000]),
            }

        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]):
            tag.extract()

        text = soup.get_text(separator='\n', strip=True)

        words = text.split()
        if len(words) > 2000:
            text = " ".join(words[:2000]) + "\n... [Content Truncated due to length] ..."

        return {
            "tool": "visit_website",
            "success": True,
            "title": soup.title.string if soup.title else "No Title",
            "content": text,
        }

    def wikipedia_lookup(self, entity):
        try:
            summary = wikipedia.summary(entity, sentences=3, auto_suggest=False)
            return {
                "tool": "wikipedia_lookup",
                "success": True,
                "summary": summary
            }
        except wikipedia.exceptions.DisambiguationError as e:
            return {
                "tool": "wikipedia_lookup",
                "success": False,
                "error": f"Disambiguation Error: '{entity}' may refer to: {e.options[:5]}"
            }
        except wikipedia.exceptions.PageError:
            return {
                "tool": "wikipedia_lookup",
                "success": False,
                "error": f"Page Error: '{entity}' does not match any Wikipedia page."
            }
        except Exception as e:
            return {
                "tool": "wikipedia_lookup",
                "success": False,
                "error": f"Wikipedia Error: {e}"
            }

    def final_answer(self, answer, reasoning):
        return {
            "tool": "final_answer",
            "success": True,
            "answer": answer,
            "reasoning": reasoning
        }

    def read_file(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return {"tool": "read_file", "success": False, "error": f"File not found or no path provided: {file_path}"}

        ext = os.path.splitext(file_path)[1].lower()
        content = ""

        if ext == '.pdf':
            reader = pypdf.PdfReader(file_path)
            text_list = [f"--- Page {i} ---\n{p.extract_text()}" for i, p in enumerate(reader.pages)]
            content = "\n".join(text_list)

        elif ext == '.csv':
            df = pd.read_csv(file_path)
            content = df.to_string()

        elif ext == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                content = "Archive Contents:\n" + "\n".join(z.namelist())

        elif ext == '.docx':
            doc = Document(file_path)
            text_list = [p.text for p in doc.paragraphs if p.text.strip()]
            content = "\n".join(text_list)

        elif ext in ['.xlsx', '.xls']:
            xl = pd.ExcelFile(file_path)
            sheet_contents = []
            for sheet_name in xl.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                if len(df) > 100:
                    df = df.head(100)
                    sheet_contents.append(f"Sheet '{sheet_name}' (First 100 rows):\n{df.to_string()}")
                else:
                    sheet_contents.append(f"Sheet '{sheet_name}':\n{df.to_string()}")
            content = "\n\n".join(sheet_contents)

        elif ext == '.pptx':
            prs = Presentation(file_path)
            slides = list(prs.slides)

            text_list = []
            for i, slide in enumerate(slides):
                slide_text = []
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        slide_text.append(shape.text)
                text_list.append(f"--- Slide {i} ---\n" + "\n".join(slide_text))
            content = "\n".join(text_list)

        elif ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
            mime_map = {
                '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.tiff': 'image/tiff', '.bmp': 'image/bmp', '.gif': 'image/gif',
            }
            image_mime = mime_map.get(ext, 'image/jpeg')
            content = call_gemini_vision(image_bytes, image_mime, "Describe the image content in detail.")

        elif ext in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
            if ext != '.wav':
                audio = AudioSegment.from_file(file_path)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    wav_path = tmp.name
                    audio.export(wav_path, format='wav')
            else:
                wav_path = file_path

            recognizer = sr.Recognizer()
            with sr.AudioFile(wav_path) as source:
                audio_data = recognizer.record(source)

            if ext != '.wav':
                os.unlink(wav_path)

            content = recognizer.recognize_google(audio_data)

        else:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

        words = content.split()
        if len(words) > 2000:
            content = " ".join(words[:2000]) + "\n... [Content Truncated] ..."

        return {
            "tool": "read_file",
            "success": True,
            "content": content,
        }

    def python_exec(self, code, file_path_var=None, work_dir=None, timeout=60):
        # Spawn a fresh interpreter via the standalone helper.
        env = os.environ.copy()
        if file_path_var:
            env["FILE_PATH"] = file_path_var

        cmd = [sys.executable, _PYTHON_EXEC_HELPER, work_dir or ""]
        try:
            completed = subprocess.run(
                cmd,
                input=code,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {
                "tool": "python_exec",
                "success": False,
                "error": f"Timeout: Code execution exceeded {timeout} seconds.",
                "partial_output": "",
            }

        # Helper writes a single JSON line to stdout. 
        stdout = completed.stdout or ""
        last_line = stdout.strip().splitlines()[-1] if stdout.strip() else ""
        try:
            result = json.loads(last_line) if last_line else None
        except ValueError:
            result = None

        if result is None:
            return {
                "tool": "python_exec",
                "success": False,
                "error": (
                    f"Helper exited without a JSON result line "
                    f"(returncode={completed.returncode}).\n"
                    f"stdout:\n{stdout}\nstderr:\n{completed.stderr or ''}"
                ),
                "partial_output": stdout,
            }

        if result.get("success"):
            return {
                "tool": "python_exec",
                "success": True,
                "output": (
                    f"Stdout/Stderr:\n{result.get('output', '')}\n"
                    f"Local Variables:\n{result.get('local_vars', '{}')}"
                ),
            }
        return {
            "tool": "python_exec",
            "success": False,
            "error": result.get("error") or "Unknown error",
            "partial_output": result.get("output", ""),
        }

    @retry(stop=stop_after_attempt(3), wait=wait_random(min=1, max=5))
    def arxiv_search(self, query, max_results=5):
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance,
            )

            results = []
            for r in client.results(search):
                results.append({
                    "title": r.title,
                    "published": r.published.strftime("%Y-%m-%d"),
                    "summary": r.summary,
                    "pdf_url": r.pdf_url,
                })

            return {
                "tool": "arxiv_search",
                "success": True,
                "results": results,
            }
        except Exception as e:
            return {
                "tool": "arxiv_search",
                "success": False,
                "error": f"Arxiv Search Error: {e}",
            }

    def execute(self, action, file_path_var=None, work_dir=None, task_id=None):
        # The LLM occasionally wraps the action in a list; pull the first dict
        # entry out before dispatching.
        if isinstance(action, list):
            first_dict = next((a for a in action if isinstance(a, dict)), None)
            if first_dict is None:
                return {
                    "error": "Invalid action type: expected dict, got list with no dict elements",
                    "action": action,
                }
            action = first_dict
        if not isinstance(action, dict):
            return {
                "error": f"Invalid action type: expected dict, got {type(action).__name__}",
                "action": action,
            }
        tool = action.get("tool")
        args = action.get("args", {})

        if tool in ["read_file"]:
            provided_path = args.get("file_path")
            if provided_path and isinstance(provided_path, str) and os.path.exists(provided_path):
                args["file_path"] = provided_path
            elif file_path_var and os.path.exists(file_path_var):
                args["file_path"] = file_path_var
            else:
                return {"tool": "read_file", "success": False, "error": "No file is associated with this task. Do not use read_file."}

        try:
            if tool == "web_search":
                return self.web_search(args.get("query"))
            
            elif tool == "visit_website":
                return self.visit_website(args.get("url"))
            
            elif tool == "wikipedia_lookup":
                return self.wikipedia_lookup(args.get("entity"))
            
            elif tool == "read_file":
                return self.read_file(args.get("file_path"))
                
            elif tool == "arxiv_search":
                return self.arxiv_search(args.get("query"))
                
            elif tool == "final_answer":
                return self.final_answer(args.get("answer"), args.get("reasoning"))

            elif tool == "python_exec":
                return self.python_exec(args.get("code"), file_path_var, work_dir=work_dir)
                
            else:
                return {
                    "tool": tool,
                    "success": False,
                    "error": f"Unknown tool: {tool}"
                }
        except Exception as e:
            return {
                "tool": tool,
                "success": False,
                "error": f"Tool execution error: {traceback.format_exc()}"
            }


class Tau2ToolBox(ToolBox):
    """ToolBox subclass exposing the τ²-Bench domain APIs in addition to the
    base toolset."""

    def __init__(self, args):
        super().__init__(args)
        from tau2.registry import registry
        self._tau2_envs = {}
        self._tau2_domain = args.dataset.replace("tau2_", "")

        temp_env = registry.get_env_constructor(self._tau2_domain)()
        tau2_tool_defs = []
        for tool in temp_env.get_tools():
            schema = tool.openai_schema
            func_info = schema.get("function", {})
            params_schema = func_info.get("parameters", {})

            args_dict = {}
            for param_name, param_info in params_schema.get("properties", {}).items():
                param_type = param_info.get("type", "any")
                param_desc = param_info.get("description", "")
                if param_desc:
                    args_dict[param_name] = f"{param_type} - {param_desc}"
                else:
                    args_dict[param_name] = param_type

            tau2_tool_defs.append({
                "name": func_info["name"],
                "description": func_info["description"],
                "args": args_dict,
            })

        # User-side device actions (only present in some domains).
        self._tau2_user_tool_names = set()
        try:
            user_tools = temp_env.get_user_tools()
        except ValueError:
            user_tools = None
        if user_tools:
            for tool in user_tools:
                schema = tool.openai_schema
                func_info = schema.get("function", {})
                params_schema = func_info.get("parameters", {})

                args_dict = {}
                for param_name, param_info in params_schema.get("properties", {}).items():
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    if param_desc:
                        args_dict[param_name] = f"{param_type} - {param_desc}"
                    else:
                        args_dict[param_name] = param_type

                tau2_tool_defs.append({
                    "name": func_info["name"],
                    "description": func_info["description"],
                    "args": args_dict,
                })
                self._tau2_user_tool_names.add(func_info["name"])

        final_answer_entry = self._registry[-1]
        self._registry = tau2_tool_defs + [final_answer_entry]
        self._tau2_tool_names = {t["name"] for t in tau2_tool_defs}

    def _get_or_create_env(self, task_id):
        from tau2.registry import registry
        parts = task_id.split("_")
        domain = parts[1]
        index = int(parts[-1])
        env_key = f"{domain}_{index}"

        if env_key not in self._tau2_envs:
            env = registry.get_env_constructor(domain)()

            tau2_tasks = registry.get_tasks_loader(domain)()
            task = tau2_tasks[index]

            if task.initial_state is not None:
                env.set_state(
                    initialization_data=task.initial_state.initialization_data,
                    initialization_actions=task.initial_state.initialization_actions,
                    message_history=task.initial_state.message_history or [],
                )

            self._tau2_envs[env_key] = {"env": env, "task": task}

        return self._tau2_envs[env_key]

    def tau2_tool_call(self, tool_action, arguments, task_id):
        from tau2.environment.environment import Environment as Tau2Environment
        env_data = self._get_or_create_env(task_id)
        env = env_data["env"]

        try:
            if tool_action in self._tau2_user_tool_names:
                resp = env.use_user_tool(tool_action, **arguments)
            else:
                resp = env.make_tool_call(tool_action, requestor="assistant", **arguments)
            env.sync_tools()
            resp_str = Tau2Environment.to_json_str(resp)
            return {"tool": tool_action, "success": True, "observation": resp_str}
        except Exception as e:
            return {"tool": tool_action, "success": False, "error": str(e)}

    def evaluate_tau2_task(self, task_id):
        from tau2.registry import registry
        env_data = self._get_or_create_env(task_id)
        predicted_env = env_data["env"]
        task = env_data["task"]

        if task.evaluation_criteria is None:
            return True, True, True, True

        domain = task_id.split("_")[1]
        gold_env = registry.get_env_constructor(domain)()

        init_data = None
        init_actions = None
        msg_history = []
        if task.initial_state is not None:
            init_data = task.initial_state.initialization_data
            init_actions = task.initial_state.initialization_actions
            msg_history = task.initial_state.message_history or []

        gold_env.set_state(
            initialization_data=init_data,
            initialization_actions=init_actions,
            message_history=msg_history,
        )

        for action in (task.evaluation_criteria.actions or []):
            try:
                gold_env.make_tool_call(
                    tool_name=action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
            except Exception as e:
                print(f"[Tau2 Eval] Warning: Error in golden action {action.name}: {e}")

        db_match = (gold_env.get_db_hash() == predicted_env.get_db_hash())
        user_db_match = (gold_env.get_user_db_hash() == predicted_env.get_user_db_hash())

        assertion_pass = True
        for assertion in (task.evaluation_criteria.env_assertions or []):
            try:
                if not predicted_env.run_env_assertion(assertion, raise_assertion_error=False):
                    assertion_pass = False
            except Exception:
                assertion_pass = False

        is_correct = bool(db_match and user_db_match and assertion_pass)
        return is_correct, db_match, user_db_match, assertion_pass

    def execute(self, action, file_path_var=None, work_dir=None, task_id=None):
        tool = action.get("tool")
        args = action.get("args", {})

        if tool in self._tau2_tool_names:
            return self.tau2_tool_call(tool, args, task_id)

        return super().execute(action, file_path_var, work_dir, task_id)

