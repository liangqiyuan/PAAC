import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr


def get_exec_globals():
    # Must stay in sync with utils.get_exec_globals.
    import math
    import random
    import re
    import datetime
    import time
    import json as _json
    import subprocess
    import requests

    return {
        "math": math,
        "os": os,
        "sys": sys,
        "re": re,
        "random": random,
        "datetime": datetime,
        "time": time,
        "json": _json,
        "requests": requests,
        "subprocess": subprocess,
    }


def main():
    work_dir = sys.argv[1] if len(sys.argv) > 1 else ""
    code_str = sys.stdin.read()

    if work_dir:
        os.makedirs(work_dir, exist_ok=True)
        os.chdir(work_dir)

    file_path = os.environ.get("FILE_PATH") or None
    exec_globals = get_exec_globals()
    if file_path:
        exec_globals["FILE_PATH"] = file_path

    output_buffer = io.StringIO()
    execution_scope = exec_globals.copy()
    result = {"success": False, "output": "", "error": "", "local_vars": "{}"}
    initial_keys = set(exec_globals.keys())

    with redirect_stdout(output_buffer), redirect_stderr(output_buffer):
        try:
            exec(code_str, execution_scope, execution_scope)
            result["success"] = True
        except BaseException:
            result["error"] = traceback.format_exc()
        finally:
            result["output"] = output_buffer.getvalue()
            display_vars = {
                k: v for k, v in execution_scope.items()
                if k not in initial_keys
                and type(v).__name__ != "module"
                and not k.startswith("__")
            }
            result["local_vars"] = str(display_vars)

    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
