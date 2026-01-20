# What the CodeAnalysis Analyzer Produces (based on ProjectAnalyzer.run/dfs and key_words_manager.py)

> TL;DR: The analyzer scans and parses every Python file in a project root, guesses a set of probable entry scripts, and from each entry runs a cross-file DFS (call-chain traversal).
>
> When it finds an “AI/inference related terminal call” (defined by `FINAL_CALL_LIST` or object `__call__` rules) at any node in the chain, it records that call path and writes all hit paths, the entry list, debug logs, and “call-file/implementation-file pair data” to the result directory.

---

## 1. How entry files are chosen (`ProjectAnalyzer.run` → `get_start_python_files`)

When you call `ProjectAnalyzer.run()`:

- If `start_file_path` is provided, only that file is analyzed.
- Otherwise it calls `tools/key_words_manager.py:get_start_python_files(project_manager)` to auto-select entries.

### 1.1 Preconditions for candidate entries
Each Python file must meet any of these to enter the candidate list:

- Has a top-level function `main`
- Or contains `if __name__ == '__main__'`
- Or imports `argparse` (`import argparse` or `from ... import argparse`)

### 1.2 Candidate entry ranking
After the preconditions, files are appended by filename keyword groups (deduped in order). Earlier groups have higher priority:

1. Generation scripts: filename contains `gen|synthesize` and also media keywords `audio|video|image|text`
2. Launch/CLI: filename contains `start|launch|run|cli` (`run/cli` via regex `\brun\b` / `\bcli\b`)
3. Inference scripts: filename contains `inference|predict`
4. Media handling: filename contains `img|image|video|audio|text|txt`
5. Examples: filename contains `example|demo|sample`
6. Tests: filename contains `test|unittest`
7. Fallback: only meets the preconditions

Finally `run()` prints:

- `Found N possible entry files for analysis.`

Then it runs DFS for each entry.

---

## 2. What DFS traverses (`ProjectAnalyzer.dfs`)

DFS nodes are not functions but:

- `(PythonFile, entry_method)` pairs

`entry_method` indicates which top-level scope in the file is being analyzed. Formats:

- Top-level function: `func_name`
- Class method: `ClassName.method_name`
- Module level: `""` meaning module entry (displayed as `<module>`)

### 2.1 How call_site is computed
`PythonFile` extracts every call via AST and tags the outermost scope (`parent_name`):

- Inside a top-level function: `parent_name = func_name`
- Inside a class method: `parent_name = ClassName.method_name`
- At module level: `parent_name = <module>`

This `parent_name` becomes `Call.call_site`.

### 2.2 Which calls DFS considers each step
For a node `(node, entry_method)`:

- If `entry_method == ""` (module-level entry):
  - Take `node.calls` (all AST calls in the file, including `<module>`/functions/methods)
  - Add object calls `obj()` converted to `Call(call_name='__call__', call_site='<module>', ...)`
- Otherwise (inside a function/method):
  - Only take calls where `call.call_site == entry_method`
  - Add object calls in the same scope, converted to `__call__`

### 2.3 How DFS jumps across files (resolving where a call is implemented)
For each call:

- If it is a method call (`call.object_name != ""`):
  - Infer possible object types `call.object_type: List[str]`
  - For each `obj_type`, build `method_str = f"{obj_type}.{call.call_name}"`
  - Use `ProjectManager.get_method_impl_pythonfile(method_str, py_file=node)` to locate the implementation file, then recurse

- If it is a function call (`call.object_name == ""`):
  - Use `method_str = call.call_name`
  - Use `get_method_impl_pythonfile(method_str, ...)` to locate the implementation file, then recurse

`ProjectManager.get_method_impl_pythonfile` searches roughly in this order:

1. Look for the definition in the current file
2. If `from x import Name` maps, follow import chains (supports relative imports and `from ... import *` expansion)
3. If still missing, do a rough global search as fallback

---

## 3. When DFS considers a hit (the analysis result you care about)

At every node DFS calls:

- `node.has_target_terminal_call(entry_method)`

If it returns a `Call`, the path is considered a hit and recorded into `call_paths`.

### 3.1 Terminal target 1: match `FINAL_CALL_LIST`
`tools/key_words_manager.py:FINAL_CALL_LIST` is a list of `Call` specs (`call_name/object_name/object_type`) representing AI/inference/LLM endpoints.

Typical entries:

- Common DL inference: `forward/topk/softmax/no_grad/inference_mode`
- Heuristics: `_inference/_predict/_generate` and `encoder.run/decoder.run`
- OpenAI API: `openai.ChatCompletion.create`, `chat.completions.create`
- Agent/LangChain style: `llm.invoke/model.invoke/agent.invoke/chain.invoke`

Match rules (simplified):

- Same scope (`call.call_site == entry_method`)
- `call.call_name == aim_call.call_name`
- If `aim_call.object_name != "ANY"`: substring contains/contained match (e.g., `chat.completions` matches `client.chat.completions`)
- `aim_call.object_type == "ANY"` or type conditions satisfied (types usually from object type inference)

### 3.2 Terminal target 2: object `obj()` (`__call__`)
When the code has `obj()`, the analyzer treats it as a potential model/inference object call.

Hit conditions (any):

- `obj_name` contains keywords: `FINAL_OBJECT_CALL_KEY_WORDS = ['model','detector','classifier','predictor']`
- Or the object type is inferred and its class `base_classes` contains `nn.Module`

Note: Extracting object calls relies on `get_object_call_invocations`, which mainly spots instances created from `from ... import ClassName` imports and then checks if those instances are called with `obj()`. This is heuristic and may miss cases.

---

## 4. Where DFS stops or skips

### 4.1 Calls containing exclusion keywords: `EXCEPT_CALL_PATH_KEY_WORDS`
`tools/key_words_manager.py:EXCEPT_CALL_PATH_KEY_WORDS = ['load','train']`

If `call.call_name` contains these substrings, DFS will not go deeper along that call (it still records other calls).

### 4.2 Max depth: `MAX_CALL_STACK_DEPTH`
`MAX_CALL_STACK_DEPTH = 100`

When the current path hits the limit, recursion stops.

### 4.3 Cycle detection and dead-end cache
DFS keeps:

- `visited` (nodes in the current path) to avoid cycles
- `dead_end_cache` (global) for `(file, entry_method)` that cannot reach a terminal target, enabling pruning

This speeds things up but is heuristic, not a full graph enumeration.

---

## 5. What gets written (output files/directories)

After calling `ProjectAnalyzer.save_analysis_result()`, a project-root-encoded directory is created under `record_base_dir`:

- `record_base_dir + "pj_root" + project_path.replace('/', '_').replace(':','') + '/'`

Inside you will see:

### 5.1 Entry list: possible_entry_files.json
File: `possible_entry_files.json`

It is an array, each element:

```json
{
  "entry_file_path": "/abs/path/to/entry.py",
  "entry_file_index": 3,
  "reason": "filename contains start/launch/run/cli keywords"
}
```

### 5.2 Hits: analysis_results.json
File: `analysis_results.json`

Note: this stores `self.analysis_results["analysis_results"]` (the hit entry list), not the outer dict with `possible_entry_files`.

Each element:

```json
{
  "entry_file_path": "/abs/path/to/entry.py",
  "entry_file_index": 3,
  "reason": "...",
  "call_paths": [
    [
      "/abs/path/to/entry.py <--- <module>",
      "/abs/path/to/some_impl.py <--- SomeClass.some_method",
      "...",
      "/abs/path/to/terminal_node.py <--- Call Matched Method: invoke , Object Name: llm, Possible Types: ['ChatOpenAI']"
    ]
  ]
}
```

#### What call_paths means
- `call_paths` is a 2D array: multiple paths, each a list of strings.
- Regular path elements look like `{file_path} <--- {entry_method}`
- The final element appends a “hit info” string:
  - `... <--- Call Matched Method: {call_name} , Object Name: {object_name}, Possible Types: {object_type_list}`

### 5.3 Project metadata
- `project_root.txt`: analyzed project root path
- `default_name.txt`: default project name (derived from dir name, `-`/`.` replaced with `_`)

### 5.4 Debug logs (optional)
When `record_debug_info=True` and logs exist:

- `debug_logs.txt`

Contains DFS enter/exit, skips, hits, max depth, cache hits, etc.

### 5.5 Per-Python-file summaries
Directory: `project_pyfile_results/`

Generates `*.py_summary.txt` mirroring each `.py`, including:

- imports / from imports
- Top-level function signatures
- Class/method signatures and parents
- (If computed) calls/object_calls lists
- Full source

### 5.6 Call-file / implementation-file pair data (key output)
Directory: `call_implementation_pairs/`

For each hit entry file it writes:

- `entry_idx_{entry_file_index}_{entry_file_basename}.json`

This JSON is an array; each element is pair data for one path (for downstream Inspection/eval):

```json
{
  "call_path_index": 1,
  "entered_from": "/abs/path/to/entry.py",
  "project_root": "/abs/project/root",
  "call_intelligent_path": ["...same as in analysis_results.json..."],
  "call_data": {
    "name": "call_{method_str}",
    "code": "<full source of entry file>",
    "description": "",
    "path": "/abs/path/to/entry.py"
  },
  "implementation_data": {
    "name": "impl_{method_str}",
    "class": "SomeClass" | null,
    "method": "some_method",
    "arguments": ["arg1", "arg2", "kw1", "kw2"],
    "implementation": "<full implementation source + optional base class source>",
    "path": "/abs/path/to/some_impl.py",
    "description": "",
    "example": []
  }
}
```

#### A gotcha (current behavior)
`_save_call_implementation_pairs()` currently treats `call_path[1]` as the “implementation node” (parsed into `impl_python_file_path` and `method`), not the final “Matched Method ...” string.

Implications:

- `implementation_data` corresponds to the **second step** in the path, not the terminal hit.
- If you need the terminal call implementation, adjust logic or post-process using the last element of `call_intelligent_path`.

#### py_data subdirectory
Also generated:

- `call_implementation_pairs/py_data/entry_idx_{idx}_{basename}/path_idx_{n}/call_code.py`
- `call_implementation_pairs/py_data/entry_idx_{idx}_{basename}/path_idx_{n}/implementation_code.py`

These store entry-source and implementation-source (plus optional base class source) with path/name headers.

---

## 6. Config switches in key_words_manager.py (what affects results)

File: `tools/key_words_manager.py`

- `IGNORE_DIRS`: directories skipped during traversal (`.git/.venv/node_modules/dist/build/...`)
- `IGNORE_PYTHON_FILES`: Python filenames to skip (e.g., `TEST.py`)
- `PYTHON_MAX_FILE_SIZE_MB`: `.py` larger than this are skipped
- `MAX_CALL_STACK_DEPTH`: DFS max depth
- `FINAL_CALL_LIST`: terminal target calls (hitting one means AI/inference/LLM chain found)
- `FINAL_OBJECT_CALL_KEY_WORDS`: keyword rule for object `obj()` calls
- `EXCEPT_CALL_PATH_KEY_WORDS`: if call name contains these, do not dive deeper (prune)
- `get_start_python_files(...)`: entry inference

---

## 7. Summary: what the analyzer ultimately outputs

In one line:

- It outputs which entry scripts can reach AI/inference/LLM-related calls, plus one or more cross-file call paths (string stacks) from entry to hit.

Core outputs to disk:

- `possible_entry_files.json`: probable entry scripts (with reasons)
- `analysis_results.json`: entry scripts that actually hit terminal calls via DFS, with `call_paths` per entry

Additionally:

- Optional `debug_logs.txt` and per-file summaries `project_pyfile_results/`
- Downstream-friendly `call_implementation_pairs/*.json` and `py_data/*`

