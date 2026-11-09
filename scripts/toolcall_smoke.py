#!/usr/bin/env python3
"""Does the server speak the OpenAI tool-calling dialect pi relies on?

Speed is worthless to tumwater if pi cannot parse the reply, so every engine
that wins scripts/tumwater_replay.py has to pass this too. Three requests, all
streaming (pi always streams):

  1. a task that needs the `bash` tool -> expect delta.tool_calls with a name and
     arguments that parse as JSON containing "command"
  2. the same transcript with the tool result appended -> expect plain content
  3. a task that needs two independent reads -> expect >= 1 tool call (parallel
     calls are a bonus, reported but not required)

Also reports whether reasoning arrives in a separate field (reasoning_content /
reasoning) rather than inline <think> text, which pi's strip-thinking needs.
Exit status 0 only if 1 and 2 pass. Stdlib only.
"""

import argparse
import json
import sys
import urllib.request

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Execute a bash command and return its output.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "the command to run"},
            "timeout": {"type": "number"}}, "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "read", "description": "Read a file's contents.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
]
SYSTEM = ("You are a coding assistant running inside pi, a coding agent harness. "
          "Use the tools to inspect files and run commands.")


def stream(url, model, messages, max_tokens, key):
    body = {"model": model, "messages": messages, "tools": TOOLS, "max_tokens": max_tokens,
            "stream": True, "temperature": 0.6, "top_p": 0.95}
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h)
    content, reasoning, calls, finish = [], [], {}, None
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            try:
                ev = json.loads(line[5:])
            except ValueError:
                continue
            for ch in ev.get("choices") or []:
                finish = ch.get("finish_reason") or finish
                d = ch.get("delta") or {}
                content.append(d.get("content") or "")
                reasoning.append(d.get("reasoning_content") or d.get("reasoning") or "")
                for tc in d.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                    c["id"] = tc.get("id") or c["id"]
                    f = tc.get("function") or {}
                    c["name"] += f.get("name") or ""
                    c["args"] += f.get("arguments") or ""
    return "".join(content), "".join(reasoning), [calls[k] for k in sorted(calls)], finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--max-tokens", type=int, default=2048)
    a = ap.parse_args()
    url = a.base_url.rstrip("/") + "/chat/completions"
    ok = True

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "How many lines are in src/index.ts? Use a tool."}]
    text, reas, calls, fin = stream(url, a.model, msgs, a.max_tokens, a.api_key)
    good = False
    for c in calls:
        try:
            good = good or (c["name"] == "bash" and "command" in json.loads(c["args"] or "{}"))
        except ValueError:
            pass
    inline_think = "<think>" in text
    print("1 tool call: %s  calls=%s finish=%s reasoning_field=%s inline_think=%s"
          % ("PASS" if good else "FAIL", [(c["name"], c["args"][:80]) for c in calls], fin,
             bool(reas.strip()), inline_think))
    ok &= good

    if calls:
        c = calls[0]
        cid = c["id"] or "call_0"
        msgs += [{"role": "assistant", "content": text,
                  "tool_calls": [{"id": cid, "type": "function",
                                  "function": {"name": c["name"], "arguments": c["args"]}}]},
                 {"role": "tool", "tool_call_id": cid, "content": "212 src/index.ts"}]
        text2, _, calls2, fin2 = stream(url, a.model, msgs, a.max_tokens, a.api_key)
        good2 = "212" in text2
        print("2 tool result -> answer: %s  finish=%s text=%r"
              % ("PASS" if good2 else "FAIL", fin2, text2.strip()[:120]))
        ok &= good2
    else:
        print("2 skipped (no call in 1)")

    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Read both package.json and tsconfig.json, then tell me "
                                        "the TypeScript target. Read them in parallel."}]
    _, _, calls3, fin3 = stream(url, a.model, msgs, a.max_tokens, a.api_key)
    print("3 parallel reads: %d call(s) %s finish=%s"
          % (len(calls3), [(c["name"], c["args"][:60]) for c in calls3], fin3))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
