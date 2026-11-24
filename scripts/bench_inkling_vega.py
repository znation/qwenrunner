#!/usr/bin/env python3
"""Offload/engine arms for Inkling 2 Mini, each run under guard.py (system VRAM cap).

usage: bench_inkling_vega.py OUT.jsonl CAP_MIB REPS LABEL:BUILD_DIR:EXTRA ARGS ...
EXTRA ARGS must set -np, -c and -t. Reps are interleaved across arms. An arm the guard kills is recorded as failed.
"""
import json, os, re, subprocess, sys, time, urllib.request

ROOT = "/home/zach/qwenrunner"
PORT = 8083
out, cap, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
arms = [a.split(":", 2) for a in sys.argv[4:]]
text = open(f"{ROOT}/data/ppl_corpus.txt").read()


def sysfs(n):
    return int(open(f"/sys/class/drm/card0/device/mem_info_{n}_used").read()) >> 20


def post(body, timeout=600):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                               json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))


def run(label, build, extra, rep):
    log = f"{os.path.dirname(os.path.abspath(out))}/log_{label}_{rep}.txt"
    cmd = ["python3", f"{ROOT}/scripts/guard.py", "--cap-mib", cap, "--min-avail-mib", "2048",
           "--label", label, "--log", log + ".srv", "--", f"{build}/bin/llama-server",
           "-m", f"{ROOT}/models/Inkling-2-Mini-IQ4_NL.gguf", 
           "--fit", "off", "-ctk", "q4_0", "-ctv", "q4_0", "-fa", "on", "--jinja",
           "--host", "127.0.0.1", "--port", str(PORT)] + extra.split()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{build}/bin:{ROOT}/tools/local/usr/lib/x86_64-linux-gnu"
    row = {"label": label, "rep": rep, "base_vram": sysfs("vram"), "load1": os.getloadavg()[0]}
    with open(log, "w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        peak = gtt = 0
        ok = False
        try:
            t0 = time.time()
            while time.time() - t0 < 240 and p.poll() is None:
                peak, gtt = max(peak, sysfs("vram")), max(gtt, sysfs("gtt"))
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1)
                    ok = True
                    break
                except Exception:
                    time.sleep(0.3)
            if ok:
                post({"prompt": "2 + 2 =", "n_predict": 16, "temperature": 0})
                import threading
                stop = threading.Event()

                def samp():
                    nonlocal peak, gtt
                    while not stop.wait(0.3):
                        peak, gtt = max(peak, sysfs("vram")), max(gtt, sysfs("gtt"))
                th = threading.Thread(target=samp); th.start()
                try:
                    t = post({"prompt": text[:4000], "n_predict": 128, "temperature": 0,
                              "ignore_eos": True, "cache_prompt": False})["timings"]
                    row["text"] = post({"prompt": "Q: Write a Python function that returns the n-th Fibonacci number, and explain it.\nA:", "n_predict": 90, "temperature": 0})["content"]
                    if "-np 2" in extra:
                        import concurrent.futures as cf
                        t2 = time.time()
                        with cf.ThreadPoolExecutor(2) as ex:
                            rs = list(ex.map(lambda o: post({"prompt": text[o:o+4000], "n_predict": 128, "temperature": 0, "ignore_eos": True, "cache_prompt": False})["timings"], [8000, 20000]))
                        row["np2"] = [round(r["predicted_per_second"], 2) for r in rs]; row["np2_wall"] = round(time.time() - t2, 1)
                    row.update(prompt_n=t["prompt_n"], pp=round(t["prompt_per_second"], 2),
                               tg=round(t["predicted_per_second"], 2))
                except Exception as e:
                    row["error"] = repr(e)
                stop.set(); th.join()
            else:
                row["error"] = f"not healthy, exit={p.poll()}"
        finally:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(20)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, 9)
            subprocess.run(["pkill", "-9", "-f", f"llama-server.*--port {PORT}"])
            time.sleep(4)
    s = open(log + ".srv").read() if os.path.exists(log + ".srv") else ""
    row.update(peak_vram=peak, peak_gtt=gtt,
               buf=re.findall(r"(\S+)\s+(model|compute) buffer size =\s+([\d.]+)", s))
    print(json.dumps(row), flush=True)
    open(out, "a").write(json.dumps(row) + "\n")


for rep in range(reps):
    for label, build, extra in arms:
        run(label, build, extra, rep)
