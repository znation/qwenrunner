#!/usr/bin/env python3
"""How many 128k-context servers can this machine run at once, and does
running more of them help?

The M5 Max has 128 GB of unified memory and a single GPU. Two separate
questions follow, and they have different answers:

  * Memory: each additional server costs its own KV cache plus compute
    buffers. The 15.4 GB of Q4_K_S weights are mapped once and shared through
    the page cache by every process, so N servers is *not* N x 15.4 GB. At
    128k with f16 KV that is ~8 GiB + ~2.5 GiB per server, i.e. the machine
    holds several. This script reports the measured resident set, not the
    arithmetic.
  * Throughput: whether N copies make the GPU do more work. Decode is
    bandwidth-bound -- the whole weight set is streamed per token -- so N
    concurrent streams share one bandwidth budget. The expected shape is
    aggregate throughput roughly flat and per-stream throughput divided by N.
    That is still the right configuration when the goal is *concurrency* (N
    agent sessions that each idle most of the time), and the wrong one when
    the goal is one fast stream.

Usage:
    python3 scripts/parallel_ab.py --instances 1,2,4 --ctx 131072
    python3 scripts/parallel_ab.py --instances 4 --n-predict 256 --json results/par4.json
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import guard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "build", "bin", "llama-server")


def http_json(url, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def wait_health(port, deadline):
    while time.time() < deadline:
        try:
            st = http_json("http://127.0.0.1:%d/health" % port, timeout=5)
            if st.get("status") in ("ok", "no slot available"):
                return True
        except (urllib.error.URLError, OSError, ValueError):
            pass
        time.sleep(1.0)
    return False


def tree_rss_mib(pid):
    """Whole process tree of `pid` in MiB -- llama-server spawns nothing, but
    the guard-style walk keeps this honest if that changes."""
    return guard.tree_rss_kib(pid) / 1024.0


class Fleet:
    def __init__(self, args, n, first_port):
        self.args = args
        self.n = n
        self.mode = args.mode
        # slots mode: one process, one weight set, N slots each with its own
        # CTX of KV (-c is the total, so it scales with N).
        self.ports = [first_port] if self.mode == "slots" else [first_port + i for i in range(n)]
        self.procs = []
        self.logs = []

    def argv_for(self, port):
        args = self.args
        cmd = [SERVER, "-m", args.model, "-ngl", str(args.ngl),
               "-t", str(args.threads), "-b", str(args.batch), "-ub", str(args.ubatch),
               "-ctk", args.ctk, "-ctv", args.ctv, "-fa", args.fa,
               "--host", "127.0.0.1", "--port", str(port),
               "--seed", "42", "-lv", "1"]
        if self.mode == "slots":
            cmd += ["-np", str(self.n), "-c", str(args.ctx * self.n)]
        else:
            cmd += ["-np", "1", "-c", str(args.ctx)]
        if args.mtp:
            cmd += ["--spec-type", "draft-mtp", "-md", args.draft,
                    "-ngld", "99", "--spec-draft-n-max", str(args.nmax)]
        return cmd

    def start(self, timeout):
        """Start servers one at a time, waiting for each to be healthy.

        N concurrent loads of the same file race on the page cache and each
        take longer than they would in sequence; waiting also turns a load
        failure into a single readable log instead of N interleaved ones.
        """
        for port in self.ports:
            # A stale server from an earlier run would answer /health instantly,
            # so the fleet would "load" in 0 s while the new processes die of
            # Metal OOM against memory the stale ones still hold. Fail loudly.
            try:
                st = http_json("http://127.0.0.1:%d/health" % port, timeout=3)
                if st.get("status") in ("ok", "no slot available"):
                    raise SystemExit(
                        "port %d is already serving; kill the stale llama-server "
                        "(pkill -f llama-server) before running this" % port)
            except (urllib.error.URLError, OSError, ValueError):
                pass
            log = os.path.join(ROOT, "logs", "parallel.%d.log" % port)
            self.logs.append(log)
            cmd = self.argv_for(port)
            out_f = open(log, "wb")
            self.procs.append((subprocess.Popen(
                cmd, stdout=out_f, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, cwd=ROOT, start_new_session=True), out_f))
            if not wait_health(port, time.time() + timeout):
                return False
        return True

    def kill(self):
        for p, f in self.procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except OSError:
                pass
        for p, f in self.procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                pass
            f.close()

    def rss_total_mib(self):
        return sum(tree_rss_mib(p.pid) for p, _ in self.procs)


def one_stream(port, prompt, n_predict, timeout):
    t0 = time.time()
    try:
        resp = http_json("http://127.0.0.1:%d/completion" % port,
                         {"prompt": prompt, "n_predict": n_predict,
                          "temperature": 0.0, "seed": 42,
                          "cache_prompt": False, "stream": False},
                         timeout=timeout)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError) as e:
        # A server that has hit the Metal allocation limit answers 500, or dies
        # outright. Record it against that stream instead of aborting the run:
        # "N servers, one of them OOM" is the result we are looking for.
        return {"port": port, "wall_s": round(time.time() - t0, 3),
                "error": "%s: %s" % (type(e).__name__, e)}
    t = resp.get("timings") or {}
    return {"port": port, "wall_s": round(time.time() - t0, 3),
            "predicted_per_second": t.get("predicted_per_second"),
            "prompt_per_second": t.get("prompt_per_second"),
            "predicted_n": t.get("predicted_n"),
            "draft_n": t.get("draft_n"),
            "draft_n_accepted": t.get("draft_n_accepted")}


def measure(args, n):
    fleet = Fleet(args, n, args.port)
    t0 = time.time()
    if not fleet.start(args.load_timeout):
        fleet.kill()
        return {"instances": n, "error": "health timeout",
                "wall_s": round(time.time() - t0, 3)}
    load_s = round(time.time() - t0, 2)
    rss_after_load = fleet.rss_total_mib()
    _, gpu_mib = guard.gpu_stats()

    dead = [port for (p, _), port in zip(fleet.procs, fleet.ports) if p.poll() is not None]
    prompt = open(os.path.join(ROOT, args.prompt_file)).read()
    rec = {"instances": n, "mode": args.mode, "load_s": load_s,
           "ports": fleet.ports,
           "dead_at_measure": dead,
           "rss_total_mib": round(rss_after_load, 1),
           "gpu_mem_mib": None if gpu_mib is None else round(gpu_mib, 1),
           "ctx": args.ctx, "mtp": bool(args.mtp),
           "ctk": args.ctk, "ctv": args.ctv}

    # Single stream: what one client sees when it is the only client.
    rec["single"] = one_stream(fleet.ports[0], prompt, args.n_predict, args.gen_timeout)

    # N concurrent streams: one request per server, all fired together.
    results = [None] * n
    threads = []

    def worker(i):
        # slots mode serves every request from the one port; procs mode spreads
        # them one per server.
        results[i] = one_stream(fleet.ports[i % len(fleet.ports)], prompt,
                                args.n_predict, args.gen_timeout)

    wall0 = time.time()
    for i in range(n):
        th = threading.Thread(target=worker, args=(i,))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    wall = time.time() - wall0

    errors = [r for r in results if r and r.get("error")]
    per = [r.get("predicted_per_second") or 0.0 for r in results if r and not r.get("error")]
    rec["stream_errors"] = len(errors)
    rec["concurrent"] = results
    rec["aggregate_tps"] = round(sum(per), 3)
    rec["per_stream_mean_tps"] = round(sum(per) / len(per), 3) if per else 0.0
    rec["concurrent_wall_s"] = round(wall, 3)
    rec["rss_peak_mib"] = round(max(rss_after_load, fleet.rss_total_mib()), 1)

    fleet.kill()
    time.sleep(1.0)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/Qwen3.8-27B-UD-Q4_K_S.gguf")
    ap.add_argument("--draft", default="models/mtp-Qwen3.8-27B-Q4_0.gguf")
    ap.add_argument("--instances", default="1,2,4",
                    help="comma-separated instance counts to measure in order")
    ap.add_argument("--mode", choices=("procs", "slots"), default="procs",
                    help="procs: N llama-server processes, one request each. "
                         "slots: ONE process with -np N slots (one weight set "
                         "and one set of compute buffers), -c scaled to N x CTX")
    ap.add_argument("--port", type=int, default=8180)
    ap.add_argument("--ctx", type=int, default=131072)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--ubatch", type=int, default=512)
    ap.add_argument("--ctk", default="f16")
    ap.add_argument("--ctv", default="f16")
    ap.add_argument("--fa", default="on")
    ap.add_argument("--mtp", type=int, default=1)
    ap.add_argument("--nmax", type=int, default=6,
                    help="--spec-draft-n-max for the MTP arms")
    ap.add_argument("--n-predict", type=int, default=128)
    ap.add_argument("--prompt-file", default="data/long_prompt.txt")
    ap.add_argument("--load-timeout", type=float, default=1800.0)
    ap.add_argument("--gen-timeout", type=float, default=3600.0)
    ap.add_argument("--json", default="results/parallel.jsonl")
    args = ap.parse_args(argv)

    if not os.path.exists(SERVER):
        raise SystemExit("missing %s (run scripts/build.sh)" % SERVER)

    counts = [int(x) for x in args.instances.split(",") if x.strip()]
    recs = []
    for n in counts:
        print("=== instances=%d ctx=%d mtp=%d" % (n, args.ctx, args.mtp))
        sys.stdout.flush()
        rec = measure(args, n)
        recs.append(rec)
        if rec.get("error"):
            print("    FAILED: %s" % rec["error"])
        else:
            print("    mode=%s rss=%.1f GiB gpu=%.1f GiB load=%.0fs  single=%.2f t/s  "
                  "concurrent: aggregate=%.2f t/s, per-stream=%.2f t/s (wall %.1fs)"
                  % (rec.get("mode", "?"), rec["rss_total_mib"] / 1024.0,
                     0.0 if rec["gpu_mem_mib"] is None else rec["gpu_mem_mib"] / 1024.0,
                     rec["load_s"], rec["single"].get("predicted_per_second") or 0,
                     rec["aggregate_tps"], rec["per_stream_mean_tps"],
                     rec["concurrent_wall_s"]))
        sys.stdout.flush()
        if args.json:
            with open(os.path.join(ROOT, args.json), "a") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")

    print("\n| instances | mode | kv | sum(RSS) GiB | single t/s | aggregate t/s | "
          "per-stream t/s | streams failing |")
    print("|---|---|---|---|---|---|---|---|")
    for rec in recs:
        if rec.get("error"):
            print("| %d | %s | %s | - | - | FAILED: %s | - | - |" % (
                rec["instances"], rec.get("mode", "?"), rec.get("ctk", "?"), rec["error"]))
            continue
        print("| %d | %s | %s | %.1f | %.2f | %.2f | %.2f | %d%s |" % (
            rec["instances"], rec.get("mode", "?"), rec.get("ctk", "?"),
            rec["rss_total_mib"] / 1024.0,
            rec["single"].get("predicted_per_second") or 0,
            rec["aggregate_tps"], rec["per_stream_mean_tps"],
            rec.get("stream_errors", 0),
            " (dead: %s)" % ",".join(map(str, rec["dead_at_measure"]))
            if rec.get("dead_at_measure") else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
