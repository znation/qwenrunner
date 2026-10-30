#!/usr/bin/env python3
"""fleet_ab.py — interleaved A/B of server configurations on the agentic turn loop.

Why a driver and not a shell loop: the box is shared, and FLINT.md item 5
measured up to 4x drift between windows of the *same* configuration. A
comparison is only readable if the arms are interleaved (A-B-A-B, repeats) and
each arm's window is characterised (load average, GPU busy, RSS peak). This
starts one `serve_flint.sh` per arm, waits for health, runs
`turn_loop_bench.py` against it, samples the server's RSS while it runs, and
records the arm's env plus the bench's own summary line.

Arms are given on the command line as `label:ENV=VAL,ENV=VAL,...[:sessions]`
where ENV are the variables `serve_flint.sh` reads (NP, CTX, BATCH, UBATCH,
CTK, CTV, SPEC, NMIN, ...) plus the special key `sessions` (turn-loop client
count) and `extra` (verbatim extra args, `+`-separated to survive the comma).

    # 4-session baseline vs 8 sessions on 8 slots, two interleaved reps
    python3 scripts/fleet_ab.py --reps 2 --turns 6 \
        --arm "np4:sessions=4" --arm "np8:NP=8,sessions=8"

Results append to results/fleet_ab.jsonl (turn-loop summaries, labelled) and
results/fleet_ab.meta.jsonl (per-arm window: load, RSS peak, wall). Stdlib only.
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
SERVE = os.path.join(ROOT, "scripts", "serve_flint.sh")
TURN = os.path.join(ROOT, "scripts", "turn_loop_bench.py")


def http_health(port, timeout=3):
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % port,
                                    timeout=timeout) as r:
            return json.loads(r.read().decode()).get("status")
    except (urllib.error.URLError, OSError, ValueError):
        return None


def wait_health(port, deadline):
    while time.time() < deadline:
        st = http_health(port)
        if st in ("ok", "no slot available"):
            return True
        time.sleep(1.0)
    return False


def parse_arm(spec):
    """'label:KEY=VAL,KEY=VAL,sessions=4,extra=-a+--b' -> (label, env, sessions, extra)"""
    label, _, kv = spec.partition(":")
    env, extra, sessions = {}, [], 4
    for item in kv.split(","):
        if not item:
            continue
        k, _, v = item.partition("=")
        if k == "sessions":
            sessions = int(v)
        elif k == "extra":
            extra = v.split("+")
        else:
            env[k] = v
    return label, env, sessions, extra


def load_avg():
    try:
        return os.getloadavg()
    except OSError:
        return (None, None, None)


class Server:
    def __init__(self, env, port, log):
        self.env = dict(os.environ)
        self.env.update(env)
        self.env["PORT"] = str(port)
        self.port = port
        self.log = log
        self.proc = None
        self.f = None
        self.rss_peak_mib = 0.0
        self._stop = threading.Event()
        self._sampler = None

    def start(self, timeout):
        self.f = open(self.log, "wb")
        self.proc = subprocess.Popen([SERVE], env=self.env, stdout=self.f,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=ROOT, start_new_session=True)
        if not wait_health(self.port, time.time() + timeout):
            return False
        return True

    def _sample(self):
        while not self._stop.wait(2.0):
            try:
                rss = guard.tree_rss_kib(self.proc.pid) / 1024.0
                self.rss_peak_mib = max(self.rss_peak_mib, rss)
            except Exception:  # noqa: BLE001
                pass

    def begin_sampling(self):
        self._sampler = threading.Thread(target=self._sample, daemon=True)
        self._sampler.start()

    def stop(self):
        self._stop.set()
        if self._sampler:
            self._sampler.join(timeout=5)
        if self.proc and self.proc.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                self.proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                pass
        # the shell can exit while guard.py/llama-server survive it (observed after
        # an arm whose server had just loaded), so wait for the port to go quiet and
        # escalate rather than leaving a stale server for the next run to trip over
        if not self._wait_down(20):
            self._signal_group(signal.SIGKILL)
            if not self._wait_down(20):
                print("fleet_ab: warning: %s still answering /health after SIGKILL"
                      % self.log, flush=True)
        if self.f:
            self.f.close()

    def _signal_group(self, sig):
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except OSError:
            pass

    def _wait_down(self, seconds):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if http_health(self.port) is None:
                return True
            time.sleep(1.0)
        return False


def run_turn_loop(args, label, sessions, port):
    cmd = [sys.executable, TURN,
           "--base-url", "http://127.0.0.1:%d/v1" % port,
           "--model", args.model, "--label", label,
           "--sessions", str(sessions), "--turns", str(args.turns),
           "--max-tokens", str(args.max_tokens),
           "--temperature", str(args.temperature),
           "--corpus", args.corpus,
           "--initial-chars", str(args.initial_chars),
           "--obs-chars", str(args.obs_chars),
           "--out", args.out]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.turn_timeout)
    return {"rc": p.returncode, "wall_s": round(time.time() - t0, 2),
            "stdout": p.stdout[-4000:], "stderr": p.stderr[-2000:]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="label:ENV=VAL,... (repeatable; run interleaved)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--model", default="flint")
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--corpus", default="data/prompt_100k.txt")
    ap.add_argument("--initial-chars", type=int, default=12000)
    ap.add_argument("--obs-chars", type=int, default=2400)
    ap.add_argument("--out", default="results/fleet_ab.jsonl")
    ap.add_argument("--meta-out", default="results/fleet_ab.meta.jsonl")
    ap.add_argument("--load-timeout", type=float, default=1800.0)
    ap.add_argument("--turn-timeout", type=float, default=3600.0)
    ap.add_argument("--idle-gpu", type=float, default=None,
                    help="if set, wait for GPU busy%% below this before each arm "
                         "(uses wait_idle.py --gpu)")
    args = ap.parse_args(argv)

    arms = [parse_arm(a) for a in args.arm]
    print("fleet_ab: %d arms x %d reps; arms=%s"
          % (len(arms), args.reps, [a[0] for a in arms]), flush=True)

    meta = []
    for rep in range(1, args.reps + 1):
        # alternate the arm order every rep: the box's ambient load drifts over a
        # long run, and a fixed order would charge that drift to whichever arm
        # happens to run late
        order = arms if rep % 2 == 1 else list(reversed(arms))
        for label, env, sessions, extra in order:
            if args.idle_gpu is not None:
                subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wait_idle.py"),
                                "--gpu", "--threshold", str(args.idle_gpu),
                                "--hold", "5", "--timeout", "900"],
                               cwd=ROOT, capture_output=True)
            if http_health(args.port) is not None:
                # a just-stopped arm's server can still be tearing down; give it a
                # minute before treating this as someone else's server on the port
                for _ in range(30):
                    time.sleep(2.0)
                    if http_health(args.port) is None:
                        break
                if http_health(args.port) is not None:
                    raise SystemExit("port %d is already serving; kill the stale server" % args.port)
            tag = "%s:rep%d" % (label, rep)
            log = os.path.join(ROOT, "logs", "fleet_ab.%s.log" % tag.replace(":", "_"))
            srv = Server(env, args.port, log)
            la0 = load_avg()
            t0 = time.time()
            if not srv.start(args.load_timeout):
                srv.stop()
                rec = {"kind": "fleet_ab_meta", "label": tag, "error": "health timeout",
                       "load_before": la0, "wall_s": round(time.time() - t0, 2)}
                meta.append(rec)
                print(json.dumps(rec), flush=True)
                continue
            load_s = round(time.time() - t0, 2)
            srv.begin_sampling()
            _, gpu_mib = guard.gpu_stats()
            try:
                run = run_turn_loop(args, tag, sessions, args.port)
            finally:
                srv.stop()
            rec = {"kind": "fleet_ab_meta", "label": tag, "arm": label,
                   "env": env, "sessions": sessions, "load_s": load_s,
                   "load_before": la0, "load_after": load_avg(),
                   "rss_peak_mib": round(srv.rss_peak_mib, 1),
                   "gpu_mem_mib": None if gpu_mib is None else round(gpu_mib, 1),
                   "turn_wall_s": run["wall_s"], "rc": run["rc"]}
            meta.append(rec)
            print(json.dumps(rec), flush=True)
            if run["rc"] != 0:
                print(run["stdout"][-1500:], run["stderr"][-800:], flush=True)
            time.sleep(3.0)

    with open(os.path.join(ROOT, args.meta_out), "a") as f:
        for r in meta:
            f.write(json.dumps(r) + "\n")
    print("fleet_ab: wrote %d meta records to %s" % (len(meta), args.meta_out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
