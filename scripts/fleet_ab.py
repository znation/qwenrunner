#!/usr/bin/env python3
"""fleet_ab.py — interleaved A/B of server configurations on the agentic turn loop.

Why a driver and not a shell loop: the box is shared, and RESULTS.md records up
to 4x drift between windows of the *same* configuration. A comparison is only
readable if the arms are interleaved (A-B-A-B, repeats) and each arm's window is
characterised (load average, GPU busy, RSS peak). This starts one `serve.sh` per
arm, waits for health, runs `turn_loop_bench.py` against it, samples the
server's RSS while it runs, and records the arm's env plus the bench's own
summary line.

Arms are given on the command line as `label:ENV=VAL,ENV=VAL,...[:sessions]`
where ENV are the variables `serve.sh` reads (NP, INSTANCES, CTX, BATCH,
UBATCH, CTK, CTV, MTP, NMAX, CRAM, ...) plus the special key `sessions`
(turn-loop client count) and `extra` (verbatim extra args, `+`-separated to
survive the comma).

    # 4-session baseline vs 8 sessions on 8 slots, two interleaved reps
    python3 scripts/fleet_ab.py --reps 2 --turns 4 \
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
SERVE = os.path.join(ROOT, "serve.sh")
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
        # serve.sh launches INSTANCES servers on consecutive ports; every one has
        # to be healthy before the arm starts, or the sessions dealt to the lagging
        # instance pay its load time as TTFT
        self.ports = [port + i for i in range(max(1, int(env.get("INSTANCES", 1))))]
        self.log = log
        self.proc = None
        self.f = None
        self.rss_peak_mib = 0.0
        self._stop = threading.Event()
        self._sampler = None

    def start(self, timeout):
        # per-arm server logs: guard.py's log is the only place the per-request
        # prompt/eval timings and the draft acceptance land, and the default path
        # is one file per port that the next arm overwrites
        log_dir = os.path.splitext(self.log)[0]
        os.makedirs(log_dir, exist_ok=True)
        self.env["LOG_DIR"] = log_dir
        self.f = open(self.log, "wb")
        self.proc = subprocess.Popen([SERVE], env=self.env, stdout=self.f,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=ROOT, start_new_session=True)
        if not all(wait_health(p, time.time() + timeout) for p in self.ports):
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
            if all(http_health(p) is None for p in self.ports):
                return True
            time.sleep(1.0)
        return False


def run_turn_loop(args, label, sessions, ports):
    cmd = [sys.executable, TURN,
           "--base-url", "http://127.0.0.1:%d/v1" % ports[0],
           "--model", args.model, "--label", label,
           "--sessions", str(sessions), "--turns", str(args.turns),
           "--max-tokens", str(args.max_tokens),
           "--temperature", str(args.temperature),
           "--corpus", args.corpus,
           "--initial-chars", str(args.initial_chars),
           "--obs-chars", str(args.obs_chars),
           "--out", args.out]
    if len(ports) > 1:
        cmd += ["--base-urls", ",".join("http://127.0.0.1:%d/v1" % p for p in ports)]
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.turn_timeout)
    return {"rc": p.returncode, "wall_s": round(time.time() - t0, 2),
            "stdout": p.stdout[-4000:], "stderr": p.stderr[-2000:]}


def kill_orphan_servers(port):
    """Kill a leftover server on `port` when its launch shell is gone.

    A sweep killed by the caller's timeout leaves serve.sh's process group behind
    (guard.py + llama-server, reparented), and the next run then sees "port is
    already serving" and aborts. Only touch a group with no live `serve.sh` in it,
    so a server the user started by hand still wins the port.
    """
    try:
        out = subprocess.run(["lsof", "-t", "-iTCP:%d" % port, "-sTCP:LISTEN"],
                             capture_output=True, text=True, timeout=20).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return False
    killed = False
    for pid in out:
        try:
            info = subprocess.run(["ps", "-o", "pgid=,command=", "-p", pid],
                                  capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        pgid, _, cmd = info.partition(" ")
        if "llama-server" not in cmd:
            continue
        try:
            group = subprocess.run(["ps", "-o", "command=", "-g", pgid.strip()],
                                   capture_output=True, text=True, timeout=20).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        if "serve.sh" in group:
            continue  # a live launcher owns it -- not ours to kill
        print("fleet_ab: killing orphaned server pid %s (pgid %s) on port %d"
              % (pid, pgid.strip(), port), flush=True)
        try:
            os.killpg(int(pgid), signal.SIGTERM)
            killed = True
        except (OSError, ValueError):
            pass
    return killed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True,
                    help="label:ENV=VAL,... (repeatable; run interleaved)")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--model", default="qwen3.8-27b")
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

    def write_meta(rec):
        # written per arm, not at the end: a sweep that is interrupted (or hits the
        # caller's timeout) otherwise loses every arm's RSS and window record
        with open(os.path.join(ROOT, args.meta_out), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)

    for rep in range(1, args.reps + 1):
        # alternate the arm order every rep: the box's ambient load drifts over a
        # long run, and a fixed order would charge that drift to whichever arm
        # happens to run late
        order = arms if rep % 2 == 1 else list(reversed(arms))
        for label, env, sessions, extra in order:
            if args.idle_gpu is not None:
                subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "wait_idle.py"),
                                "--gpu", "--idle-gpu-pct", str(args.idle_gpu),
                                "--hold", "5", "--timeout", "900"],
                               cwd=ROOT, capture_output=True)
            if http_health(args.port) is not None:
                # a just-stopped arm's server can still be tearing down; give it a
                # minute before treating this as someone else's server on the port
                for _ in range(30):
                    time.sleep(2.0)
                    if http_health(args.port) is None:
                        break
                if http_health(args.port) is not None and kill_orphan_servers(args.port):
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
                write_meta(rec)
                continue
            load_s = round(time.time() - t0, 2)
            srv.begin_sampling()
            _, gpu_mib = guard.gpu_stats()
            try:
                run = run_turn_loop(args, tag, sessions, srv.ports)
            finally:
                srv.stop()
            rec = {"kind": "fleet_ab_meta", "label": tag, "arm": label,
                   "env": env, "sessions": sessions, "load_s": load_s,
                   "load_before": la0, "load_after": load_avg(),
                   "rss_peak_mib": round(srv.rss_peak_mib, 1),
                   "gpu_mem_mib": None if gpu_mib is None else round(gpu_mib, 1),
                   "turn_wall_s": run["wall_s"], "rc": run["rc"]}
            write_meta(rec)
            if run["rc"] != 0:
                print(run["stdout"][-1500:], run["stderr"][-800:], flush=True)
            time.sleep(3.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
