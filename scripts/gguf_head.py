#!/usr/bin/env python3
"""Parse GGUF header + KV + tensor directory from a full file or a byte-range prefix.

Usage:
  python3 scripts/gguf_head.py <file-or-url> [--bytes N] [--grep PATTERN] [--kv]

For a URL, only the first --bytes (default 4 MiB, grown on demand) are fetched
via HTTP Range, so metadata can be inspected without downloading the whole file.
"""
import argparse
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

GGUF_MAGIC = b"GGUF"
# type -> (struct fmt, size)
SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8),
}
GGML_TYPE = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 31: "Q4_0_4_4", 32: "Q4_0_4_8", 33: "Q4_0_8_8",
    34: "TQ1_0", 35: "TQ2_0", 36: "IQ4_NL_4_4",
}


class Truncated(Exception):
    pass


class Reader:
    def __init__(self, buf):
        self.b = buf
        self.p = 0

    def take(self, n):
        if self.p + n > len(self.b):
            raise Truncated()
        s = self.b[self.p:self.p + n]
        self.p += n
        return s

    def u32(self):
        return struct.unpack("<I", self.take(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.take(8))[0]

    def scalar(self, t):
        fmt, sz = SCALAR[t]
        return struct.unpack(fmt, self.take(sz))[0]

    def string(self):
        n = self.u64()
        return self.take(n).decode("utf-8", "replace")

    def value(self, t):
        if t == 8:
            return self.string()
        if t == 9:  # array
            et = self.u32()
            n = self.u64()
            if et == 9:
                raise ValueError("nested array unsupported")
            vals = [self.value(et) for _ in range(n)]
            return vals
        return self.scalar(t)


def fetch(url, nbytes):
    fd, path = tempfile.mkstemp(prefix="ggufhdr.")
    subprocess.run(["curl", "-sL", "-r", f"0-{nbytes - 1}", "-o", path, url], check=True)
    with open(path, "rb") as f:
        data = f.read()
    return data


def parse(buf):
    r = Reader(buf)
    if r.take(4) != GGUF_MAGIC:
        raise SystemExit("not a GGUF file")
    version = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    kv = {}
    for _ in range(n_kv):
        key = r.string()
        t = r.u32()
        kv[key] = r.value(t)
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        ttype = r.u32()
        off = r.u64()
        tensors.append((name, dims, GGML_TYPE.get(ttype, f"?{ttype}"), off))
    return version, kv, tensors, r.p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--bytes", type=int, default=4 << 20)
    ap.add_argument("--grep", default=None)
    ap.add_argument("--kv", action="store_true", help="dump all KV pairs")
    ap.add_argument("--json", default=None, help="write tensor directory as JSON")
    a = ap.parse_args()

    buf = None
    for attempt in range(6):
        n = a.bytes * (2 ** attempt)
        buf = fetch(a.path, n) if a.path.startswith("http") else open(a.path, "rb").read(n)
        try:
            version, kv, tensors, used = parse(buf)
            break
        except Truncated:
            if not a.path.startswith("http"):
                if n >= os.path.getsize(a.path):
                    raise SystemExit("file truncated (whole file read)")
            continue
    else:
        raise SystemExit("could not parse within range budget")

    print(f"version={version} n_tensors={len(tensors)} n_kv={len(kv)} dir_end={used}")
    if a.kv or a.grep:
        for k, v in kv.items():
            if a.grep and not re.search(a.grep, k):
                continue
            s = str(v)
            print(f"  kv {k} = {s[:300]}")
    if a.grep:
        hits = [t for t in tensors if re.search(a.grep, t[0])]
        print(f"-- tensors matching /{a.grep}/: {len(hits)}")
        for name, dims, ttype, off in hits[:80]:
            print(f"   {name} {dims} {ttype} off={off}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump([[n, d, t, o] for n, d, t, o in tensors], f, indent=0)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    sys.exit(main())
