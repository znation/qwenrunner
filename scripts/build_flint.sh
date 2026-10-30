#!/usr/bin/env bash
# Build llama.cpp *with Flint support*, into third_party/llama-flint-build.
#
# Why this is a separate build and not a re-pin:
#
# The pinned checkout (third_party/llama.cpp @ aa5e0092, the revision every
# number in RESULTS.md was measured on) does not know the Flint architecture at
# all -- `flint` is absent from LLM_ARCH_NAMES there, so llama.cpp refuses the
# GGUF outright ("unknown model architecture"). Support landed upstream as PR
# #30928, "models: add Flint model (FlintForCausalLM)", merge commit
# a37cc46104eb50900d5ab4a9ba02a0abdac039db (2026-10-21): a new
# llama_model_flint (2 dense + 47 MoE layers, 256 experts / 6 used + 2 shared,
# per-head QK RMSNorm, attention sinks, sliding-window pattern, and a rope
# pattern that leaves the 14 full-attention layers unrotated), a `flint` BPE
# pre-tokenizer, and a K-shift guard for layers without RoPE.
#
# Moving the pin would invalidate that evidence, so instead this adds a *second*
# checkout of the same clone as a git worktree (third_party/llama-flint) at the
# merge commit, and builds it with the identical cmake flags -- which is why
# scripts/build.sh now takes LLAMA_DIR / BUILD_DIR / LOG_DIR from the
# environment. Both checkouts stay usable: ./scripts/build.sh still builds the
# pinned Qwen revision into build/, this one builds Flint into
# third_party/llama-flint-build/.
#
# Everything lands under third_party/ and models/, both gitignored.
#
#   ./scripts/build_flint.sh              # idempotent: reuses the worktree
#   REV=<sha> ./scripts/build_flint.sh    # any revision with Flint support
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/third_party/llama.cpp"
WORKTREE="$ROOT/third_party/llama-flint"
REV="${REV:-a37cc46104eb50900d5ab4a9ba02a0abdac039db}"

if [ ! -d "$SRC/.git" ]; then
  echo "build_flint: $SRC is missing. Run:" >&2
  echo "build_flint:   git clone https://github.com/ggml-org/llama.cpp $SRC" >&2
  exit 1
fi

if [ ! -e "$WORKTREE/.git" ]; then
  if ! git -C "$SRC" cat-file -e "$REV^{commit}" 2>/dev/null; then
    echo "build_flint: fetching $REV"
    git -C "$SRC" fetch --filter=blob:none origin "$REV" 2>/dev/null \
      || git -C "$SRC" fetch --filter=blob:none origin master
  fi
  echo "build_flint: adding worktree $WORKTREE at $REV"
  git -C "$SRC" worktree add --detach "$WORKTREE" "$REV"
fi

have="$(git -C "$WORKTREE" rev-parse HEAD)"
if [ "$have" != "$REV" ]; then
  echo "build_flint: worktree is at $have, switching to $REV"
  git -C "$WORKTREE" checkout --detach "$REV"
fi

# Local patches. The worktree is a pinned upstream revision, so anything this repo
# needs on top of it lives as a tracked patch in scripts/patches/ and is applied
# here, idempotently: a worktree that already carries a patch is left alone.
for patch in "$ROOT"/scripts/patches/*.patch; do
  [ -e "$patch" ] || continue
  if git -C "$WORKTREE" apply --reverse --check "$patch" 2>/dev/null; then
    echo "build_flint: already applied $(basename "$patch")"
  elif git -C "$WORKTREE" apply --check "$patch" 2>/dev/null; then
    git -C "$WORKTREE" apply "$patch"
    echo "build_flint: applied $(basename "$patch")"
  else
    echo "build_flint: cannot apply $(basename "$patch") -- worktree is dirty or the pin moved?" >&2
    exit 1
  fi
done

echo "build_flint: llama.cpp at $(git -C "$WORKTREE" log --oneline -1)"
# Build logs go inside the (gitignored) build directory on purpose: logs/ is
# tracked in this repo as measurement evidence, and logs/build.log documents the
# pinned Qwen build.
LLAMA_DIR="$WORKTREE" \
BUILD_DIR="${BUILD_DIR:-$ROOT/third_party/llama-flint-build}" \
LOG_DIR="${LOG_DIR:-$ROOT/third_party/llama-flint-build/logs}" \
  exec "$ROOT/scripts/build.sh"
