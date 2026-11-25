#!/bin/bash
# Reproduce the case's numbers from scratch.
#
#   ASSETS=/path/to/assets bash reproduce.sh            # judge the tree the numbers belong to
#   ASSETS=/path/to/assets bash reproduce.sh <commit>   # judge another commit of Faster-State
#
# ASSETS must contain:
#   data/dataset/tahoe_slice.h5ad     the 7.4 GB slice   (hf download YHDGD5/state-tahoe-case)
#   data/gene_metadata.parquet
#   env/bin/python                     a python with torch 2.15, lightning, geomloss, cell_load, anndata
#   hf/                                HF_HOME (offline is fine once the data is here)
#
# It lays out the three trees the bench expects, creates the final-evaluation marker that `check --final`
# requires, and runs the judge. The marker is what stops a candidate from asking for a final verdict of its
# own: the harness creates it outside the sandbox, so here it is created explicitly.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
UPSTREAM=9bbfe78a434a55205e4de834e1ea99f85f7a3add
REF_REPO=${REF_REPO:-https://github.com/arcinstitute/state.git}
CAND_REPO=${CAND_REPO:-https://github.com/YHDGD5/Faster-State.git}
COMMIT=${1:-}

: "${ASSETS:?set ASSETS to the directory holding data/, env/ and hf/}"
export ASSETS
cd "$HERE"

[ -x "$ASSETS/env/bin/python" ] || { echo "no $ASSETS/env/bin/python"; exit 1; }
[ -f "$ASSETS/data/dataset/tahoe_slice.h5ad" ] || { echo "no slice under $ASSETS/data/dataset/"; exit 1; }

# The reference tree: bench.json points at targets/reference, and check_reference() refuses anything that is
# not a clean checkout of the frozen upstream commit.
if [ ! -d targets/reference/.git ]; then
  mkdir -p targets
  git clone "$REF_REPO" targets/reference
fi
git -c safe.directory='*' -C targets/reference checkout -q "$UPSTREAM"

# The candidate tree.
if [ ! -d trees/state/.git ]; then
  mkdir -p trees
  git clone "$CAND_REPO" trees/state
fi
git -C trees/state checkout -q "${COMMIT:-main}"
echo "judging $(git -C trees/state rev-parse --short HEAD) ($(git -C trees/state log -1 --format=%s))"

# The marker `check --final` requires. Removed on exit, like the harness does.
MARKER=task/final-evaluation
date +%s > "$MARKER"
trap 'rm -f "$MARKER"' EXIT

python3 bench/bench.py check --final --tree trees/state --out "runs/reproduce-$(date +%m%d_%H%M%S)"
