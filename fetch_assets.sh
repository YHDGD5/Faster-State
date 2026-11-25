#!/bin/bash
# Download the judge's frozen inputs from Hugging Face and verify them.
set -euo pipefail
cd "$(dirname "$0")"
REPO=${REPO:-YHDGD5/state-tahoe-case}
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p assets/data/dataset
python -m huggingface_hub.commands.huggingface_cli download "$REPO" \
  tahoe_slice.h5ad gene_metadata.parquet fixture.pt --local-dir .hf 2>/dev/null \
  || hf download "$REPO" tahoe_slice.h5ad gene_metadata.parquet fixture.pt --local-dir .hf
mv -f .hf/tahoe_slice.h5ad assets/data/dataset/tahoe_slice.h5ad
mv -f .hf/gene_metadata.parquet assets/data/gene_metadata.parquet
mv -f .hf/fixture.pt bench/fixture.pt
rmdir .hf 2>/dev/null || true
sha256sum -c SHA256SUMS
echo "assets OK"
