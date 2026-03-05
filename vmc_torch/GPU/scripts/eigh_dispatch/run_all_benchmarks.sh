#!/bin/bash
# Run all 10 benchmark configurations sequentially (GPU can only run one at a time)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$SCRIPT_DIR/test_eigh_full.py"
OUT_DIR="$SCRIPT_DIR/benchmark_output"
mkdir -p "$OUT_DIR"

CLEAN_PY="/home/sijingdu/TNVMC/VMC_code/clean_symmray/bin/python"
DEV_PY="/home/sijingdu/TNVMC/VMC_code/pytorch_dev/bin/python"

# (B, dtype) pairs
configs=("1 float64" "32 float64" "64 float64" "64 float32" "256 float32")

for cfg in "${configs[@]}"; do
    B=$(echo $cfg | cut -d' ' -f1)
    DTYPE=$(echo $cfg | cut -d' ' -f2)
    echo "=== Running B=$B dtype=$DTYPE ==="

    echo "  Without fix..."
    $CLEAN_PY "$SCRIPT" --B $B --dtype $DTYPE > "$OUT_DIR/default_B${B}_${DTYPE}.txt" 2>&1
    echo "  With fix..."
    $DEV_PY "$SCRIPT" --B $B --dtype $DTYPE > "$OUT_DIR/fixed_B${B}_${DTYPE}.txt" 2>&1

    echo "  Done."
done

echo "All benchmarks complete. Output in $OUT_DIR/"
