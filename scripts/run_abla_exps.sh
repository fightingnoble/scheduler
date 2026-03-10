#!/bin/bash
# Ablation Experiments Runner Script
# 运行三个消融实验

set -e

OUTPUT_DIR=${1:-"./abla_exp_results"}
NUM_HP=${2:-100}

echo "=========================================="
echo "Ablation Experiments"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Number of hyperperiods: $NUM_HP"
echo ""

# Case 1: cyc(S) vs cyc - 预留在串行执行下的影响
echo "Running Ablation-1: cyc(S) vs cyc..."
python -m scripts.abla_exp_runner \
    --case 1 \
    --output_dir "$OUTPUT_DIR" \
    --num_hp $NUM_HP

# Case 2: pglb vs glb - 隔离的作用
echo ""
echo "Running Ablation-2: pglb vs glb..."
python -m scripts.abla_exp_runner \
    --case 2 \
    --output_dir "$OUTPUT_DIR" \
    --num_hp $NUM_HP

# Case 3: reserv vs pglb - 预留在并行下的影响
echo ""
echo "Running Ablation-3: reserv vs pglb..."
python -m scripts.abla_exp_runner \
    --case 3 \
    --output_dir "$OUTPUT_DIR" \
    --num_hp $NUM_HP

echo ""
echo "=========================================="
echo "All ablation experiments completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
