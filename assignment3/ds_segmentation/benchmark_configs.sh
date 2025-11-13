#!/bin/bash
# Benchmark script to run DeepSpeed training with different configurations
# and collect timing results

# Create results directory
RESULTS_DIR="benchmark_results"
mkdir -p "$RESULTS_DIR"

# Get timestamp for this benchmark run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$RESULTS_DIR/benchmark_summary_${TIMESTAMP}.txt"

echo "======================================" | tee "$SUMMARY_FILE"
echo "DeepSpeed Benchmark Run: $TIMESTAMP" | tee -a "$SUMMARY_FILE"
echo "======================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Configuration 1: 1 node, 2 GPUs
echo "[1/3] Submitting: 1 node, 2 GPUs" | tee -a "$SUMMARY_FILE"
JOB1=$(sbatch --parsable \
    --job-name=ds_1n2g \
    --nodes=1 \
    --ntasks=2 \
    --ntasks-per-node=2 \
    --gres=gpu:2 \
    --output="$RESULTS_DIR/ds_1n2g_${TIMESTAMP}.out" \
    --error="$RESULTS_DIR/ds_1n2g_${TIMESTAMP}.err" \
    launch_job_benchmark.sh)
echo "  Job ID: $JOB1" | tee -a "$SUMMARY_FILE"

# Wait for job 1 to complete before submitting job 2
echo "  Waiting for job $JOB1 to complete..." | tee -a "$SUMMARY_FILE"
while squeue -j $JOB1 2>/dev/null | grep -q $JOB1; do
    sleep 10
done
echo "  Job $JOB1 completed" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Configuration 2: 2 nodes, 2 GPUs per node
echo "[2/3] Submitting: 2 nodes, 2 GPUs per node" | tee -a "$SUMMARY_FILE"
JOB2=$(sbatch --parsable \
    --job-name=ds_2n2g \
    --nodes=2 \
    --ntasks=4 \
    --ntasks-per-node=2 \
    --gres=gpu:2 \
    --output="$RESULTS_DIR/ds_2n2g_${TIMESTAMP}.out" \
    --error="$RESULTS_DIR/ds_2n2g_${TIMESTAMP}.err" \
    launch_job_benchmark.sh)
echo "  Job ID: $JOB2" | tee -a "$SUMMARY_FILE"

# Wait for job 2 to complete before submitting job 3
echo "  Waiting for job $JOB2 to complete..." | tee -a "$SUMMARY_FILE"
while squeue -j $JOB2 2>/dev/null | grep -q $JOB2; do
    sleep 10
done
echo "  Job $JOB2 completed" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Configuration 3: 2 nodes, 4 GPUs per node
echo "[3/3] Submitting: 2 nodes, 4 GPUs per node" | tee -a "$SUMMARY_FILE"
JOB3=$(sbatch --parsable \
    --job-name=ds_2n4g \
    --nodes=2 \
    --ntasks=8 \
    --ntasks-per-node=4 \
    --gres=gpu:4 \
    --output="$RESULTS_DIR/ds_2n4g_${TIMESTAMP}.out" \
    --error="$RESULTS_DIR/ds_2n4g_${TIMESTAMP}.err" \
    launch_job_benchmark.sh)
echo "  Job ID: $JOB3" | tee -a "$SUMMARY_FILE"

# Wait for job 3 to complete
echo "  Waiting for job $JOB3 to complete..." | tee -a "$SUMMARY_FILE"
while squeue -j $JOB3 2>/dev/null | grep -q $JOB3; do
    sleep 10
done
echo "  Job $JOB3 completed" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Parse results from all jobs
echo "======================================" | tee -a "$SUMMARY_FILE"
echo "BENCHMARK RESULTS SUMMARY" | tee -a "$SUMMARY_FILE"
echo "======================================" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# Function to extract timing information
extract_timing() {
    local outfile=$1
    local config=$2
    
    if [ -f "$outfile" ]; then
        echo "Configuration: $config" | tee -a "$SUMMARY_FILE"
        
        # Extract total time (last line with "Total training time")
        TOTAL_TIME=$(grep "Total training time" "$outfile" | tail -1)
        if [ -n "$TOTAL_TIME" ]; then
            echo "  $TOTAL_TIME" | tee -a "$SUMMARY_FILE"
        fi
        
        # Extract throughput if available
        THROUGHPUT=$(grep -i "throughput\|samples.*sec" "$outfile" | tail -1)
        if [ -n "$THROUGHPUT" ]; then
            echo "  $THROUGHPUT" | tee -a "$SUMMARY_FILE"
        fi
        
        # Extract final loss/accuracy
        FINAL_METRICS=$(grep -E "Final|Test|Validation" "$outfile" | tail -3)
        if [ -n "$FINAL_METRICS" ]; then
            echo "  Metrics:" | tee -a "$SUMMARY_FILE"
            echo "$FINAL_METRICS" | sed 's/^/    /' | tee -a "$SUMMARY_FILE"
        fi
        
        echo "" | tee -a "$SUMMARY_FILE"
    else
        echo "Configuration: $config - OUTPUT FILE NOT FOUND" | tee -a "$SUMMARY_FILE"
        echo "" | tee -a "$SUMMARY_FILE"
    fi
}

# Extract results for each configuration
extract_timing "$RESULTS_DIR/ds_1n2g_${TIMESTAMP}.out" "1 node, 2 GPUs"
extract_timing "$RESULTS_DIR/ds_2n2g_${TIMESTAMP}.out" "2 nodes, 2 GPUs per node (4 total)"
extract_timing "$RESULTS_DIR/ds_2n4g_${TIMESTAMP}.out" "2 nodes, 4 GPUs per node (8 total)"

echo "======================================" | tee -a "$SUMMARY_FILE"
echo "Benchmark complete!" | tee -a "$SUMMARY_FILE"
echo "Summary saved to: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "Individual results in: $RESULTS_DIR/" | tee -a "$SUMMARY_FILE"
echo "======================================" | tee -a "$SUMMARY_FILE"
