#!/bin/bash
# shellcheck disable=SC2153

#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --account=es_ilic
#SBATCH --output=xldumps/R-%x.%j.out
#SBATCH --qos=lowest
#SBATCH --time=5:00:00

chunk=$1
use_sparse=$2

set -x
eval "$(/home/konstantind/miniconda3/bin/conda shell.bash hook)"

conda activate vllm-env
# tune parameters below
MAX_MODEL_LEN=32000
GPU_UTILIZATION=0.9

export HF_HOME="/checkpoint/amaia/explore/konstantind/hf_home"

# Nothing to change below this line
echo "$HF_HOME"
echo "starting the server"

# export RAY_ADDRESS="local"
export RAY_TMPDIR=$(mktemp -d)
export OUTLINES_CACHE_DIR="/scratch/${SLURM_JOB_ID}/outlines_cache"
export TORCHINDUCTOR_LAYOUT_OPTIMIZATION=0

# Getting the node names
mapfile -t nodes_array < <(scontrol show hostnames "$SLURM_JOB_NODELIST")


head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 --overlap -w "$head_node" hostname --ip-address)

SLURM_GPUS_PER_TASK=8

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == " " ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=$(comm -23 <(seq 49152 65535) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | grep "[0-9]\{1,5\}" | sort | uniq) | shuf | head -n 1)
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port="$port" --block &

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 --cpus-per-task=8 -w "$node_i" \
        ray start --address "$ip_head"  --block &
    sleep 5
done
  # --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" -
  #  --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" 

sleep 5

echo "Starting vllm serve process"

if [ "$use_sparse" = true ]; then
    # Generate with sparse and threshold options
    srun --nodes=1 --ntasks=1 --overlap -w "$head_node"  python generate_async.py chunk=${chunk} sparse=true thrs=0.4 twindow=48
else
    # Generate without sparse and threshold options
    srun --nodes=1 --ntasks=1 --overlap -w "$head_node"  python generate_async.py chunk=${chunk}
fi



# srun --nodes=1 --ntasks=1 --overlap -w "$head_node" \
#     vllm serve deepseek-ai/DeepSeek-R1-Zero --enforce-eager --tensor-parallel-size=8 \
#     --pipeline-parallel-size=4 --max-model-len=32000 \
#     --gpu-memory-utilization=0.9

