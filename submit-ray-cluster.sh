# salloc --nodes 1  --gpus=1 --mem-per-cpu=20g --gres=gpumem:40g --time=16:00:00  --account=es_ilic

source ~/.bashrc
conda activate vllm-env
# tune parameters below
MAX_MODEL_LEN=32000
GPU_UTILIZATION=0.9



# export HF_HOME="/checkpoint/amaia/explore/konstantind/hf_home"

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

SLURM_GPUS_PER_TASK=2

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


sleep 5

# # number of nodes other than the head node
# worker_num=$((SLURM_JOB_NUM_NODES - 1))

# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"
#     srun --nodes=1 --ntasks=1  -w $node_i \
#         ray start --address "$ip_head"  --block &
#     sleep 5
# done
# sleep 5

# echo "Starting vllm serve process"

