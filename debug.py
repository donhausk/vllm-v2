import os
import json
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from pathlib import Path
from omegaconf import OmegaConf
from vllm import LLM, LLMEngine, SamplingParams



def main():
    # Set up the download directory
    download_dir = "/cluster/project/yang/donhausk/hf_home"
    max_seq_len = 32768

    # Initialize the LLM model with custom settings
    llm = LLM(
        model='Qwen/Qwen2.5-7B-Instruct',
        download_dir=download_dir,
        tensor_parallel_size=2,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len= max_seq_len,
    )

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_seq_len,
    )
    # # Create output directory

    os.makedirs('results', exist_ok=True)

    # Load and filter prompts
    prompts_in = ["This is a test prompt"]
    # Generate outputs
    outputs = llm.generate(prompts_in, sampling_params=sampling_params, use_tqdm=True)

    # Save results
    print(outputs)

if __name__ == '__main__':
    # Parse command line arguments

    with torch.no_grad():
        main()