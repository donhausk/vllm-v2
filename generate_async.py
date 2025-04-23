import os
import json
import asyncio
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from omegaconf import OmegaConf
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.utils import random_uuid
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
from vllm import LLM, LLMEngine, SamplingParams



@dataclass
class Specification:
    twindow: int = 112
    sparse: bool = False
    thrs: float = 0.0
    tinit: int = 16
    mode: str = 'oracle'
    layer: int = 0
    ofirst: bool = False
    dense: Optional[List] = None
    static: Optional[Union[str, float]] = None
    static_thrs: float = 0.5
    combined: Optional[float] = None
    tokenizer: Optional[str] = None
    use_hf: bool = False
    data_dir: Optional[str] = None
    task: Optional[str] = None
    save_dir: Optional[str] = None
    seqlen: Optional[int] = 8192
    sparkq_r: int = 32
    sparkq: Optional[int] = None
    moments: Optional[str] = None
    quest: Optional[int] = None
    quest_block_size: int = 128
    subset: Optional[str] = None
    return_types: Optional[str] = ""
    topk: Optional[int] = 0
    STR_subtract: Optional[int] = None
    STR_base_shift: Optional[int] = None
    STR_seqlen: Optional[int] = None
    STR_shift: Optional[int] = None
    STR: Optional[bool] = None
    layerstop: Optional[float] = None
    add_comp_token: bool = False
    chunk: int = 0
    layerstart: int = 0
    layerstop_aggressiveth: Optional[float] = 0.2
    layerstop_baseth: Optional[float] = 0.6
    layerstop_pval: Optional[float] = 0.1
    layerstop_update: Optional[str] = 'oracle'

    def get_folder_name(self) -> str:
        defaults = Specification().__dict__
        conf_dict = self.config_to_dict()
        filtered_items = {}
        
        for key, value in conf_dict.items():
            if value is not None and value != defaults[key]:
                if key == 'tokenizer':
                    continue
                if isinstance(value, str) and "/" in value:
                    value = value.split("/")[-1]
                filtered_items[key] = value

        folder_name = "_".join(f"{key}={value}" for key, value in filtered_items.items())
        folder_name = folder_name.replace(" ", "_")
        return folder_name if folder_name else 'base'

    def get_string_representation(self) -> str:
        conf = OmegaConf.structured(self)
        return OmegaConf.to_yaml(conf)

    def config_to_dict(self) -> dict:
        conf = OmegaConf.structured(self)
        return OmegaConf.to_container(conf, resolve=True)

    @staticmethod
    def string_to_representation(string: str):
        conf = OmegaConf.create(string)
        return OmegaConf.to_object(conf, Specification)

def load_existing_results(output_file: str) -> set:
    """Load already processed request IDs from the output file."""
    processed_indices = set()
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                if 'index' in data:
                    processed_indices.add(data['index'])
    return processed_indices

def load_prompts(input_file: str, processed_indices: set) -> List[Dict]:
    """Load unprocessed prompts from the input file."""
    prompts = []
    with open(input_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('index') not in processed_indices:
                prompts.append(data)
    return prompts


def prepare_engine(cfg):
    engine_args = AsyncEngineArgs(
        model='meta-llama/Meta-Llama-3-8B',
        download_dir="/cluster/project/yang/donhausk/hf_home",
        tensor_parallel_size=2,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=32768*2,
        spec=cfg,
        pipeline_parallel_size=4,
        gpu_memory_utilization=0.9,
    )
    # Remove await and use synchronous initialization
    return AsyncLLMEngine.from_engine_args(engine_args)




async def batch_process_prompts(
    engine,
    spec,
    input_file: str,
    output_dir: str = 'results',
) -> Dict:
    """
    Batch process prompts using vLLM engine.
    
    Example:
        >>> # Initialize parameters
        >>> model_path = 'deepseek-ai/DeepSeek-R1-Zero'
        >>> input_file = 'data_repeats/gpqa_diamond/chunk_0.jsonl'
        >>> chunk_number = 0
        >>> 
        >>> # Run the batch processing
        >>> results = await batch_process_prompts(
        >>>     model_path=model_path,
        >>>     input_file=input_file,
        >>>     chunk_number=chunk_number
        >>> )
    """
    # Set up output directory and file
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, spec.get_folder_name() + ".jsonl")
    
    # Load existing results and prompts
    processed_indices = load_existing_results(output_file)
    prompts = load_prompts(input_file, processed_indices)
    
    if not prompts:
        return {"message": "No new prompts to process.", "processed_count": 0}
    
    # Initialize sampling parameters
    sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=64*1024,)
    
    # Process prompts
    prompts_in = prompts
    file_write_lock = asyncio.Lock()
    




    async def process_single_prompt(prompt):
        try:
            request_id = random_uuid()
            results_generator = engine.generate(prompt['prompt_messsage_polished'], sampling_params, request_id)
            final_output  = None
            async for output in results_generator:
                final_output = output

            output = final_output

            if final_output is not None:
                result_data = {
                    'prompt': output.prompt,
                    'generated_text': output.outputs[0].text,
                    'token_count': len(output.outputs[0].token_ids),
                    'request_id': output.request_id,
                    'timestamp': str(datetime.now())
                }
                prompt.update(result_data)
                
                async with file_write_lock:
                    with open(output_file, 'a') as f:
                        f.write(json.dumps(prompt) + '\n')
        except Exception as e:
            print(f"Error processing prompt: {e}")
    
    # Process all prompts with progress tracking
    tasks = [process_single_prompt(prompt) for prompt in prompts_in]
    for f in tqdm_asyncio.as_completed(tasks, total=len(tasks), desc="Processing Prompts"):
        await f
    
    return {
        "message": f"Results saved to {output_file}",
        "processed_count": len(prompts_in),
        "output_file": output_file
    }

async def main(engine, cfg):

    results = await batch_process_prompts(
        engine,
        cfg,
        input_file=f'data_repeats/gpqa_diamond/chunk_{cfg.chunk}.jsonl' if not cfg.sparse else f'data_repeats/gpqa_diamond_half/chunk_{cfg.chunk}.jsonl' ,
    )
    print(f"Processing complete:")
    print(f"- Processed count: {results['processed_count']}")
    print(f"- Output file: {results['output_file']}")
    print(f"- Message: {results['message']}")

if __name__ == '__main__':

    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(Specification())
    cfg = OmegaConf.merge(default_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    engine = prepare_engine(cfg)
    """Main function to run the batch processing."""

    print("Starting batch processing...")
    asyncio.run(main(engine, cfg))