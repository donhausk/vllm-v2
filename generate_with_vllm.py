import ray

from vllm import LLM
from vllm import SamplingParams

# Initialize Ray with explicit resource configuration
# ray.init(address='auto')  # Connects to existing Ray cluster

# ray stop  # Stop existing Ray instance
# ray start --head --node-ip-address=$(hostname -I | awk '{print $1}') --port=6380  # Use different port
# ray.init(address="auto", local_mode=False)
# Define the download directory
import os
from datetime import datetime
from vllm import LLMEngine, SamplingParams
import json
from typing import List, Dict
from pathlib import Path






@dataclass
class Specification:
    sw: int = 128
    sparse: bool = False
    thrs: float = 0.0
    init_window: int = 16
    mode: str = 'oracle'
    layer: int = 0
    ofirst: bool =False
    dense:Optional[List]  = None
    model:str = 'llama3-3'
    static:Optional[Union[str, float]]= None
    static_thrs: float = 0.5
    combined:Optional[float] = None
    tokenizer:Optional[str] = None
    use_hf:bool = False

    data_dir:Optional[str] = None
    task:Optional[str] = None
    save_dir:Optional[str] = None

    seqlen:Optional[int] = 8192

    sparkq_r:int = 32
    sparkq: Optional[int] = None

    moments:Optional[str] = None
    quest:Optional[int] =None
    quest_block_size:int = 128
    subset:Optional[str] = None
    return_types:Optional[str] = ""

    topk:Optional[int] = 0
    STR_subtract: Optional[int] = None
    STR_base_shift: Optional[int] = None
    STR_seqlen: Optional[int] = None
    STR_shift: Optional[int] = None
    STR: Optional[bool] = None
    layerstop: Optional[float] = None

    add_comp_token: bool = False
    
    layerstart:int =0

    layerstop_aggressiveth: Optional[float] = 0.2
    layerstop_baseth: Optional[float] = 0.6
    layerstop_pval: Optional[float] = 0.1
    layerstop_update: Optional[str] = 'oracle'

    def get_folder_name(self) -> str:
        # Generate a folder name based on non-default values
        defaults = Specification().__dict__  # Get default values
        conf_dict = self.config_to_dict()
        filtered_items = {}

        for key, value in conf_dict.items():
            if value is not None and value != defaults[key]:  # Filter out defaults
                if key =='tokenizer': continue
                if isinstance(value, str) and "/" in value:  # If it is a path, take only the last part
                    value = value.split("/")[-1]
                filtered_items[key] = value

        # Generate folder name from the filtered dictionary
        folder_name = "_".join(f"{key}={value}" for key, value in filtered_items.items())
        folder_name = folder_name.replace(" ", "_")
        if len(folder_name) == 0:
            folder_name = 'base'
        return folder_name


    def get_string_representation(self) -> str:
        # Create a string representation using OmegaConf
        conf = OmegaConf.structured(self)
        return OmegaConf.to_yaml(conf)
    def config_to_dict(self) -> dict:
        # Convert the instance to a dictionary using OmegaConf
        conf = OmegaConf.structured(self)
        return OmegaConf.to_container(conf, resolve=True)
    @staticmethod
    def string_to_representation(string: str):
        # Create an instance from a string representation using OmegaConf
        conf = OmegaConf.create(string)
        return OmegaConf.to_object(conf, Specification)






def main(spec):







    # Set up the download directory
    download_dir = "/checkpoint/amaia/explore/konstantind/hf_home/hub"

    # Initialize the LLM model with custom settings
    llm = LLM('deepseek-ai/DeepSeek-R1-Zero',
            download_dir=download_dir,
            tensor_parallel_size=16,
            trust_remote_code=True,
            max_model_len=25_000,
            enforce_eager=True,
            spec=spec)  # Set temperature to 0.7

    # Set up logging
    # Set sampling parameters with 20k max tokens
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=20_000
    )
    os.makedirs('results', exist_ok=True)

    
    output_file = 'results/' + spec.get_folder_name()+".jsonl"
    # Example usage


    prompts = []
    with open('data/gpqa.jsonl', 'r') as f:
        for line in f:
            line_ = json.loads(line.strip())['problem']

            line_['index']
            prompts.append()

            # FILTER all results that are already in the output 
            # Ther eis hte key 

    prompts = prompts[:16]



    output = llm.generate(prompts, sampling_params=sampling_params)
    print(prompts)


    # Get outputs from model

    # Create or append to JSONL file
    with open(output_file, 'a') as f:
        for output in output:
            output_dict = {
                'prompt': output.prompt,
                'generated_text': output.outputs[0].text,
                'token_count': len(output.outputs[0].token_ids),
                'request_id': output.request_id
            }
            # Write each output as a single line of JSON
            f.write(json.dumps(output_dict) + '\n')











if __name__ == '__main__':
    cli_args = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(Specification())
    cfg = OmegaConf.merge(default_cfg, cli_args)
    cfg = OmegaConf.to_object(cfg)
    args = Config(data_dir = cfg.data_dir, task=cfg.task, save_dir=cfg.save_dir)
    cfg.data_dir = None
    cfg.task = None
    cfg.save_dir = None
    with torch.no_grad():
        main(args, cfg)








