from sys import argv

ckpt_path = argv[1]
print("Using ckpt:", ckpt_path)

from EasyLM.models.gemma.gemma_model import GemmaConfig, FlaxGemmaForCausalLMModule
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.jax_utils import get_float_dtype_by_name, match_partition_rules, make_shard_and_gather_fns, tree_apply, next_rng
from transformers import FlaxGemmaForCausalLM

_, param = StreamingCheckpointer.load_trainstate_checkpoint(load_from=f'trainstate_params::{ckpt_path}')

gemma_config = GemmaConfig.from_pretrained("gemmathon/gemma-2b-pro")
# gemma_config = GemmaConfig.from_pretrained("google/gemma-2b")

# EasyLM Gemma
# model = FlaxGemmaForCausalLMModule(
#     gemma_config, 
#     dtype=get_float_dtype_by_name('bfloat16')
# )

model_ps = match_partition_rules(GemmaConfig.get_partition_rules(), param)
shard_fns, _ = make_shard_and_gather_fns(
    model_ps, get_float_dtype_by_name('bfloat16')
)

mesh = GemmaConfig.get_jax_mesh('1, 1, 1')
with mesh:
    params = tree_apply(shard_fns, param)
    # sharded_rng = next_rng()

auto_model = FlaxGemmaForCausalLM(config=gemma_config) # HF Gemma
auto_model.params = params['params']
auto_model.params = auto_model.to_fp32(auto_model.params)

# 단일 파일로 로드해야 함
auto_model.save_pretrained('./flax-gemma-ko-2b', max_shard_size='999GB')

# HF Flax --> HF SafeTensors
import torch
from transformers import GemmaForCausalLM

hf_model = GemmaForCausalLM.from_pretrained('./flax-gemma-ko-2b/', 
                                            from_flax=True)
hf_model.to(torch.bfloat16)
hf_model.config.torch_dtype = torch.bfloat16
hf_model.save_pretrained('./gemma-ko-2b-dev', max_shard_size='4GB')

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
tokenizer.save_pretrained('./gemma-ko-2b-dev')
