from transformers import FlaxGemmaForCausalLM, GemmaForCausalLM

pytorch_model = GemmaForCausalLM.from_pretrained("gemmathon/gemma-2b-pro-layer24")
pytorch_model.model.save_pretrained('./gemma.pt', safe_serialization=False)
flax_model = FlaxGemmaForCausalLM.from_pretrained('./gemma.pt', from_pt=True)
flax_model.params = flax_model.to_bf16(flax_model.params)
flax_model.save_pretrained("./flax-concatted", max_shard_size="99GB")