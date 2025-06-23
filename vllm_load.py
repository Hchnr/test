from safetensors import safe_open
from vllm.config import VllmConfig
from vllm.model_executor.models.ernie import ErnieForCausalLM


MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split/model.safetensors"

vllm_config = VllmConfig(
    model=MODEL_PATH,
    tensor_parallel_size=1,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    trust_remote_code=True,
)    

# VllmConfig(model='/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split', speculative_config=None, tokenizer='/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split', skip_tokenizer_init=False, tokenizer_mode=auto, revision=None, override_neuron_config={}, tokenizer_revision=None, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=40960, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, disable_custom_all_reduce=False, quantization=None, enforce_eager=True, kv_cache_dtype=auto,  device_config=cuda, decoding_config=DecodingConfig(backend='auto', disable_fallback=False, disable_any_whitespace=False, disable_additional_properties=False, reasoning_backend=''), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None), seed=0, served_model_name=/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split, num_scheduler_steps=1, multi_step_stream_outputs=True, enable_prefix_caching=True, chunked_prefill_enabled=True, use_async_output_proc=False, pooler_config=None, compilation_config={"level":0,"debug_dump_path":"","cache_dir":"","backend":"","custom_ops":[],"splitting_ops":[],"use_inductor":true,"compile_sizes":[],"inductor_compile_config":{"enable_auto_functionalized_v2":false},"inductor_passes":{},"use_cudagraph":true,"cudagraph_num_of_warmups":0,"cudagraph_capture_sizes":[],"cudagraph_copy_inputs":false,"full_cuda_graph":false,"max_capture_size":0,"local_cache_dir":null})

# ModelConfig(model='/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split', task='generate', tokenizer='/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split', tokenizer_mode='auto', trust_remote_code=True, dtype=torch.bfloat16, seed=0, hf_config_path=None, allowed_local_media_path='', revision=None, code_revision=None, rope_scaling={}, rope_theta=None, tokenizer_revision=None, max_model_len=40960, spec_target_max_model_len=None, quantization=None, enforce_eager=True, max_seq_len_to_capture=8192, max_logprobs=20, disable_sliding_window=False, disable_cascade_attn=False, skip_tokenizer_init=False, enable_prompt_embeds=False, served_model_name='/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split', limit_mm_per_prompt={}, use_async_output_proc=False, config_format=<ConfigFormat.AUTO: 'auto'>, hf_token=None, hf_overrides={}, mm_processor_kwargs=None, disable_mm_preprocessor_cache=False, override_neuron_config={}, pooler_config=None, override_pooler_config=None, logits_processor_pattern=None, generation_config='auto', override_generation_config={}, enable_sleep_mode=False, model_impl='auto', override_attention_dtype=None)


vllm_model = ErnieForCausalLM(vllm_config=vllm_config)

weights = {}

with safe_open(MODEL_PATH, framework="pt") as f:
    for key in f.keys():
        weights[key] = f.get_tensor(key)

vllm_model.load_weights(weights)
