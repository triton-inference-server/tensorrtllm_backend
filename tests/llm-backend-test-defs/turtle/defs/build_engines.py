import os

from trt_test.misc import check_call, print_info

install_requirement_cmd = "pip3 install -r requirements.txt"


def prepare_gpt_350m_engine(type, tensorrt_llm_gpt_example_root,
                            gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    model_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                             "gpt_350m")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/hf_gpt_convert.py",
        f"-i={gpt_tokenizer_model_root}", "--storage-type=float16",
        f"-o={model_dir}"
    ]

    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt350m_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt350m_ifb")
    elif type == "medium_ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt350m_medium_ifb")
    build_cmd = [
        "python3",
        f"{tensorrt_llm_gpt_example_root}/build.py",
        f"--model_dir={model_dir}/1-gpu",
        "--dtype=float16",
        "--use_gpt_attention_plugin=float16",
        "--use_gemm_plugin=float16",
        "--use_layernorm_plugin=float16",
        "--enable_context_fmha",
        "--use_paged_context_fmha",
        "--remove_input_padding",
        "--max_batch_size=64",
        "--max_input_len=924",
        "--max_output_len=100",
        "--hidden_act=gelu",
        f"--output_dir={engine_dir}",
    ]

    if type == "medium_ifb":
        build_cmd += [
            "--max_draft_len=5",
        ]

    if type == "ifb" or type == "medium_ifb":
        build_cmd += [
            "--use_inflight_batching",
            "--paged_kv_cache",
        ]
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_2b_lora_engine(type, tensorrt_llm_gpt_example_root,
                               gpt_2b_lora_model_root, models_root):
    # Convert GPT from NeMo
    model_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                             "gpt_2b_lora")
    gpt_2b_nemo_model = os.path.join(models_root, "GPT-2B-001_bf16_tp1.nemo")

    convert_ckpt_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/nemo_ckpt_convert.py",
        f"-i={gpt_2b_nemo_model}", "--storage-type=float16", f"-o={model_dir}"
    ]

    # prepare more test metrials
    gpt_2b_lora_900_nemo_model = os.path.join(gpt_2b_lora_model_root,
                                              "gpt2b_lora-900.nemo")
    convert_lora_train_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/nemo_lora_convert.py",
        f"-i={gpt_2b_lora_900_nemo_model}", "--storage-type=float16",
        "--write-cpp-runtime-tensors", f"-o=gpt-2b-lora-train-900"
    ]
    convert_lora_train_tllm_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/nemo_lora_convert.py",
        f"-i={gpt_2b_lora_900_nemo_model}", "--storage-type=float16",
        f"-o=gpt-2b-lora-train-900-tllm"
    ]

    check_call(f"cp {gpt_2b_lora_model_root}/gpt2b_lora-900.nemo ./",
               shell=True,
               cwd=tensorrt_llm_gpt_example_root)
    check_call(f"cp {gpt_2b_lora_model_root}/input.csv ./",
               shell=True,
               cwd=tensorrt_llm_gpt_example_root)

    # Build GPT
    engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                              "gpt_2b_lora_ib")

    build_cmd = [
        "python3",
        f"{tensorrt_llm_gpt_example_root}/build.py",
        f"--model_dir={model_dir}/1-gpu",
        "--dtype=float16",
        "--use_gpt_attention_plugin=float16",
        "--use_gemm_plugin=float16",
        "--use_layernorm_plugin=float16",
        "--use_lora_plugin=float16",
        "--lora_target_modules=attn_qkv",
        "--remove_input_padding",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_output_len=128",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--use_inflight_batching",
            "--paged_kv_cache",
        ]
    convert_ckpt_cmd = " ".join(convert_ckpt_cmd)
    build_cmd = " ".join(build_cmd)
    convert_lora_train_cmd = " ".join(convert_lora_train_cmd)
    convert_lora_train_tllm_cmd = " ".join(convert_lora_train_tllm_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_ckpt_cmd, shell=True)
        check_call(convert_lora_train_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_lora_train_tllm_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_ckpt_cmd}")
        print_info(f"Skipped: {build_cmd}")
        print_info(f"Skipped: {convert_lora_train_cmd}")
        print_info(f"Skipped: {convert_lora_train_tllm_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_175b_engine(type, tensorrt_llm_gpt_example_root):
    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt_175b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt_175b_ifb")
    build_cmd = [
        "python3",
        f"{tensorrt_llm_gpt_example_root}/build.py",
        "--world_size=8",
        "--remove_input_padding",
        "--hidden_act=gelu",
        "--n_layer=96",
        "--n_embd=12288",
        "--n_head=96",
        "--max_batch_size=32",
        "--max_input_len=512",
        "--max_output_len=32",
        "--use_gpt_attention_plugin",
        "--use_gemm_plugin",
        "--use_layernorm_plugin",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--use_inflight_batching",
            "--paged_kv_cache",
        ]

    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v2_7b_engine(type, tensorrt_llm_llama_example_root,
                               llama_v2_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "llama_v2_7b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "llama_v2_7b_ifb")
    # The path of weights in data server
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "7B")
    build_cmd = [
        "python3",
        f"{tensorrt_llm_llama_example_root}/build.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        "--dtype=bfloat16",
        "--use_gpt_attention_plugin=bfloat16",
        "--remove_input_padding",
        "--use_gemm_plugin=bfloat16",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--use_inflight_batching",
            "--paged_kv_cache",
        ]

    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v2_70b_engine(type, tensorrt_llm_llama_example_root,
                                llama_v2_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "llama_v2_70b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "llama_v2_70b_ifb")
    # The path of weights in data server
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "70B")
    build_cmd = [
        "python3",
        f"{tensorrt_llm_llama_example_root}/build.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        "--dtype=bfloat16",
        "--use_gpt_attention_plugin=bfloat16",
        "--n_kv_head=32",
        "--remove_input_padding",
        "--use_gemm_plugin=bfloat16",
        f"--output_dir={engine_dir}",
        "--world_size=8",
        "--tp_size=8",
    ]

    if type == "ifb":
        build_cmd += [
            "--use_inflight_batching",
            "--paged_kv_cache",
        ]

    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_gpt_next_ptuning_engine(type, tensorrt_llm_gpt_example_root,
                                    gpt_next_ptuning_model_root):
    if type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "gpt_next_ptuning_ifb")

    # Convert weights from HF
    nemo_model_path = os.path.join(gpt_next_ptuning_model_root,
                                   "megatron_converted_8b_tp4_pp1.nemo")
    output_model_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                                    "gpt_next_ptuning")
    convert_weights_cmd = [
        "python3",
        "nemo_ckpt_convert.py",
        f"-i={nemo_model_path}",
        f"-o={output_model_dir}",
        "--storage-type=float16",
        " --tensor-parallelism=1",
        "--processes=1",
    ]

    # Convert ptuning table
    nemo_model_path = os.path.join(gpt_next_ptuning_model_root,
                                   "email_composition.nemo")
    convert_table_cmd = [
        "python3",
        "nemo_prompt_convert.py",
        f"-i={nemo_model_path}",
        "-o=email_composition.npy",
    ]

    # Copy input.csv
    check_call(f"cp {gpt_next_ptuning_model_root}/input.csv ./",
               shell=True,
               cwd=tensorrt_llm_gpt_example_root)

    # Build engine
    build_cmd = [
        "python3",
        "build.py",
        f"--model_dir={output_model_dir}/1-gpu",
        "--dtype=float16",
        "--use_inflight_batching",
        "--use_gpt_attention_plugin=float16",
        "--paged_kv_cache",
        "--use_gemm_plugin=float16",
        "--use_layernorm_plugin=float16",
        "--remove_input_padding",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_output_len=128",
        "--max_beam_width=1",
        f"--output_dir={engine_dir}",
        "--hidden_act=gelu",
        "--enable_context_fmha",
        "--max_prompt_embedding_table_size=800",
    ]

    convert_weights_cmd = " ".join(convert_weights_cmd)
    convert_table_cmd = " ".join(convert_table_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_weights_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_table_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_weights_cmd}")
        print_info(f"Skipped: {convert_table_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    assert os.path.exists(
        output_model_dir), f"{output_model_dir} does not exists."
    return engine_dir, output_model_dir


def prepare_mistral_v1_7b_engine(type, tensorrt_llm_llama_example_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "mistral_v1_7b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "mistral_v1_7b_ifb")

    build_cmd = [
        "python3",
        f"build.py",
        "--dtype=float16",
        "--n_layer=2",
        "--enable_context_fmha",
        "--use_gpt_attention_plugin",
        "--use_gemm_plugin",
        "--use_rmsnorm_plugin",
        f"--output_dir={engine_dir}",
        "--max_input_len=8192",
    ]

    if type == "ifb":
        build_cmd += [
            "--use_inflight_batching",
            "--paged_kv_cache",
        ]

    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_rcca_nvbug_4323566_engine(type, tensorrt_llm_gpt_example_root,
                                      gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    model_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                             "rcca_nvbug_4323566")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/hf_gpt_convert.py",
        f"-i={gpt_tokenizer_model_root}", "--storage-type=float16",
        f"-o={model_dir}"
    ]

    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "rcca_nvbug_4323566_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "rcca_nvbug_4323566_ifb")
    build_cmd = [
        "python3",
        f"{tensorrt_llm_gpt_example_root}/build.py",
        f"--model_dir={model_dir}/1-gpu",
        "--dtype=float16",
        "--use_gpt_attention_plugin=float16",
        "--use_gemm_plugin=float16",
        "--use_layernorm_plugin=float16",
        "--enable_context_fmha",
        "--remove_input_padding",
        "--max_batch_size=64",
        "--max_input_len=924",
        "--max_output_len=100",
        "--hidden_act=gelu",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--use_inflight_batching",
            "--paged_kv_cache",
        ]
    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_rcca_nvbug_4342666_engine(type, tensorrt_llm_llama_example_root,
                                      llama_v2_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir",
                                  "rcca_nvbug_4342666_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "rcca_nvbug_4342666_ifb")
    # Weights of Llama-v2-7b-chat model
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "7BF")
    build_cmd = [
        "python3",
        "build.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        "--dtype=bfloat16",
        "--use_gpt_attention_plugin=bfloat16",
        "--remove_input_padding",
        "--use_gemm_plugin=bfloat16",
        "--max_beam_width=4",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--use_inflight_batching",
            "--paged_kv_cache",
        ]

    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir
