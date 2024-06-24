import os

from trt_test.misc import check_call, print_info

install_requirement_cmd = "pip3 install -r requirements.txt"


def prepare_medusa_vicuna_7b_engine(tensorrt_llm_medusa_example_root,
                                    vicuna_7b_model_root,
                                    medusa_vicuna_7b_model_root):
    # Convert Medusa from HF
    ckpt_dir = os.path.join(tensorrt_llm_medusa_example_root, "model_dir",
                            "medusa_vicuna_7b")
    convert_cmd = [
        "python3", f"{tensorrt_llm_medusa_example_root}/convert_checkpoint.py",
        f"--model_dir={vicuna_7b_model_root}",
        f"--medusa_model_dir={medusa_vicuna_7b_model_root}",
        f"--output_dir={ckpt_dir}", "--dtype=float16", "--num_medusa_heads=4"
    ]

    # Build Medusa: float16
    engine_dir = os.path.join(tensorrt_llm_medusa_example_root, "engine_dir",
                              "medusa_vicuna_7b")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gemm_plugin=float16",
        "--max_batch_size=8",
        "--max_input_len=300",
        "--max_output_len=300",
        "--speculative_decoding_mode=medusa",
    ]

    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_medusa_example_root)
        check_call(convert_cmd, shell=True)
        check_call(build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_t5_small_engine(tensorrt_llm_enc_dec_example_root,
                            t5_small_model_root):
    # Convert T5 from HF
    ckpt_dir = os.path.join(tensorrt_llm_enc_dec_example_root, "model_dir",
                            "t5_small")
    convert_cmd = [
        "python3",
        f"{tensorrt_llm_enc_dec_example_root}/convert_checkpoint.py",
        "--model_type=t5",
        f"--model_dir={t5_small_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    # Build encoder and decoder
    encoder_engine_dir = os.path.join(tensorrt_llm_enc_dec_example_root,
                                      "engine_dir", "t5_small_encoder")
    decoder_engine_dir = os.path.join(tensorrt_llm_enc_dec_example_root,
                                      "engine_dir", "t5_small_decoder")

    encoder_build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}/encoder",
        f"--output_dir={encoder_engine_dir}",
        "--paged_kv_cache=disable",
        "--moe_plugin=disable",
        "--enable_xqa=disable",
        "--max_beam_width=1",
        "--max_batch_size=8",
        "--max_seq_len=300",
        "--gemm_plugin=float16",
        "--bert_attention_plugin=float16",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=disable",
        "--use_custom_all_reduce=disable",
    ]
    decoder_build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}/decoder",
        f"--output_dir={decoder_engine_dir}",
        "--moe_plugin=disable",
        "--enable_xqa=disable",
        "--max_beam_width=1",
        "--max_batch_size=8",
        "--max_seq_len=300",
        "--gemm_plugin=float16",
        "--bert_attention_plugin=float16",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=disable",
        "--max_input_len=1",
        "--use_custom_all_reduce=disable",
    ]

    convert_cmd = " ".join(convert_cmd)
    encoder_build_cmd = " ".join(encoder_build_cmd)
    decoder_build_cmd = " ".join(decoder_build_cmd)
    if not os.path.exists(encoder_build_cmd):
        check_call(convert_cmd, shell=True)
        check_call(encoder_build_cmd, shell=True)
        check_call(decoder_build_cmd, shell=True)

    else:
        print_info(f"Reusing engine: {encoder_engine_dirr}")
        print_info(f"Reusing engine: {decoder_engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {encoder_build_cmd}")
        print_info(f"Skipped: {decoder_build_cmd}")

    assert os.path.exists(
        encoder_engine_dir), f"{encoder_engine_dir} does not exists."
    assert os.path.exists(
        decoder_engine_dir), f"{decoder_engine_dir} does not exists."
    return encoder_engine_dir, decoder_engine_dir


def prepare_gpt_350m_engine(type, tensorrt_llm_gpt_example_root,
                            gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_350m")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={gpt_tokenizer_model_root}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
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
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        "--use_paged_context_fmha=enable",
        "--remove_input_padding=enable",
        "--max_batch_size=64",
        "--max_input_len=924",
        "--max_output_len=100",
        "--gather_generation_logits",
        f"--output_dir={engine_dir}",
    ]

    if type == "medium_ifb":
        build_cmd += [
            "--max_draft_len=5",
            "--speculative_decoding_mode=draft_tokens_external",
        ]

    if type == "ifb" or type == "medium_ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
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


def prepare_gpt_gather_logits_engine(type, tensorrt_llm_gpt_example_root,
                                     gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_gather_logits")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={gpt_tokenizer_model_root}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
    ]

    # Build GPT
    engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                              "gpt_gather_logits")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        "--remove_input_padding=enable",
        "--max_batch_size=128",
        "--max_input_len=300",
        "--max_output_len=300",
        "--gather_all_token_logits",
        "--max_num_tokens=38400",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
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


def prepare_gpt_return_logits_engine(type, tensorrt_llm_gpt_example_root,
                                     gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_return_logits")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={gpt_tokenizer_model_root}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
    ]

    # Build GPT
    engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                              "gpt_return_logits")

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--max_batch_size=4",
        "--max_input_len=512",
        "--max_output_len=28",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_num_tokens=38400",
        "--use_paged_context_fmha=enable",
        "--gather_generation_logits",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
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
                               gpt_2b_lora_model_root, models_root,
                               weight_streaming):
    # Convert GPT from NeMo
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_2b_lora")
    gpt_2b_nemo_model = os.path.join(models_root, "GPT-2B-001_bf16_tp1.nemo")

    convert_ckpt_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--nemo_ckpt_path={gpt_2b_nemo_model}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
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
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--lora_plugin=float16",
        f"--lora_dir={gpt_2b_lora_900_nemo_model}",
        "--lora_ckpt_source=nemo",
        "--lora_target_modules=attn_qkv",
        "--remove_input_padding=enable",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_output_len=128",
        f"--output_dir={engine_dir}",
    ]

    if weight_streaming:
        build_cmd += ["--gemm_plugin=disable", "--weight_streaming"]
    else:
        build_cmd += [
            "--gemm_plugin=float16",
        ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
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

    convert_cmd = [
        "python3",
        f"{tensorrt_llm_gpt_example_root}/../generate_checkpoint_config.py",
        f"--output_path={engine_dir}/ckpt_config.json",
        "--architecture=GPTForCausalLM", "--dtype=float16",
        "--num_hidden_layers=96", "--num_attention_heads=96",
        "--hidden_size=12288", "--vocab_size=51200", "--hidden_act=gelu",
        "--tp_size=8"
    ]

    build_cmd = [
        "trtllm-build",
        f"--model_config={engine_dir}/ckpt_config.json",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--gemm_plugin=float16",
        "--max_batch_size=32",
        "--max_input_len=512",
        "--max_output_len=32",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
        ]

    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_gpt_example_root)
        check_call(convert_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_gpt_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v2_7b_engine(type, tensorrt_llm_llama_example_root,
                               llama_v2_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "llama_v2_7b_python_backend")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_7b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "llama_v2_7b_ifb")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_7b_ifb")
    # The path of weights in data server
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "7B")

    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        f"--output_dir={ckpt_dir}",
        "--dtype=bfloat16",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        "--gemm_plugin=bfloat16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
    ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
        ]

    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_llama_v2_70b_engine(type, tensorrt_llm_llama_example_root,
                                llama_v2_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "llama_v2_70b_python_backend")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_70b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "llama_v2_70b_ifb")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "llama_v2_70b_ifb")
    # The path of weights in data server
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "70B")
    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        f"--output_dir={ckpt_dir}",
        "--dtype=bfloat16",
        "--tp_size=8",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        "--gemm_plugin=bfloat16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
    ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
        ]

    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
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
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "gpt_next_ptuning")
    convert_weights_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--nemo_ckpt_path={nemo_model_path}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
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
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--remove_input_padding=enable",
        "--paged_kv_cache=enable",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        "--max_batch_size=8",
        "--max_input_len=924",
        "--max_output_len=128",
        "--max_beam_width=1",
        f"--output_dir={engine_dir}",
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
    assert os.path.exists(ckpt_dir), f"{ckpt_dir} does not exists."
    return engine_dir, ckpt_dir


def prepare_mistral_v1_7b_engine(type, tensorrt_llm_llama_example_root,
                                 mistral_v1_tokenizer_model_root):
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "mistral_v1_7b_python_backend")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "mistral_v1_7b_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "mistral_v1_7b_ifb")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "mistral_v1_7b_ifb")

    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--model_dir={mistral_v1_tokenizer_model_root}",
        f"--output_dir={ckpt_dir}",
        "--dtype=float16",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_input_len=8192",
    ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
        ]
    elif type == "python_backend":
        build_cmd += [
            "--max_batch_size=8",
        ]

    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir


def prepare_rcca_nvbug_4323566_engine(type, tensorrt_llm_gpt_example_root,
                                      gpt_tokenizer_model_root):
    # Convert GPT weights from HF
    ckpt_dir = os.path.join(tensorrt_llm_gpt_example_root, "model_dir",
                            "rcca_nvbug_4323566")
    convert_cmd = [
        "python3", f"{tensorrt_llm_gpt_example_root}/convert_checkpoint.py",
        f"--model_dir={gpt_tokenizer_model_root}", "--dtype=float16",
        f"--output_dir={ckpt_dir}"
    ]

    # Build GPT
    if type == "python_backend":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "rcca_nvbug_4323566_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_gpt_example_root, "engine_dir",
                                  "rcca_nvbug_4323566_ifb")
    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        "--gpt_attention_plugin=float16",
        "--gemm_plugin=float16",
        "--context_fmha=enable",
        "--remove_input_padding=enable",
        "--max_batch_size=64",
        "--max_input_len=924",
        "--max_output_len=100",
        f"--output_dir={engine_dir}",
    ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
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
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "rcca_nvbug_4342666_python_backend")
    elif type == "ifb":
        engine_dir = os.path.join(tensorrt_llm_llama_example_root,
                                  "engine_dir", "rcca_nvbug_4342666_ifb")
        ckpt_dir = os.path.join(tensorrt_llm_llama_example_root, "ckpt_dir",
                                "rcca_nvbug_4342666_ifb")
    # Weights of Llama-v2-7b-chat model
    meta_ckpt_dir = os.path.join(llama_v2_tokenizer_model_root, "7BF")
    convert_cmd = [
        "python3",
        "convert_checkpoint.py",
        f"--meta_ckpt_dir={meta_ckpt_dir}",
        f"--output_dir={ckpt_dir}",
        "--dtype=bfloat16",
    ]

    build_cmd = [
        "trtllm-build",
        f"--checkpoint_dir={ckpt_dir}",
        f"--output_dir={engine_dir}",
        "--gpt_attention_plugin=bfloat16",
        "--gemm_plugin=bfloat16",
        "--remove_input_padding=enable",
        "--context_fmha=enable",
        "--max_beam_width=4",
    ]

    if type == "ifb":
        build_cmd += [
            "--paged_kv_cache=enable",
        ]

    convert_cmd = " ".join(convert_cmd)
    build_cmd = " ".join(build_cmd)
    if not os.path.exists(engine_dir):
        check_call(install_requirement_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(convert_cmd,
                   shell=True,
                   cwd=tensorrt_llm_llama_example_root)
        check_call(build_cmd, shell=True, cwd=tensorrt_llm_llama_example_root)

    else:
        print_info(f"Reusing engine: {engine_dir}")
        print_info(f"Skipped: {convert_cmd}")
        print_info(f"Skipped: {build_cmd}")

    assert os.path.exists(engine_dir), f"{engine_dir} does not exists."
    return engine_dir
