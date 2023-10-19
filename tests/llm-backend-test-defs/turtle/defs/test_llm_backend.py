from .conftest import venv_check_call, venv_check_output


def test_llm_backend_gpt_accuracy(llm_backend_gpt_example_root,
                                  llm_backend_venv):
    print("Execute client.py...")
    run_cmd = [
        f"{llm_backend_gpt_example_root}/client.py",
        "--text=Born in north-east France, Soyer trained as a",
        "--output_len=10",
        "--tokenizer_dir=gpt2",
        "--tokenizer_type=auto",
    ]

    output = venv_check_output(llm_backend_venv,
                               run_cmd).strip().split("\n")[-1]

    valid_outputs = [
        "Output:  chef and eventually became a chef at a Michelin",
        "Output:  chef before moving to London in the early 1990s",
        "Output:  chef before moving to London in the late 1990s",
    ]
    assert output in valid_outputs, "bad output"


def test_llm_backend_gpt_e2e(llm_backend_gpt_example_root, llm_backend_venv):
    print("Execute end_to_end_test.py...")
    run_cmd = [
        f"{llm_backend_gpt_example_root}/end_to_end_test.py",
        "--tokenizer_dir=gpt2",
        "--tokenizer_type=auto",
    ]
    venv_check_call(llm_backend_venv, run_cmd)


def test_inflight_batcher_llama_default(inflight_batcher_llm_client_root,
                                        llama_v2_tokenizer_model_root,
                                        llm_backend_venv):
    print("Execute inflight_batcher_llm_client.py on default mode...")
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer_dir={llama_v2_tokenizer_model_root}",
        "--tokenizer_type=llama",
    ]
    venv_check_call(llm_backend_venv, run_cmd)


def test_inflight_batcher_llama_streaming(inflight_batcher_llm_client_root,
                                          llama_v2_tokenizer_model_root,
                                          llm_backend_venv):
    print("Execute inflight_batcher_llm_client.py on streaming mode...")
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer_dir={llama_v2_tokenizer_model_root}",
        "--tokenizer_type=llama",
        "--streaming",
    ]
    venv_check_call(llm_backend_venv, run_cmd)


def test_inflight_batcher_gpt_default(inflight_batcher_llm_client_root,
                                      gpt_tokenizer_model_root,
                                      llm_backend_venv):
    print("Execute inflight_batcher_llm_client.py on default mode...")
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer_dir={gpt_tokenizer_model_root}",
        "--tokenizer_type=auto",
    ]
    venv_check_call(llm_backend_venv, run_cmd)


def test_inflight_batcher_gpt_streaming(inflight_batcher_llm_client_root,
                                        gpt_tokenizer_model_root,
                                        llm_backend_venv):
    print("Execute inflight_batcher_llm_client.py on streaming mode...")
    run_cmd = [
        f"{inflight_batcher_llm_client_root}/inflight_batcher_llm_client.py",
        f"--tokenizer_dir={gpt_tokenizer_model_root}",
        "--tokenizer_type=auto",
        "--streaming",
    ]
    venv_check_call(llm_backend_venv, run_cmd)
