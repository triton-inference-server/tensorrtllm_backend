from .conftest import venv_check_call, venv_check_output


def test_llm_backend_gpt_accuracy(llm_backend_gpt_example_root,
                                  llm_backend_venv):
    print("Execute client.py...")
    run_cmd = [
        f"{llm_backend_gpt_example_root}/client.py",
        "--text=Born in north-east France, Soyer trained as a",
        "--output_len=10",
    ]

    output = venv_check_output(llm_backend_venv,
                               run_cmd).strip().split("\n")[-1]

    valid_outputs = [
        "Output:  chef and eventually became a chef at a Michelin",
        "Output:  chef before moving to London in the early 1990s",
    ]
    assert output in valid_outputs, "bad output"


def test_llm_backend_gpt_e2e(llm_backend_gpt_example_root, llm_backend_venv):
    print("Execute end_to_end_test.py...")
    run_cmd = [
        f"{llm_backend_gpt_example_root}/end_to_end_test.py",
    ]
    venv_check_call(llm_backend_venv, run_cmd)
