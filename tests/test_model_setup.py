from local_impact_ranking.model_setup import (
    build_hf_download_command,
    build_vllm_serve_command,
)


def test_build_hf_download_command_uses_local_dir_and_resume():
    command = build_hf_download_command(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        local_dir="models/deepseek-r1-distill-qwen-32b",
    )

    assert command == [
        "huggingface-cli",
        "download",
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "--local-dir",
        "models/deepseek-r1-distill-qwen-32b",
        "--resume-download",
    ]


def test_build_vllm_serve_command_configures_tensor_parallelism():
    command = build_vllm_serve_command(
        model_path="models/qwen3-235b-fp8",
        tensor_parallel_size=8,
        port=8000,
        served_model_name="qwen3-local",
    )

    assert command == [
        "vllm",
        "serve",
        "models/qwen3-235b-fp8",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--tensor-parallel-size",
        "8",
        "--served-model-name",
        "qwen3-local",
    ]
