from __future__ import annotations


def build_hf_download_command(model_id: str, local_dir: str) -> list[str]:
    return [
        "huggingface-cli",
        "download",
        model_id,
        "--local-dir",
        local_dir,
        "--resume-download",
    ]


def build_vllm_serve_command(
    model_path: str,
    tensor_parallel_size: int,
    port: int,
    served_model_name: str,
) -> list[str]:
    return [
        "vllm",
        "serve",
        model_path,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--served-model-name",
        served_model_name,
    ]
