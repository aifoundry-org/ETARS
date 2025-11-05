from typing import Optional
from pathlib import Path


def resolve_model_path(
    model_path: Optional[Path],
    hf_repo: Optional[str],
    hf_filename: str,
    hf_revision: Optional[str],
    hf_token: Optional[str],
    hf_cache: Optional[str],
) -> Path:
    if hf_repo:
        try:
            from huggingface_hub import hf_hub_download
        except Exception as e:
            raise RuntimeError(
                "huggingface_hub is required to load from Hugging Face. "
                "Install with: pip install huggingface_hub"
            ) from e

        downloaded = hf_hub_download(
            repo_id=hf_repo,
            filename=hf_filename,
            revision=hf_revision,
            token=hf_token,
            cache_dir=hf_cache,
            local_files_only=False,
        )
        return Path(downloaded)

    if model_path is None:
        raise ValueError("Provide either a local model_path or hf_repo + hf_filename.")
    return Path(model_path)
