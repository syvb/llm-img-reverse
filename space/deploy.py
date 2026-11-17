"""Upload the Space and request ZeroGPU hardware.

    HF_TOKEN=... python space/deploy.py <owner>/<space>

Uploads space/ as the Space root, plus the llm_img_reverse package and the example images.
"""

import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

ROOT = Path(__file__).resolve().parent.parent


def main(repo_id):
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
    files = {path.name: path for path in (ROOT / "space").iterdir() if path.name != "deploy.py" and path.is_file()}
    files |= {f"llm_img_reverse/{p.name}": p for p in (ROOT / "llm_img_reverse").glob("*.py")}
    files |= {f"examples/{p.name}": p for p in (ROOT / "examples").glob("*.png")}
    api.create_commit(
        repo_id, repo_type="space", commit_message="Deploy from syvb/llm-img-reverse",
        operations=[CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path)) for name, path in files.items()],
    )
    runtime = api.get_space_runtime(repo_id)
    if "zero-a10g" not in (runtime.hardware, runtime.requested_hardware):
        api.request_space_hardware(repo_id, "zero-a10g")
    print(f"https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main(sys.argv[1])
