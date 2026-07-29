"""
Upload the staged Pro-Worker AI Benchmark dataset to a Hugging Face dataset repo.

Prerequisites (do these once, in your browser):
  1. Sign up for an anonymous HF account: https://huggingface.co/join
     - Use a throwaway email (NOT your real one — submission is double-blind).
     - Pick a username that does NOT identify you (e.g., 'pwb-anon-2026').
  2. Create an empty dataset repo: https://huggingface.co/new-dataset
     - Owner: your new anonymous username
     - Dataset name: pro-worker-ai-benchmark
     - License: cc-by-4.0
     - Visibility: Public
  3. Generate an access token: https://huggingface.co/settings/tokens
     - Type: 'Write'
     - Save the token somewhere; you'll paste it below.

Then run this script:
    pip install huggingface_hub
    HF_USERNAME=your-anon-username HF_TOKEN=hf_xxx python upload_to_hf.py

It uploads ~315 MB across ~80 files. Expect 5-10 minutes on a normal connection.
"""

import json
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, login
except ImportError:
    sys.exit("Run: pip install huggingface_hub")


REPO_NAME = "pro-worker-ai-benchmark"
STAGING_DIR = Path(__file__).parent / "dataset_staging"


def main():
    username = os.environ.get("HF_USERNAME")
    token = os.environ.get("HF_TOKEN")
    if not username or not token:
        sys.exit(
            "Set HF_USERNAME and HF_TOKEN environment variables.\n"
            "Example: HF_USERNAME=pwb-anon-2026 HF_TOKEN=hf_xxx python upload_to_hf.py"
        )

    if not STAGING_DIR.exists():
        sys.exit(f"Staging directory not found: {STAGING_DIR}")

    repo_id = f"{username}/{REPO_NAME}"

    # Update croissant.json with the real HF URL before upload
    croissant_path = STAGING_DIR / "croissant.json"
    with open(croissant_path) as f:
        croissant = json.load(f)
    croissant["url"] = f"https://huggingface.co/datasets/{repo_id}"
    with open(croissant_path, "w") as f:
        json.dump(croissant, f, indent=2)
    print(f"Patched croissant.json url -> {croissant['url']}")

    # Update README.md with the real repo_id
    readme_path = STAGING_DIR / "README.md"
    readme = readme_path.read_text()
    readme = readme.replace("REPLACE_WITH_HF_USERNAME/pro-worker-ai-benchmark", repo_id)
    readme_path.write_text(readme)
    print(f"Patched README.md repo references -> {repo_id}")

    login(token=token)
    api = HfApi()

    # Confirm the repo exists (created in browser during prerequisites)
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Found dataset repo: {repo_id}")
    except Exception as e:
        sys.exit(
            f"Could not find dataset repo {repo_id!r}.\n"
            f"Create it first at https://huggingface.co/new-dataset\n"
            f"Underlying error: {e}"
        )

    # Upload everything in staging
    print(f"Uploading {STAGING_DIR} -> {repo_id} ...")
    api.upload_folder(
        folder_path=str(STAGING_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Initial release: v2.0 — 7 models, 11 dimensions, ~96k scored instances",
    )

    print()
    print(f"Done. Dataset live at: https://huggingface.co/datasets/{repo_id}")
    print()
    print("Next steps:")
    print(f"  1. Open https://huggingface.co/datasets/{repo_id} in incognito to verify it loads without login.")
    print(f"  2. Validate the croissant: paste {croissant['url']} into")
    print(f"     https://mlcommons.github.io/croissant/")
    print(f"  3. Update submission/urls.txt with the URL above and re-zip supplementary.zip.")


if __name__ == "__main__":
    main()
