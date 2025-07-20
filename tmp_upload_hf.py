from huggingface_hub import create_repo, upload_folder

# Replace with your values
repo_id = "RyanWW/ExtremCountAV"  # your Hugging Face repo ID
local_dir = "/dockerx/share/AudioBench/Cropped_Videos"  # the folder you want to upload

# Create the repo (only once)
create_repo(repo_id, repo_type="dataset", private=False)

# Upload folder
upload_folder(
    repo_id=repo_id,
    folder_path=local_dir,
    repo_type="dataset",  # or "model"
)