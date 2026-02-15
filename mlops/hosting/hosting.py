from huggingface_hub import HfApi
import os

api =HfApi(token=os.getenv("HFTOKEN"))
api.upload_folder(
    folder_path="mlops/deployment",
    repo_id="grkavi0912/Tpro",
    repo_type="space",
    path_in_repo="",

)
