from huggingface_hub import snapshot_download
import os
import shutil

snapshot_download(repo_id="xishushu/SyncHuman", local_dir="./tmp/")

os.rename("./tmp/ckpts", "./ckpts")
shutil.rmtree("./tmp")
