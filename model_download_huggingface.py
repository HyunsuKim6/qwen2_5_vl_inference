from huggingface_hub import snapshot_download

# 모델 저장할 경로를 지정 가능
local_dir = "./model_weight/qwen2_5_vl_7B_huggingface_download"

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
    local_dir=local_dir,
    resume_download=True,  # 중간에 끊겨도 이어받기 가능
)