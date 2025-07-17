from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
import torch

# 1) 베이스 모델 로드
base_model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="bfloat16",
    device_map="auto",
    cache_dir="./model_cache",
)

# 2) LoRA 모델 로드
lora_model_path = "./model_weight/aihub_data_v1_1/qwen2vl_7B_lora/llamafactory_2epoch"
model = PeftModel.from_pretrained(base_model, lora_model_path)

# LoRA 가중치를 합쳐서 하나의 모델로 만듦
merged_model = model.merge_and_unload()

# 저장 전 train 모드 확인
merged_model.train()

# 확실하게 dtype 명시
merged_model.config.torch_dtype = torch.bfloat16

# 모델 가중치 및 구성 저장
save_path = "./model_weight/aihub_data_v1_1/qwen2vl_7B_lora/llamafactory_2epoch_merged_for_train"
merged_model.save_pretrained(save_path, safe_serialization=True)

# 예) AutoProcessor, AutoTokenizer 사용 시
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# processor = AutoProcessor.from_pretrained(your_model_path)  # 필요 시
 
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained(your_model_path)  # 필요 시

processor.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)