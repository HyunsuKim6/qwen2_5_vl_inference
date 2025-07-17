from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel
import torch

# 1) 베이스 모델 로드
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype="bfloat16",
    device_map="auto",
    cache_dir="./model_cache",
)

# 확실하게 dtype 명시
base_model.config.torch_dtype = torch.bfloat16

# 모델 가중치 및 구성 저장
save_path = "./model_weight/qwen2_5_vl_7B_original/"
base_model.save_pretrained(save_path, safe_serialization=True)

# 예) AutoProcessor, AutoTokenizer 사용 시
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# processor = AutoProcessor.from_pretrained(your_model_path)  # 필요 시
 
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained(your_model_path)  # 필요 시

processor.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)