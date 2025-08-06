from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from peft import PeftModel

# 1) 베이스 모델 로드
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    torch_dtype="bfloat16",
    # attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="../qwen2vl_inference/model_cache",
)

# 2) LoRA 모델 로드
lora_model_path = "./model_weight/aihub_data_v1_1_crowdworks/qwen2_5_vl_3B_lora/llamafactory_maxpx_1M_4epoch"
model = PeftModel.from_pretrained(base_model, lora_model_path)

# LoRA 가중치를 합쳐서 하나의 모델로 만듦
merged_model = model.merge_and_unload()

# 모델 가중치 및 구성 저장
save_path = "./model_weight/aihub_data_v1_1_crowdworks/qwen2_5_vl_3B_lora/llamafactory_maxpx_1M_4epoch_merged"
merged_model.save_pretrained(save_path)

# 예) AutoProcessor, AutoTokenizer 사용 시
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# processor = AutoProcessor.from_pretrained(your_model_path)  # 필요 시

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained(your_model_path)  # 필요 시

processor.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)