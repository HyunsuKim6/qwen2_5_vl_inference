import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig

BASE_MODEL = "./model_weight/qwen2_5_vl_7B_huggingface_download"   # 원하는 기본 모델로 바꾸세요
SAVE_DIR   = "./model_weight/qwen2_5_vl_7B_huggingface_download_bnb_4bit_fp4_nodq_bf16"  # 저장 경로

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                     # 4bit 로드
    bnb_4bit_use_double_quant=False,        # double quant (nested)
    bnb_4bit_quant_type="fp4",             # "nf4" 또는 "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16  # 연산 dtype: torch.float16 / torch.bfloat16 권장
)

# 토크나이저 저장
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tok.save_pretrained(SAVE_DIR)

processor = AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)
processor.save_pretrained(SAVE_DIR)

# 모델 로드(4bit)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)

# 주의: bnb 4bit로 저장할 때, 환경/버전에 따라 가중치가
# 실제 'qweight' 형태의 bitsandbytes 텐서로 저장되지 않고,
# 설정 위주로만 저장되는 경우가 있습니다.
# 이 경우 vLLM이 사전 양자화 체크포인트로 인식하지 못할 수 있습니다.
# 가능한 최신 transformers/bitsandbytes를 권장합니다.

model.save_pretrained(SAVE_DIR)
print(f"Saved to {SAVE_DIR}")