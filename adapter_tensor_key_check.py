import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

BASE = "./model_weight/qwen2_5_vl_3B_huggingface_download"
LORA_DIR = "./model_weight/chart_qlora_test_3b"  # 어댑터 폴더

# 1) 4bit quantization 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # bfloat16 연산 (A100이면 권장)
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 2) 원본(base) 모델을 4bit로 로드
base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE,
    quantization_config=bnb_config,
    device_map="auto"
).eval()

# 3) 어댑터(peft) 모델도 같은 방식으로 로드 후 주입
peft = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE,
    quantization_config=bnb_config,
    device_map="auto"
)
peft = PeftModel.from_pretrained(peft, LORA_DIR).eval()

# 4) 입력 준비
proc = AutoProcessor.from_pretrained(BASE)
inputs = proc(text=["테스트"], return_tensors="pt").to("cuda")

# 5) 로짓 비교
with torch.no_grad():
    logits_base = base(**inputs).logits[:, -1, :]
    logits_peft = peft(**inputs).logits[:, -1, :]

diff = (logits_base - logits_peft).abs().max().item()
print(f"max|Δlogits| (base(4bit) vs peft(4bit)) = {diff:.6g}")