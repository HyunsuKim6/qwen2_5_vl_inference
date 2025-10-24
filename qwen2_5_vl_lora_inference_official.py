
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
from natsort import natsorted
from pathlib import Path
from tqdm import tqdm
import os
import json
import time
from peft import PeftModel

# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto", cache_dir="./model_cache"
# )

# 로컬 모델 경로 설정
local_model_path = "./model_weight/qwen2_5_vl_3B_huggingface_download"
# hugging_face_model_path = "Qwen/Qwen2-VL-7B-Instruct"

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    local_model_path,
    # torch_dtype=torch.bfloat16,
    torch_dtype="auto",
    attn_implementation="flash_attention_2",
    device_map="auto",
    cache_dir="./model_cache",
    quantization_config=quantization_config
)

# default processer
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", cache_dir="./model_cache")

# # 로컬 LoRA 모델의 경로를 지정하여 병합
lora_model_path = "./model_weight/chart_qlora_test_3b"  # 로컬 LoRA 모델 경로
model = PeftModel.from_pretrained(model, lora_model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 50000
max_pixels = 1000000
processor = AutoProcessor.from_pretrained(local_model_path, min_pixels=min_pixels, max_pixels=max_pixels, cache_dir="./model_cache")

input_dir = "../test_data/chart_test_v2_0/images"

image_paths = natsorted(
        [str(p) for p in Path(input_dir).glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    )
# results = []

# 추론 시간을 기록할 리스트
inference_times = []

# 결과 저장 디렉토리 생성
save_dir = "./qwen2_5_vl_3b_qlora_test/"
os.makedirs(save_dir, exist_ok=True)

for image_path in tqdm(image_paths, desc="Running inference"):
    # 시작 시간 기록
    start_time = time.time()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "file://" + image_path,
                },
                {"type": "text", "text": "차트를 테이블로 변환해줘. 테이블만 출력해줘."},
            ],
        }
    ]

    # # tokenizer_config.json 파일 경로 지정
    # tokenizer_config_path = "../Qwen2-VL-Finetune/output/bar_chart_synth_eng_kor_v1_0/full_finetuning/checkpoint-3500/tokenizer_config.json"

    # # tokenizer_config.json 파일에서 chat_template 읽어오기
    # with open(tokenizer_config_path, 'r') as f:
    #     tokenizer_config = json.load(f)

    # # chat_template이 존재하는지 확인하고 설정
    # chat_template = tokenizer_config.get("chat_template", None)

    # processor 초기화
    # processor = AutoProcessor.from_pretrained(hugging_face_model_path, cache_dir="./model_cache")

    # # chat_template이 있으면 processor에 설정
    # if chat_template:
    #     processor.chat_template = chat_template
    # else:
    #     print("chat_template이 tokenizer_config.json에 정의되지 않았습니다.")

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # results.append({"image_path": image_path, "generated_text": output_text})
    # 이미지 파일 이름에서 확장자 제거하고 .md로 변경
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    md_file_name = f"{image_name}.md"
    md_file_path = os.path.join(save_dir, md_file_name)

    # Markdown 파일로 저장
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(f"<chart>{output_text}</chart>\n")
    
    print({"image_path": image_path, "generated_text": output_text})

    # 종료 시간 기록
    end_time = time.time()
    
    # 추론 시간 계산 및 리스트에 추가
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    print(f"추론 시간: {inference_time:.2f}초")

# save_path = "./"
# os.makedirs(save_path, exist_ok=True)
# output_file = os.path.join(save_path, "qwen2vl_7B_aihub_lora_epoch_1_results_level_3.json")
# with open(output_file, 'w') as f:
#     json.dump(results, f, ensure_ascii=False, indent=4)

# 전체 평균 추론 시간 계산
average_inference_time = sum(inference_times) / len(inference_times)
print(f"전체 평균 추론 시간: {average_inference_time:.2f}초")