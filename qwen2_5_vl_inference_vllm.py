from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from natsort import natsorted
from pathlib import Path
from tqdm import tqdm
import time
import json
from vllm.lora.request import LoRARequest
import os

MODEL_PATH = "./model_weight/aihub_data_v1_1_crowdworks/qwen2_5_vl_3B_lora/llamafactory_4epoch_merged"
# MODEL_PATH = "Qwen/Qwen2-VL-7B-Instruct"

llm = LLM(
    model=MODEL_PATH,
    # limit_mm_per_prompt={"image": 10, "video": 10},
    limit_mm_per_prompt={"image": 1, "video": 0},
    # gpu_memory_utilization=0.95,
    gpu_memory_utilization=0.95,
    max_model_len=16384,
    download_dir="./model_cache",
    # quantization="AWQ",
    # enable_lora=True
    dtype="bfloat16"
)

min_pixels = 256*28*28
# max_pixels = 1280*28*28
max_pixels = 262144

processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=min_pixels,
    max_pixels=max_pixels, 
    cache_dir="./model_cache"
)

print(processor)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    # repetition_penalty=1.2,
    # presence_penalty=1.5,
    max_tokens=512,
    stop_token_ids=[],
)

# sampling_params = SamplingParams(
#     temperature=0.01,
#     top_p=0.001,
#     top_k=1,
#     repetition_penalty=1.0,
#     max_tokens=512,
#     stop_token_ids=[],
# )

# input_dir = "../test_data/chart_test_v2_0/images"
input_dir = "../test_data/chart_test_v2_0/images"
# JSON 파일로 저장
# output_file = "qwen2vl_7B_aihub_llamafactory_lora_epoch_3_vLLM_results.json"

image_paths = natsorted(
        [str(p) for p in Path(input_dir).glob('*') if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    )

# 추론 결과를 저장할 리스트
results = []
# 추론 시간을 기록할 리스트
inference_times = []

# 결과 저장 디렉토리 생성
save_dir = "./qwen2_5_vl_3B_aihub_v1_1_crowdworks_4epoch_maxpx_262144_result_md/"
os.makedirs(save_dir, exist_ok=True)
                    
for image_path in tqdm(image_paths, desc="Running inference"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "file://" + image_path,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels
                },
                {"type": "text", "text": "차트를 테이블로 변환해줘. 테이블만 출력해줘."},
            ],
        },
    ]
    # For video input, you can pass following values instead:
    # "type": "video",
    # "video": "<video URL>",

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    # 시작 시간 측정
    start_time = time.time()

    # lora_adapter_path = "./model_weight/aihub_data_v1_1/qwen2vl_7B_lora/llamafactory_3epoch"
    
    outputs = llm.generate(
        [llm_inputs],
        sampling_params=sampling_params,
        # lora_request=LoRARequest("lora_adapter", 1, lora_adapter_path)
    )

    generated_text = outputs[0].outputs[0].text

    # 종료 시간 측정
    end_time = time.time()

    # 이미지 파일 이름에서 확장자 제거하고 .md로 변경
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    md_file_name = f"{image_name}.md"
    md_file_path = os.path.join(save_dir, md_file_name)

    # Markdown 파일로 저장
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(f"<chart>{generated_text}</chart>\n")
    
    # 추론 시간 계산 및 리스트에 추가
    inference_time = end_time - start_time
    inference_times.append(inference_time)
    print(f"추론 시간: {inference_time:.2f}초")
    print({"image_path": image_path, "generated_text": generated_text})
    
    # print(generated_text)

    # generated_text_rep_filtered = remove_trailing_repeated_rows(generated_text)
    # print(generated_text_rep_filtered)

    # generated_text_rep_filtered = remove_trailing_repeated_rows(generated_text)
    # print(generated_text_rep_filtered)

    # 결과를 리스트에 추가
    result = {
        "image_path": image_path,
        "generated_text": [generated_text]
    }
    results.append(result)

# 전체 평균 추론 시간 계산
average_inference_time = sum(inference_times) / len(inference_times)
print(f"전체 평균 추론 시간: {average_inference_time:.2f}초")