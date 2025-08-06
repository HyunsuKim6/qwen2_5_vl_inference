from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from io import BytesIO
import time
import uvicorn
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
import requests
from typing import List

# def remove_trailing_repeated_rows(table_text, max_repeats=1):
#     # 입력된 텍스트를 줄 단위로 분리
#     lines = table_text.strip().split('\n')
    
#     if len(lines) < 3:
#         return table_text  # 헤더와 구분선이 없으면 원본 반환
    
#     header = lines[0]
#     separator = lines[1]
#     data_rows = lines[2:]
    
#     # 헤더의 열 수를 기준으로 각 행을 정규화
#     def normalize_row(row, num_columns):
#         cells = [cell.strip() for cell in row.strip('|').split('|')]
#         # 열 수가 부족하면 빈 문자열로 채움
#         if len(cells) < num_columns:
#             cells += [''] * (num_columns - len(cells))
#         return '| ' + ' | '.join(cells) + ' |'
    
#     # 헤더 기준 열 수
#     num_columns = len([cell.strip() for cell in header.strip('|').split('|')])
    
#     # 모든 데이터 행을 정규화
#     normalized_data_rows = [normalize_row(row, num_columns) for row in data_rows]
    
#     # 뒤에서부터 반복되는 행을 제거
#     unique_rows = []
#     repeat_count = {}
#     for row in reversed(normalized_data_rows):
#         row_key = row.strip()
#         if row_key in repeat_count:
#             repeat_count[row_key] += 1
#         else:
#             repeat_count[row_key] = 1
        
#         if repeat_count[row_key] <= max_repeats:
#             unique_rows.insert(0, row)
    
#     # 테이블을 다시 결합
#     filtered_table = '\n'.join([header, separator] + unique_rows)
#     return filtered_table

class UrlModel(BaseModel):
    url: str
    crop_coords: List[List[int]] = Field(..., description="Coordinates list of chart image in format [[x1, y1, x2, y2], ...]")
    
app = FastAPI()

# Qwen2-VL + LoRA 파인튜닝 + vLLM + AWQ INT4 변환 (X)
MODEL_PATH = "./model_weight/aihub_data_v1_1_crowdworks/qwen2_5_vl_3B_lora/llamafactory_4epoch_merged"

# 모델 로드
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 1},
    gpu_memory_utilization=0.95,
    max_model_len=16384,
    download_dir="./model_cache",
    # quantization="AWQ"
)

min_pixels = 256 * 28 * 28
max_pixels = 1280 * 28 * 28
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    cache_dir="./model_cache"
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=512,
    stop_token_ids=[],
)

# 기본 형태 API
@app.post("/chart_rec_default")
async def chart_rec_default(
    image: UploadFile = File(...),
):
    # 이미지 판별
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # 이미지 로드
    image_data = await image.read()
    image_stream = BytesIO(image_data)
    pil_image = Image.open(image_stream).convert("RGB")

    # 리사이징 (비율 유지, 최대 크기 제한)
    # max_side = 1200  # 원하는 최대 한 변의 길이
    max_side = 1200
    w, h = pil_image.size
    print("original_size: ", w, h)
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        pil_image = pil_image.resize(new_size, Image.LANCZOS)
        print("new_size: ", new_size)

    prompt_text = "차트를 테이블로 변환해줘. 테이블만 출력해줘."

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": pil_image,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
                {"type": "text", "text": prompt_text},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    mm_data = {"image": [pil_image]} if pil_image else {}
    
    # image_inputs, video_inputs = process_vision_info(messages)

    # mm_data = {}
    # if image_inputs is not None:
    #     mm_data["image"] = image_inputs
    # if video_inputs is not None:
    #     mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }

    # 모델 추론 및 시간 측정
    start_time = time.time()
    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    end_time = time.time()

    generated_text = outputs[0].outputs[0].text

    # generated_text_rep_filtered = remove_trailing_repeated_rows(generated_text)

    response = {
            "status": "succeeded",
            "analyzeResult": {
                "content": generated_text
            },
            "inference_time": round(end_time - start_time, 2)
        }

    return JSONResponse(content=response)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)