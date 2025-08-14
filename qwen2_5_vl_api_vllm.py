from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, status
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
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

# 실제 환경에서는 이 토큰을 환경 변수나 비밀 관리 시스템에서 가져오는 것을 권장합니다.
VALID_TOKEN = "vision!@#"

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.scheme != "Bearer" or credentials.credentials != VALID_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication token",
        )

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

min_pixels = 50000
max_pixels = 1000000
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
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    # 이미지 판별
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # 이미지 로드
    image_data = await image.read()
    image_stream = BytesIO(image_data)
    pil_image = Image.open(image_stream).convert("RGB")

    # 리사이징 (비율 유지, 최대 크기 제한)
    # max_side = 2400  # 원하는 최대 한 변의 길이
    # max_side = 800
    # w, h = pil_image.size
    # print("original_size: ", w, h)
    # if max(w, h) > max_side:
    #     scale = max_side / max(w, h)
    #     new_size = (int(w * scale), int(h * scale))
    #     pil_image = pil_image.resize(new_size, Image.LANCZOS)
    #     print("new_size: ", new_size)

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

    # mm_data = {"image": [pil_image]} if pil_image else {}
    
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