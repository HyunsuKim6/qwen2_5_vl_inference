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
from vllm.lora.request import LoRARequest
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from diagram_util import remove_duplicates, filter_specific_languages
import re
import statistics

import os
import multiprocessing as mp

# 1) 반드시 최상단에서 spawn 고정
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# 2) vLLM에 명시적으로 spawn 사용 지시 (권장)
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# 필요시 V1 엔진 유지 (기본이 1이지만 명시해도 무방)
os.environ.setdefault("VLLM_USE_V1", "1")


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
# MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_PATH = "./model_weight/qwen2_5_vl_3B_huggingface_download_bnb_4bit_fp4_nodq_bf16"

llm = None

@app.on_event("startup")
async def _startup():
    global llm
    if llm is None:
        llm = LLM(
            model=MODEL_PATH,
            download_dir="./model_cache",
            max_model_len=16384,
            gpu_memory_utilization=0.6,
            limit_mm_per_prompt={"image": 1},
            enable_lora=True,
            max_lora_rank=32,
            quantization="bitsandbytes"
            
            # 필요시 기타 인자
            # tensor_parallel_size=1,
            # data_parallel_size=1,                     # ✅ DP 비활성화
            # disable_mm_preprocessor_cache=True,       # ✅ 멀티모달 캐시 비활성화
        )

min_pixels = 50000
max_pixels = 1000000
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
    cache_dir="./model_cache"
)

chart_sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.01,
    max_tokens=512,
    stop_token_ids=[],
)

diagram_sampling_params = SamplingParams(
    temperature=0.01,
    top_p=0.001,
    repetition_penalty=1.08,
    max_tokens=1024,
    stop=["###", "</s>", "\n- - -"]
)

chart_complexity_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    repetition_penalty=1.0,
    max_tokens=5,
    stop_token_ids=[],
)

diagram_complexity_params = SamplingParams(
    max_tokens=5,
    temperature=0.01,
    top_p=0.001,
    repetition_penalty=1.08,
    stop=["###", "</s>", "\n- - -"]
)

lora_paths = {
    "chart_rec": "./model_weight/aihub_data_v1_1_crowdworks/qwen2_5_vl_3B_qlora/fp4_nodq_bf16/checkpoint-105000",
    "diagram_rec": "./model_weight/diagram_qlora_fp4_adapter_3b"
}

# 단일 인자만 받는 경우: (adapter_name, adapter_id, lora_path)
lora_requests = {
    name: LoRARequest(name, i+1, path)
    for i, (name, path) in enumerate(lora_paths.items())
}

def make_prompt(pil_image, min_pixels, max_pixels, prompt_text):
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
    
    return prompt
    
# 기본 형태 API
@app.post("/chart_diagram_rec_default")
async def chart_diagram_rec_default(
    image: UploadFile = File(...),
    content_class: str = Form(...),
    token: HTTPAuthorizationCredentials = Depends(verify_token)
):
    allowed_classes = {"chart", "picture"}
    if content_class not in allowed_classes:
        return JSONResponse(
            content={
                "status": "failed",
                "message": f"Invalid content_class '{content_class}'. Allowed values: {', '.join(allowed_classes)}"
            },
            status_code=400  # 200으로 하면 오류 페이지 대신 정상 응답
        )
                
    # 이미지 판별
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image format")
    
    # 이미지 로드
    image_data = await image.read()
    image_stream = BytesIO(image_data)
    pil_image = Image.open(image_stream).convert("RGB")

    if content_class == "chart":
        prompt_text = "차트를 테이블로 변환해줘. 테이블만 출력해줘."
        min_pixels = 50000
        max_pixels = 1000000
        
        complexity_prompt_texts = [
            "Evaluate the chart’s structural complexity (axes, labels, legend density, number of series, visual clutter). Give one integer from 0 to 100. Output only the integer.",
            "Score the chart’s cognitive load from 0 (very simple) to 100 (very complex). Consider how hard it is to read all labels and compare values. Output only an integer.",
            "Rate the complexity of the chart (0–100) based on the quantity of elements: data points, text labels, gridlines, colors, and overlapping shapes. Output a single integer only.",
            "Give a complexity score between 0 and 100. Consider density, number of components, and overall readability. Return only a single integer with no explanation.",
            "Assess the chart’s complexity from 0 to 100. Do not explain. Do not use words. Output exactly one integer."
        ]
                
    elif content_class == "picture":
        prompt_text = "이 이미지에 대해 자세히 설명해 주세요."
        min_pixels = 50000
        max_pixels = 2000000

        complexity_prompt_texts = [
            "Rate the diagram's complexity from 0 to 100 using: 0–30 very simple, 31–60 moderate, 61–85 complex, 86–100 very complex. Output only an integer.",
            "From 0 (very simple) to 100 (very complex), how complex is this diagram? Output only an integer.",
            "Give a single integer in [0,100] for the diagram's complexity. No words, just the number.",
            "Score the diagram complexity 0-100 (see density of elements, labels, branches). Output integer only.",
            "Assess complexity (0–100). Consider quantity of elements, variety of connectors, labels, and layout depth. Integer only."
        ]

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
    
    try:
        image_inputs, video_inputs = process_vision_info(messages)
    except Exception as e:
        return JSONResponse(
            content={
                "status": "failed",
                "message": f"이미지 처리 중 오류 발생: {str(e)}"
            },
            status_code=400
        )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
    }
    
    complexity_prompts = {
        make_prompt(pil_image, min_pixels, max_pixels, c_p_t)
        for c_p_t in complexity_prompt_texts
    }
    
    complexity_inputs = [{"prompt": c_p, "multi_modal_data": mm_data} for c_p in complexity_prompts]

    generated_text = ""
    complexity_score = 0
    
    if content_class == "chart":
        # 모델 추론 및 시간 측정
        start_time = time.time()
        outputs = llm.generate([llm_inputs], sampling_params=chart_sampling_params,
            lora_request=lora_requests["chart_rec"]
            )
        end_time = time.time()
        generated_text = outputs[0].outputs[0].text
        
        complexity_outputs = llm.generate(
            complexity_inputs,
            sampling_params=chart_complexity_params,
            lora_request=lora_requests["chart_rec"],
        )
        
    elif content_class == "picture":
        # 모델 추론 및 시간 측정
        start_time = time.time()
        outputs = llm.generate([llm_inputs], sampling_params=diagram_sampling_params,
            lora_request=lora_requests["diagram_rec"]
            )
        end_time = time.time()
        generated_text = outputs[0].outputs[0].text
        generated_text = remove_duplicates(generated_text)
        generated_text = filter_specific_languages(generated_text)
        
        complexity_outputs = llm.generate(
            complexity_inputs,
            sampling_params=diagram_complexity_params,
            lora_request=lora_requests["diagram_rec"],
        )
    
    complexity_scores = []
    
    for c_output in complexity_outputs:
        if c_output and len(c_output.outputs) > 0:
            c_output_text = c_output.outputs[0].text.strip()
            c_output_numbers = re.findall(r'\d+', c_output_text)
            if c_output_numbers:
                val = int(c_output_numbers[0])
                val = max(0, min(100, val))
                complexity_scores.append(val)
    
    if complexity_scores:
        complexity_score = int(statistics.median(complexity_scores))
    
    response = {
            "status": "succeeded",
            "analyzeResult": {
                "content": generated_text,
                "complexity_score": complexity_score
            },
            # "inference_time": round(end_time - start_time, 2)
        }

    return JSONResponse(content=response)
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)