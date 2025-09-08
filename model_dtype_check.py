from safetensors import safe_open


import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "./model_weight/qwen2_5_vl_3B_AWQ_huggingface_download"
ADAPTER_NAME = "my_lora"
ADAPTER_PATH = "./model_weight/aihub_data_v1_1_crowdworks/qwen2_5_vl_3B_lora/llamafactory_maxpx_1M_5epoch"


llm = LLM(
    model=MODEL_PATH,
    enable_lora=True,            # LoRA 동적 로드 허용
    # dtype="bfloat16",          # vLLM 버전에 따라 옵션명 다름. 필요 시 지정
    # quantization="AWQ",        # 사전 양자화 모델일 때만 힌트로 사용
)

# vLLM에 어댑터 등록 (요청마다 다르게 쓸 수도 있음)
lora_req = LoRARequest(
    lora_request_id=1,
    adapter_name=ADAPTER_NAME,
    adapter_path=ADAPTER_PATH,
)

# 내부 torch model 추출(버전별 후보 경로를 순차 시도)
def _get_possible_torch_modules_from_vllm(llm):
    cands = []
    eng = getattr(llm, "llm_engine", None)
    if eng is None:
        return []

    # 1) 단일 워커 모델 러너
    try:
        cands.append(eng.model_executor.driver_worker.model_runner.model)
    except Exception:
        pass

    # 2) 멀티 워커: 각 워커의 모델 러너
    try:
        workers = eng.model_executor.driver_workers
        for w in workers:
            try:
                cands.append(w.model_runner.model)
            except Exception:
                pass
    except Exception:
        pass

    # 3) 다른 버전 경로들(있으면 추가)
    for attr in ["model", "model_runner", "executor", "workers"]:
        try:
            obj = getattr(eng, attr, None)
            if obj is not None:
                cands.append(obj)
        except Exception:
            pass

    # 중복 제거
    uniq = []
    seen = set()
    for m in cands:
        if m is not None and id(m) not in seen:
            uniq.append(m)
            seen.add(id(m))
    return uniq

def print_lora_dtypes_from_vllm(llm):
    print("=== LoRA dtype check (vLLM runtime, best-effort) ===")
    mods = _get_possible_torch_modules_from_vllm(llm)
    if not mods:
        print("[WARN] Could not locate underlying torch module(s) from vLLM. "
              "Version may hide internals. Try the Transformers/PEFT method.")
        return
    found = False
    for idx, mod in enumerate(mods):
        # 모듈처럼 보이는 객체만 골라 named_parameters() 시도
        for attr in ["named_parameters", "parameters"]:
            if hasattr(mod, attr):
                try:
                    for name, p in mod.named_parameters():
                        if "lora" in name.lower():
                            print(f"[m{idx}] {name:70s} {str(p.dtype):>12s} {str(p.device)}")
                            found = True
                    break
                except Exception:
                    pass
    if not found:
        print("[INFO] No visible 'lora' parameters found in exposed modules. "
              "vLLM may manage LoRA weights outside the exposed torch module for your version.")

# 어댑터 로드가 실제로 적용된 상태에서 dtype 확인을 원하면,
# 간단히 한 번 호출(프롬프트는 짧게)하여 로딩을 트리거해두면 도움이 될 때가 있습니다.
_ = llm.generate(
    prompts=["Hello"],
    sampling_params=SamplingParams(max_tokens=1),
    lora_request=lora_req,
)

print_lora_dtypes_from_vllm(llm)