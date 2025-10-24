from safetensors import safe_open
import torch
import os

def compare_safetensors(file1, file2, atol=1e-6):
    """
    두 safetensors 파일의 파라미터를 비교합니다.
    atol: float 허용 오차 (float값 비교용)
    """
    if not os.path.exists(file1) or not os.path.exists(file2):
        raise FileNotFoundError("두 파일 중 하나가 존재하지 않습니다.")

    with safe_open(file1, framework="pt") as f1, safe_open(file2, framework="pt") as f2:
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())

        # 키(파라미터 이름) 비교
        if keys1 != keys2:
            print("⚠️ 키(파라미터 이름)가 다릅니다.")
            print("파일1에만 있는 키:", keys1 - keys2)
            print("파일2에만 있는 키:", keys2 - keys1)
            return False

        # 각 파라미터별 비교
        all_equal = True
        for key in sorted(keys1):
            t1 = f1.get_tensor(key)
            t2 = f2.get_tensor(key)

            if t1.shape != t2.shape:
                print(f"❌ {key}: shape 다름 ({t1.shape} vs {t2.shape})")
                all_equal = False
                continue

            if not torch.allclose(t1, t2, atol=atol):
                diff = torch.max(torch.abs(t1 - t2)).item()
                print(f"❌ {key}: 값 다름 (최대 차이 {diff:.6f})")
                all_equal = False

        if all_equal:
            print("✅ 두 safetensors 파일의 내용이 완전히 같습니다.")
        else:
            print("⚠️ 일부 파라미터가 다릅니다.")
        return all_equal


# 사용 예시
compare_safetensors("./model_weight/aihub_data_v1_1_crowdworks/qwen2_5_vl_3B_qlora/nf4_dq_bf16_total_3epoch/llamafactory_3epoch/adapter_model.safetensors", "./model_weight/aihub_data_v1_1_crowdworks/qwen2_5_vl_3B_qlora/nf4_dq_bf16/checkpoint-60000/adapter_model.safetensors")