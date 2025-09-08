#!/usr/bin/env python3
# compare_weights.py
import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from tqdm import tqdm

try:
    from safetensors import safe_open
except Exception:
    safe_open = None


def is_safetensors_file(p: Path) -> bool:
    return p.suffix == ".safetensors"


def is_bin_file(p: Path) -> bool:
    return p.suffix in {".bin", ".pt", ".pth"}


def find_index_json(dir_path: Path) -> Optional[Path]:
    cand = dir_path / "model.safetensors.index.json"
    return cand if cand.exists() else None


class WeightSource:
    """
    추상화된 weight 소스:
      - 단일 safetensors 파일
      - 여러 개의 safetensors 샤드(인덱스 json 기반)
      - pytorch .bin/.pt/.pth
    get_tensor(name) 호출 시에만 로드 -> 메모리 절약
    """

    def __init__(self, path: Path):
        self.path = path
        self.kind = None  # "single_st", "sharded_st", "pt_bin"
        self.weight_map: Dict[str, Path] = {}
        self._names_cache: Optional[Set[str]] = None

        if path.is_file() and is_safetensors_file(path):
            self.kind = "single_st"
            # 미리 텐서 이름을 캐시
            self._names_cache = set()
            with safe_open(str(path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    self.weight_map[k] = path
                    self._names_cache.add(k)

        elif path.is_dir() and find_index_json(path):
            self.kind = "sharded_st"
            index_path = find_index_json(path)
            j = json.loads(index_path.read_text())
            # "weight_map": {tensor_name: "pytorch_model-00001-of-00016.safetensors", ...}
            base = index_path.parent
            for k, rel in j.get("weight_map", {}).items():
                self.weight_map[k] = base / rel
            self._names_cache = set(self.weight_map.keys())

        elif path.is_dir():
            # safetensors 샤드 인덱스가 없지만, 디렉터리에 *.safetensors 가 있는 케이스
            sts = sorted(path.glob("*.safetensors"))
            if sts and safe_open is None:
                raise RuntimeError("safetensors 가 설치되어야 합니다: pip install safetensors")
            if sts:
                self.kind = "multi_st_no_index"
                # 각 파일의 키를 읽어 합집합
                self._names_cache = set()
                for st in sts:
                    with safe_open(str(st), framework="pt", device="cpu") as f:
                        for k in f.keys():
                            # 마지막에 본 파일로 매핑 (중복이 있으면 경고)
                            if k in self.weight_map:
                                print(f"[WARN] duplicate tensor key {k} in {st.name} (keeping latest)", file=sys.stderr)
                            self.weight_map[k] = st
                            self._names_cache.add(k)
            else:
                # .bin류가 있을 수도
                bins = sorted([p for p in path.iterdir() if is_bin_file(p)])
                if bins:
                    self.kind = "pt_bin_dir"
                    # 하나의 거대한 state_dict가 여러 파일에 나뉘었을 수도 있으니 모두 병합(느림)
                    # -> names() 호출 시 한 번만 로드해 키 목록 캐시
                    self._names_cache = None
                else:
                    raise FileNotFoundError(f"지원되는 weight 파일을 찾지 못했습니다: {path}")

        elif path.is_file() and is_bin_file(path):
            self.kind = "pt_bin"
            self._names_cache = None
        else:
            raise FileNotFoundError(f"지원되지 않는 경로/형식: {path}")

    def names(self) -> Set[str]:
        if self._names_cache is not None:
            return set(self._names_cache)
        # pt_bin*(단일/디렉터리) 의 경우 여기서 한번 불러 키 목록만 캐시
        keys = set()
        if self.kind == "pt_bin":
            sd = torch.load(self.path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            keys = set(sd.keys())
            sd = None
        elif self.kind == "pt_bin_dir":
            for p in sorted(self.path.iterdir()):
                if not is_bin_file(p):
                    continue
                sd = torch.load(p, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                keys.update(sd.keys())
                sd = None
        else:
            # safetensors 계열은 이미 캐시됨
            keys = set(self._names_cache or [])
        self._names_cache = set(keys)
        return set(self._names_cache)

    def get_meta(self, name: str) -> Tuple[torch.dtype, Tuple[int, ...]]:
        """dtype, shape 만 빠르게 반환 (safetensors는 전체 로드 없이 meta 조회 가능)"""
        if self.kind in {"single_st", "sharded_st", "multi_st_no_index"}:
            st_path = self.weight_map[name]
            with safe_open(str(st_path), framework="pt", device="cpu") as f:
                info = f.get_tensor(name)  # 이 시점에 load
                return (info.dtype, tuple(info.shape))
        else:
            # pt_bin
            t = self.get_tensor(name)  # 어쩔 수 없이 로드
            meta = (t.dtype, tuple(t.shape))
            del t
            return meta

    def get_tensor(self, name: str) -> torch.Tensor:
        """실제 텐서 값 CPU로 반환"""
        if self.kind in {"single_st", "sharded_st", "multi_st_no_index"}:
            st_path = self.weight_map[name]
            with safe_open(str(st_path), framework="pt", device="cpu") as f:
                return f.get_tensor(name)
        elif self.kind == "pt_bin":
            sd = torch.load(self.path, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                sd = sd["state_dict"]
            t = sd[name]
            return t if isinstance(t, torch.Tensor) else torch.tensor(t)
        elif self.kind == "pt_bin_dir":
            # 여러 파일에 흩어져 있을 수 있으니 파일을 순회하며 찾음
            for p in sorted(self.path.iterdir()):
                if not is_bin_file(p):
                    continue
                sd = torch.load(p, map_location="cpu")
                if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
                    sd = sd["state_dict"]
                if name in sd:
                    t = sd[name]
                    return t if isinstance(t, torch.Tensor) else torch.tensor(t)
            raise KeyError(f"{name} not found in {self.path}")
        else:
            raise RuntimeError("unknown kind")


def sha256_tensor(t: torch.Tensor) -> str:
    # CPU float-contiguous 보장 후 바이트로 해시
    if not t.device.type == "cpu":
        t = t.cpu()
    t = t.contiguous()
    h = hashlib.sha256(t.numpy().tobytes()).hexdigest()
    return h


def compare_two_sources(a: WeightSource, b: WeightSource, atol: float, rtol: float,
                        checksum_only: bool = False, limit: Optional[int] = None):
    names_a = a.names()
    names_b = b.names()

    common = sorted(names_a & names_b)
    only_a = sorted(names_a - names_b)
    only_b = sorted(names_b - names_a)

    print(f"# 텐서 이름 요약")
    print(f"- 공통 텐서 수: {len(common)}")
    print(f"- A에만 존재: {len(only_a)}")
    print(f"- B에만 존재: {len(only_b)}")
    if only_a:
        print(f"  · A-only (앞 20개): {only_a[:20]}")
    if only_b:
        print(f"  · B-only (앞 20개): {only_b[:20]}")
    print()

    meta_mismatch = []
    value_mismatch = []

    bar = tqdm(common if limit is None else common[:limit], desc="비교 중", unit="tensor")
    equal_count = 0
    checksum_equal = 0

    for name in bar:
        try:
            dt_a, sh_a = a.get_meta(name)
            dt_b, sh_b = b.get_meta(name)
        except Exception as e:
            meta_mismatch.append((name, f"메타 조회 실패: {e}"))
            continue

        if (dt_a != dt_b) or (tuple(sh_a) != tuple(sh_b)):
            meta_mismatch.append((name, f"dtype/shape 불일치: A={dt_a}/{sh_a}, B={dt_b}/{sh_b}"))
            continue

        if checksum_only:
            try:
                ta = a.get_tensor(name)
                tb = b.get_tensor(name)
                ha = sha256_tensor(ta)
                hb = sha256_tensor(tb)
                if ha == hb:
                    checksum_equal += 1
                else:
                    value_mismatch.append((name, "checksum 불일치"))
                del ta, tb
            except Exception as e:
                value_mismatch.append((name, f"체크섬 비교 실패: {e}"))
            continue

        # 실제 수치 비교
        try:
            ta = a.get_tensor(name).cpu()
            tb = b.get_tensor(name).cpu()
            if ta.dtype != tb.dtype:
                # 이미 위에서 meta 체크했지만 안전망
                meta_mismatch.append((name, f"dtype 상이(재검증): {ta.dtype} vs {tb.dtype}"))
                del ta, tb
                continue

            # 허용오차 내 동일성 체크
            same = torch.allclose(ta, tb, atol=atol, rtol=rtol)
            diff = (ta - tb).abs()
            max_abs = diff.max().item() if diff.numel() else 0.0
            mean_abs = diff.mean().item() if diff.numel() else 0.0
            l2 = torch.linalg.vector_norm(diff).item() if diff.numel() else 0.0

            if same:
                equal_count += 1
            else:
                value_mismatch.append((name, f"max|Δ|={max_abs:.3e}, mean|Δ|={mean_abs:.3e}, ||Δ||2={l2:.3e}"))

            del ta, tb, diff
        except Exception as e:
            value_mismatch.append((name, f"수치 비교 실패: {e}"))

    print("\n# 결과 요약")
    if checksum_only:
        print(f"- 체크섬 동일: {checksum_equal}/{len(common)}")
    else:
        print(f"- 값이 허용오차 내 동일: {equal_count}/{len(common)}")
    print(f"- 메타데이터(shape/dtype) 불일치: {len(meta_mismatch)}")
    print(f"- 값 불일치: {len(value_mismatch)}")

    if meta_mismatch:
        print("\n## 메타데이터 불일치 목록(상위 50)")
        for n, msg in meta_mismatch[:50]:
            print(f"- {n}: {msg}")

    if value_mismatch:
        print("\n## 값 불일치 목록(상위 50)")
        for n, msg in value_mismatch[:50]:
            print(f"- {n}: {msg}")


def main():
    parser = argparse.ArgumentParser(description="두 모델/샤드 weight 비교 도구")
    parser.add_argument("path_a", type=str, help="A 모델 경로(파일 또는 디렉토리)")
    parser.add_argument("path_b", type=str, help="B 모델 경로(파일 또는 디렉토리)")
    parser.add_argument("--atol", type=float, default=0.0, help="절대 오차 허용치 (torch.allclose)")
    parser.add_argument("--rtol", type=float, default=0.0, help="상대 오차 허용치 (torch.allclose)")
    parser.add_argument("--checksum-only", action="store_true", help="SHA256 체크섬 비교만 수행(빠름, 차이 유무 확인용)")
    parser.add_argument("--limit", type=int, default=None, help="비교할 공통 텐서 개수 제한(디버그/속도용)")
    args = parser.parse_args()

    pa = Path(args.path_a)
    pb = Path(args.path_b)

    if safe_open is None:
        # safetensors가 전혀 필요 없을 수도 있지만, 보통 필요하니 안내
        print("[INFO] safetensors 미설치. safetensors 파일 비교 시 오류가 날 수 있습니다. pip install safetensors", file=sys.stderr)

    src_a = WeightSource(pa)
    src_b = WeightSource(pb)

    compare_two_sources(src_a, src_b, atol=args.atol, rtol=args.rtol,
                        checksum_only=args.checksum_only, limit=args.limit)


if __name__ == "__main__":
    main()