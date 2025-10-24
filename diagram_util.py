import re

def remove_duplicates(text: str) -> str:
    """
    텍스트에서 중복된 단어나 문장을 제거합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        중복이 제거된 텍스트
    """
    if not text or not text.strip():
        return text
    
    # 1. 반복 패턴 제거 (예: "모델" (1) - "모델" (2) - "모델" (3)...)
    text = remove_repetitive_patterns(text)
    
    # 2. 문장 단위로 분리 (마침표, 느낌표, 물음표 기준)
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # 3. 중복 제거 (순서 유지)
    seen_sentences = set()
    unique_sentences = []
    
    for sentence in sentences:
        # 문장을 정규화 (공백 제거, 소문자 변환)
        normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
        if normalized not in seen_sentences:
            seen_sentences.add(normalized)
            unique_sentences.append(sentence)
    
    # 4. 단어 단위 중복도 확인 (같은 단어가 연속으로 반복되는 경우)
    if unique_sentences:
        # 마지막 문장에서 연속 중복 단어 제거
        last_sentence = unique_sentences[-1]
        words = last_sentence.split()
        deduplicated_words = []
        prev_word = None
        
        for word in words:
            if word.lower() != prev_word:
                deduplicated_words.append(word)
                prev_word = word.lower()
        
        unique_sentences[-1] = ' '.join(deduplicated_words)
    
    # 5. 문장들을 다시 조합
    result = '. '.join(unique_sentences)
    
    # 6. 원본 텍스트가 마침표로 끝나지 않았다면 마침표 추가
    if text.strip().endswith(('.', '!', '?')):
        result += text.strip()[-1]
    
    return result

def remove_repetitive_patterns(text: str) -> str:
    """
    반복되는 패턴을 제거합니다.
    예: "모델" (1) - "모델" (2) - "모델" (3)... -> "모델" (1) - "모델" (2) - "모델" (3)"
    
    Args:
        text: 원본 텍스트
        
    Returns:
        반복 패턴이 제거된 텍스트
    """
    if not text or not text.strip():
        return text
    
    # 패턴 1: "텍스트" (숫자) - "텍스트" (숫자) - "텍스트" (숫자)...
    # 예: "모델" (1) - "모델" (2) - "모델" (3)...
    pattern1 = r'("([^"]+)"\s*\(\d+\)\s*-\s*"([^"]+)"\s*\(\d+\)\s*-\s*"([^"]+)"\s*\(\d+\))(?:\s*-\s*"([^"]+)"\s*\(\d+\))*'
    
    def replace_pattern1(match):
        base_text = match.group(2)  # 첫 번째 따옴표 안의 텍스트
        # 패턴이 3번 이상 반복되면 처음 3개만 유지
        if base_text == match.group(3) == match.group(4):
            return match.group(1)
        return match.group(0)
    
    text = re.sub(pattern1, replace_pattern1, text)
    
    # 패턴 2: 일반적인 반복 패턴 (3번 이상 반복되는 경우)
    # 예: "단어1 단어2 단어3" - "단어1 단어2 단어3" - "단어1 단어2 단어3"...
    pattern2 = r'((?:[^-\n]+?)(?:\s*-\s*[^-\n]+?){2,})(?:\s*-\s*[^-\n]+?)*'
    
    def replace_pattern2(match):
        parts = [part.strip() for part in match.group(0).split(' - ')]
        if len(parts) >= 3:
            # 처음 3개 부분이 모두 동일한지 확인
            first_part = parts[0].strip()
            if all(part.strip() == first_part for part in parts[1:3]):
                # 3번 이상 반복되면 처음 3개만 유지
                return ' - '.join(parts[:3])
        return match.group(0)
    
    text = re.sub(pattern2, replace_pattern2, text)
    
    # 패턴 3: 숫자 증가 패턴 (예: 1, 2, 3, 4, 5...)
    # 10개 이상 연속으로 증가하는 숫자 패턴을 1, 2, 3...으로 축약
    pattern3 = r'(\d+)(?:\s*,\s*\d+){9,}'
    
    def replace_pattern3(match):
        first_num = match.group(1)
        return f"{first_num}, ..."
    
    text = re.sub(pattern3, replace_pattern3, text)
    
    # 패턴 4: 동일한 단어나 구문이 3번 이상 연속 반복
    # 예: "모델 모델 모델 모델..." -> "모델"
    pattern4 = r'(\b\w+\b)(?:\s+\1){2,}'
    
    def replace_pattern4(match):
        return match.group(1)
    
    text = re.sub(pattern4, replace_pattern4, text)
    
    # 패턴 5: 숫자와 함께 나열되는 동일한 단어 패턴
    # 예: "모델1 모델2 모델3 모델4..." -> "모델1 모델2 모델3..."
    # 숫자가 앞에 오는 경우: "1모델 2모델 3모델 4모델..." -> "1모델 2모델 3모델..."
    # 숫자 옆에 점이 붙는 경우: "모델1. 모델2. 모델3. 모델4." -> "모델1. 모델2. 모델3."
    pattern5 = r'(\b\w+\d+\.?\b)(?:\s+\1){2,}'
    
    def replace_pattern5(match):
        parts = match.group(0).split()
        if len(parts) >= 3:
            # 처음 3개만 유지
            return ' '.join(parts[:3])
        return match.group(0)
    
    text = re.sub(pattern5, replace_pattern5, text)
    
    # 패턴 6: 숫자가 앞에 오는 동일한 단어 패턴
    # 예: "1모델 2모델 3모델 4모델..." -> "1모델 2모델 3모델..."
    pattern6 = r'(\b\d+\.?\w+\b)(?:\s+\1){2,}'
    
    def replace_pattern6(match):
        parts = match.group(0).split()
        if len(parts) >= 3:
            # 처음 3개만 유지
            return ' '.join(parts[:3])
        return match.group(0)
    
    text = re.sub(pattern6, replace_pattern6, text)
    
    return text

def filter_specific_languages(text):
    """
    텍스트에서 특정 언어의 문자들을 제거합니다.
    중국어, 일본어, 아랍어, 러시아어만 필터링하고 한국어는 유지합니다.

    Args:
        text: 필터링할 텍스트

    Returns:
        filtered_text: 특정 언어 문자가 제거된 텍스트 (한국어는 유지)
    """
    # 필터링할 언어의 유니코드 범위만 포함 (한국어는 제외)
    language_patterns = [
        # 중국어/한자
        r"[\u4e00-\u9fff]",  # CJK Unified Ideographs
        r"[\u3400-\u4dbf]",  # CJK Extension A
        r"[\u2e80-\u2eff]",  # CJK Radicals Supplement
        r"[\u2f00-\u2fdf]",  # Kangxi Radicals
        r"[\u31c0-\u31ef]",  # CJK Strokes
        r"[\u3200-\u32ff]",  # Enclosed CJK Letters and Months
        r"[\u3300-\u33ff]",  # CJK Compatibility
        r"[\uf900-\ufaff]",  # CJK Compatibility Ideographs
        r"[\uff00-\uffef]",  # Halfwidth and Fullwidth Forms
        # 일본어 히라가나/가타카나
        r"[\u3040-\u309f]",  # Hiragana
        r"[\u30a0-\u30ff]",  # Katakana
        r"[\u31f0-\u31ff]",  # Katakana Phonetic Extensions
        # 아랍어
        r"[\u0600-\u06ff]",  # Arabic
        r"[\u0750-\u077f]",  # Arabic Supplement
        r"[\u08a0-\u08ff]",  # Arabic Extended-A
        r"[\ufb50-\ufdff]",  # Arabic Presentation Forms-A
        r"[\ufe70-\ufeff]",  # Arabic Presentation Forms-B
        # 러시아어/키릴 문자
        r"[\u0400-\u04ff]",  # Cyrillic
        r"[\u0500-\u052f]",  # Cyrillic Supplement
    ]

    # 모든 패턴을 하나의 정규식으로 결합
    combined_pattern = "|".join(language_patterns)
    language_pattern = re.compile(combined_pattern)

    # 특정 언어 문자를 제거
    filtered_text = language_pattern.sub("", text)

    return filtered_text