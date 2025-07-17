from PIL import Image
from pathlib import Path
from collections import defaultdict

def analyze_image_sizes_by_range(directory, bin_size=400):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    widths = []
    heights = []
    bins = defaultdict(int)

    directory = Path(directory)

    for img_path in directory.iterdir():
        if img_path.suffix.lower() in image_extensions:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)

                    max_dim = max(width, height)
                    bin_index = max_dim // bin_size
                    bins[bin_index] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if not widths:
        print("No valid images found.")
        return

    # 구간별 출력
    print("\n[크기별 이미지 수]")
    for bin_index in sorted(bins):
        start = bin_index * bin_size
        end = (bin_index + 1) * bin_size
        print(f"{start:4d} ~ {end:4d}: {bins[bin_index]}장")

    # 평균 크기 출력
    avg_width = sum(widths) / len(widths)
    avg_height = sum(heights) / len(heights)
    print(f"\n[평균 크기]")
    print(f"Average Width: {avg_width:.2f}, Average Height: {avg_height:.2f}")

def count_large_images(directory, pixel_threshold=1003520):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    directory = Path(directory)

    total_images = 0
    large_images = []

    for img_path in directory.iterdir():
        if img_path.suffix.lower() in image_extensions:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    total_images += 1
                    num_pixels = width * height
                    if num_pixels > pixel_threshold:
                        excess = num_pixels - pixel_threshold
                        large_images.append((img_path.name, width, height, num_pixels, excess))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"총 이미지 수: {total_images}장")
    print(f"{pixel_threshold:,} 픽셀 초과 이미지 수: {len(large_images)}장")
    if total_images > 0:
        print(f"비율: {(len(large_images) / total_images) * 100:.2f}%")

    if large_images:
        print("\n[초과 이미지 목록]")
        for name, w, h, pixels, excess in large_images:
            print(f"{name:40s} | {w:4d}x{h:4d} = {pixels:8,d} px | 초과: {excess:8,d} px")

# 사용 예시
analyze_image_sizes_by_range("/home/hskim/Develop/chart_recog/test_data/chart_test_v2_0/images")

# 사용 예시
count_large_images("/home/hskim/Develop/chart_recog/test_data/chart_test_v2_0/images")