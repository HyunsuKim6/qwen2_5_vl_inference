from PIL import Image
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from collections import defaultdict

def analyze_image_sizes_by_multiple_ranges(directory, bin_sizes=(100_000, 1_000_000),
                                           save_chart=True, chart_dir="charts",
                                           max_pixel_xaxis=10000000):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    pixel_counts = []
    bin_maps = {bin_size: defaultdict(int) for bin_size in bin_sizes}

    directory = Path(directory)
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    for img_path in directory.iterdir():
        if img_path.suffix.lower() in image_extensions:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    num_pixels = width * height
                    pixel_counts.append(num_pixels)

                    for bin_size in bin_sizes:
                        bin_index = num_pixels // bin_size
                        bin_maps[bin_size][bin_index] += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if not pixel_counts:
        print("No valid images found.")
        return

    # 각 bin_size에 대해 출력 및 시각화
    for bin_size in bin_sizes:
        print(f"\n[픽셀 수 기준: {bin_size:,} 픽셀 단위]")
        bins = bin_maps[bin_size]

        bin_labels = []
        bin_counts = []
        for bin_index in sorted(bins):
            start = bin_index * bin_size
            end = (bin_index + 1) * bin_size

            if end > max_pixel_xaxis:
                continue  # 200만 픽셀 초과 bin은 제외

            count = bins[bin_index]
            bin_labels.append(f"~{end}")  # 간결한 라벨
            bin_counts.append(count)
            print(f"{start:10,d} ~ {end:10,d} px: {count}장")

        # 차트 저장
        if save_chart:
            plt.figure(figsize=(14, 6))
            plt.bar(bin_labels, bin_counts, color='skyblue')
            plt.title(f"Image Size Distribution (Bin: {bin_size:,} px, Max X: {max_pixel_xaxis:,} px)")
            plt.xlabel("Pixel Range")
            plt.ylabel("Number of Images")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            save_path = chart_dir / f"image_size_distribution_{bin_size//1000}k_upto_{max_pixel_xaxis//1000}k.png"
            plt.savefig(save_path)
            plt.close()
            print(f"✅ 차트 저장됨: {save_path}")

    # 평균 출력
    avg_pixels = sum(pixel_counts) / len(pixel_counts)
    print(f"\n[평균 픽셀 수]")
    print(f"Average Pixels: {avg_pixels:,.2f}")

# def analyze_image_sizes_width_height_by_range(directory, bin_size=400):
#     image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
#     widths = []
#     heights = []
#     bins = defaultdict(int)

#     directory = Path(directory)

#     for img_path in directory.iterdir():
#         if img_path.suffix.lower() in image_extensions:
#             try:
#                 with Image.open(img_path) as img:
#                     width, height = img.size
#                     widths.append(width)
#                     heights.append(height)

#                     max_dim = max(width, height)
#                     bin_index = max_dim // bin_size
#                     bins[bin_index] += 1
#             except Exception as e:
#                 print(f"Error processing {img_path}: {e}")

#     if not widths:
#         print("No valid images found.")
#         return

#     # 구간별 출력
#     print("\n[크기별 이미지 수]")
#     for bin_index in sorted(bins):
#         start = bin_index * bin_size
#         end = (bin_index + 1) * bin_size
#         print(f"{start:4d} ~ {end:4d}: {bins[bin_index]}장")

#     # 평균 크기 출력
#     avg_width = sum(widths) / len(widths)
#     avg_height = sum(heights) / len(heights)
#     print(f"\n[평균 크기]")
#     print(f"Average Width: {avg_width:.2f}, Average Height: {avg_height:.2f}")

def count_large_images(directory, pixel_threshold=1200000):
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

    # if large_images:
    #     print("\n[초과 이미지 목록]")
    #     for name, w, h, pixels, excess in large_images:
    #         print(f"{name:40s} | {w:4d}x{h:4d} = {pixels:8,d} px | 초과: {excess:8,d} px")

def count_small_images(directory, pixel_threshold=50000):
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'}
    directory = Path(directory)

    total_images = 0
    small_images = []

    for img_path in directory.iterdir():
        if img_path.suffix.lower() in image_extensions:
            try:
                with Image.open(img_path) as img:
                    width, height = img.size
                    total_images += 1
                    num_pixels = width * height
                    if num_pixels < pixel_threshold:
                        shortfall = pixel_threshold - num_pixels
                        small_images.append((img_path.name, width, height, num_pixels, shortfall))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"총 이미지 수: {total_images}장")
    print(f"{pixel_threshold:,} 픽셀 미만 이미지 수: {len(small_images)}장")
    if total_images > 0:
        print(f"비율: {(len(small_images) / total_images) * 100:.2f}%")

    # if small_images:
    #     print("\n[미만 이미지 목록]")
    #     for name, w, h, pixels, shortfall in small_images:
    #         print(f"{name:40s} | {w:4d}x{h:4d} = {pixels:8,d} px | 부족: {shortfall:8,d} px")

# 사용 예시
# analyze_image_sizes_by_multiple_ranges(
#     "/media/hskim/disk/크라우드웍스/차트 인식 데이터/최종 결과물/train_images/images_all",
#     bin_sizes=(100_000, 1_000_000),
#     save_chart=True,
#     chart_dir="./",
#     max_pixel_xaxis=3000000  # 2000k까지만 시각화
# )

# 사용 예시
count_large_images("/media/hskim/disk/크라우드웍스/차트 인식 데이터/최종 결과물/train_images/images_all", pixel_threshold=1000000)

count_small_images("/media/hskim/disk/크라우드웍스/차트 인식 데이터/최종 결과물/train_images/images_all", pixel_threshold=50000)