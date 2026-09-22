import cv2
import numpy as np
import os
from PIL import Image


def unsharp_mask(img: np.ndarray, radius: int = 3, amount: float = 1.5) -> np.ndarray:
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=float(radius))
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def preprocess_and_tile(
    image_path: str,
    *,
    target_dpi: int = 300,
    output_dir: str = "tiles",
    overlap: int = 20,
    use_unsharp: bool = True,      # вкл/выкл усиление резкости
    unsharp_radius: int = 3,
    unsharp_amount: float = 1.5,
    use_clahe: bool = False,       # вкл/выкл адаптивный контраст
    clahe_clip: float = 2.0,
) -> list[str]:
    pil_img = Image.open(image_path)
    img = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

    dpi_info = pil_img.info.get("dpi", (72, 72))
    current_dpi = int(dpi_info[0]) if dpi_info else 72

    h_orig, w_orig = img.shape[:2]
    if current_dpi <= 72:
        if w_orig > 2000:
            current_dpi = 300
        elif w_orig > 1000:
            current_dpi = 200
        else:
            current_dpi = 150

    scale = float(target_dpi) / float(current_dpi)
    if abs(scale - 1.0) > 0.01:
        new_w = int(round(w_orig * scale))
        new_h = int(round(h_orig * scale))
        img = cv2.resize(
            img,
            (new_w, new_h),
            interpolation=cv2.INTER_CUBIC,
        )

    h, w = img.shape[:2]

    # Deskew
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) > 100:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.3:
            M = cv2.getRotationMatrix2D(
                (float(w) / 2.0, float(h) / 2.0), float(angle), 1.0
            )
            img = cv2.warpAffine(
                img, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise (только на gray для поиска строк)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    noise_level = float(np.median(np.abs(laplacian)))

    if noise_level > 15:
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
    elif noise_level > 7:
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
    else:
        denoised = gray.copy()

    # Adaptive threshold — ТОЛЬКО для поиска строк
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 15,
    )

    # Поиск линий реза
    dark_pixels = (binary < 128).astype(np.float32)
    row_sums = np.sum(dark_pixels, axis=1)

    kernel_size = max(3, h // 200)
    if kernel_size % 2 == 0:
        kernel_size += 1
    row_sums_smooth = cv2.GaussianBlur(
        row_sums.reshape(-1, 1), (kernel_size, 1), 0
    ).flatten()

    rough_y1 = int(h * 0.33)
    rough_y2 = int(h * 0.67)

    def find_best_cut_line(rough_y, search_range, n_blocks=24):
        y_low = max(0, rough_y - search_range)
        y_high = min(h, rough_y + search_range)

        block_width = max(1, w // n_blocks)
        points = []

        for i in range(n_blocks):
            x_start = i * block_width
            if x_start >= w:
                break

            block_sums = row_sums_smooth[y_low:y_high]
            if len(block_sums) == 0:
                points.append((x_start, rough_y))
                continue

            local_min_idx = int(np.argmin(block_sums))
            best_y = y_low + local_min_idx
            points.append((x_start, best_y))

        if points:
            points.append((w, points[-1][1]))
        else:
            points = [(0, rough_y), (w, rough_y)]
        return points

    search_range = max(50, h // 20)
    cut_line_1 = find_best_cut_line(rough_y1, search_range)
    cut_line_2 = find_best_cut_line(rough_y2, search_range)

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    def cut_along_line(img_full, line_points, overlap_px=overlap):
        h_full, w_full = img_full.shape[:2]
        mask_upper = np.zeros((h_full, w_full), dtype=np.uint8)

        pts = np.array(line_points, dtype=np.int32)
        upper_poly = np.vstack([pts, [[w_full, 0]], [[0, 0]]])
        cv2.fillPoly(mask_upper, [upper_poly], 255)

        upper = cv2.bitwise_and(img_full, img_full, mask=mask_upper)
        lower_mask = cv2.bitwise_not(mask_upper)
        lower = cv2.bitwise_and(img_full, img_full, mask=lower_mask)

        max_y_upper = min(h_full, max(p[1] for p in line_points) + overlap_px)
        upper_cropped = upper[:max_y_upper]

        min_y_lower = max(0, min(p[1] for p in line_points) - overlap_px)
        lower_cropped = lower[min_y_lower:]

        return upper_cropped, lower_cropped

    tile1, remainder = cut_along_line(img, cut_line_1)

    offset_y = max(0, min(p[1] for p in cut_line_1) - overlap)
    cut_line_2_adj = [(x, y - offset_y) for x, y in cut_line_2 if y > offset_y]

    if not cut_line_2_adj:
        mid = remainder.shape[0] // 2
        tile2 = remainder[:mid]
        tile3 = remainder[mid:]
    else:
        tile2, tile3 = cut_along_line(remainder, cut_line_2_adj)

    tiles_data = [tile1, tile2, tile3]
    tile_paths = []

    for i, tile in enumerate(tiles_data, 1):
        # Обработка для OCR: denoise + unsharp + (опц.) clahe
        if len(tile.shape) == 3:
            tile_gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
        else:
            tile_gray = tile

        # Denoise финального тайла (если шум всё ещё заметен)
        lap_tile = cv2.Laplacian(tile_gray, cv2.CV_64F)
        noise_tile = float(np.median(np.abs(lap_tile)))
        if noise_tile > 10:
            tile_gray = cv2.fastNlMeansDenoising(tile_gray, h=12)

        # Unsharp mask — главное улучшение для размытых снимков
        if use_unsharp:
            tile_gray = unsharp_mask(tile_gray, radius=unsharp_radius, amount=unsharp_amount)

        # CLAHE — если бумага слишком серая
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
            tile_gray = clahe.apply(tile_gray)

        out_path = os.path.join(output_dir, f"{base_name}_tile{i}.png")
        cv2.imwrite(out_path, tile_gray)
        tile_paths.append(out_path)

    return tile_paths
