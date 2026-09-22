import cv2
import numpy as np
from PIL import Image

def prepare_book_scan(path: str, out_path: str, enable_deskew=True, enable_crop=False):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Не удалось прочитать изображение")

    original_h, original_w = img.shape[:2]

    # 1. Deskew (только если наклон заметный)
    if enable_deskew:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(thresh)

        if coords is not None and len(coords) > 100:  # достаточно точек
            rect = cv2.minAreaRect(coords)
            angle = rect[2]  # угол от -90 до 0

            # Коррекция угла: делаем так, чтобы он был близок к 0
            if angle < -45:
                angle = 90 + angle
            else:
                angle = angle

            # Поворачиваем только если угол больше порога (например, 2 градуса)
            abs_angle = abs(angle)
            if abs_angle > 2.0:
                h, w = img.shape[:2]
                center = (w / 2, h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                # Используем BORDER_CONSTANT с белым цветом, чтобы не было чёрных полос
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

    # 2. Обрезка полей (ОПАСНО: лучше сначала протестировать на одном изображении)
    if enable_crop:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Чуть мягче порог, чтобы не отрезать текст у края
        _, thresh = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            margin = 15
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            img = img[y:y+h, x:x+w]

    # 3. Изменение размера: длинная сторона 1600 px, пропорции сохраняются
    h, w = img.shape[:2]
    target_long = 1600
    scale = target_long / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 4. Конвертация в RGB и сохранение
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    pil_img.save(out_path, format="PNG", optimize=True)