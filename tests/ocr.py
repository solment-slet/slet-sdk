import asyncio
import pathlib

import cv2
import numpy as np
from slet_sdk import SletClient

from prepare import preprocess_and_tile


def deskew(img):
    """Удаляет перекос страницы."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    # Берём самый большой контур
    c = max(contours, key=cv2.contourArea)

    # Минимальный прямоугольник с поворотом
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int32)

    # Вычисляем угол и поворачиваем
    angle = rect[2]
    if angle < -45:
        angle = 90 + angle

    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def fix_perspective(img):
    """Выпрямляет страницу, снятую под углом (нужны 4 угла листа)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    c = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    # Нужен четырёхугольник
    if len(approx) != 4:
        return img

    # Сортируем точки: верх-лево, верх-право, низ-право, низ-лево
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # Вычисляем размеры выходного изображения
    width = max(int(np.linalg.norm(rect[0] - rect[1])),
                int(np.linalg.norm(rect[2] - rect[3])))
    height = max(int(np.linalg.norm(rect[0] - rect[3])),
                 int(np.linalg.norm(rect[1] - rect[2])))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def remove_shadows(gray):
    """Убирает тени и выравнивает фон через разность с морфологическим 'дном'."""
    # Морфологическое «дно» — аппроксимирует фон (то, что за буквами)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Вычитаем фон, выравнивая освещение
    diff = cv2.absdiff(gray, bg)

    # Инвертируем: тёмные буквы на светлом фоне → светлые на тёмном
    norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
    norm = 255 - norm
    return norm


def preprocess_universal(img, output_path="prepared.png"):
    """Универсальная предобработка для фото страниц с телефона или камеры."""
    img = img

    # --- 1. Уменьшаем, если слишком большой (ускоряет и стабилизирует) ---
    h, w = img.shape[:2]
    max_side = 2000  # максимальная сторона ~2000 px — достаточно для OCR
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    # --- 2. Слегка размываем (убирает шум от камеры) ---
    img = cv2.GaussianBlur(img, (3, 3), 0)

    img = fix_perspective(img)

    # --- 3. Выравниваем перекос страницы ---
    img = deskew(img)

    # --- 4. В оттенки серого ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 5. Удаляем тени и выравниваем фон ---
    gray = remove_shadows(gray)

    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=25,
        C=10
    )

    cv2.imwrite(output_path, binary)
    return output_path


path = pathlib.Path("/home/esolment/Изображения/14G.jpg")

from prepare_book import prepare_book_scan

prepare_book_scan(str(path), "prepared.png")

output = preprocess_and_tile(str(path), target_dpi=300, use_clahe=True)

print(output)

async def main():
    async with SletClient(base_url="http://localhost:8080", ssl_verify=False) as c:
        await c.signin("testesolment@gmail.com", "20132061netT%")

        print(c.access_token)
        return

        ocr = await c.aelite.ocr.extract_text(
            path,
            combine_line_breaks=True,
        )
        print(ocr.plain_text)
        print()
        print(ocr)
        print()


asyncio.run(main())