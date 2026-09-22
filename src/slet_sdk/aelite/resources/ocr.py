import pathlib
from io import BytesIO
from typing import Union

from slet_sdk.core.mixins import BaseResource
from slet_sdk.aelite.schemas.ocr import OCRResponse

ImageSource = Union[bytes, bytearray, pathlib.Path]


class OCRResource(BaseResource):
    async def extract_text(
        self,
        image: ImageSource,
        *,
        combine_line_breaks: bool = False,
    ) -> OCRResponse:
        """
        Extract text from an image via OCR.

        Parameters
        ----------
        image : bytes, bytearray, or pathlib.Path
            Raw image bytes or a path to an image file.
            Supported formats: PNG, JPEG, TIFF, BMP, WEBP.
        combine_line_breaks : bool
            Whether to merge consecutive line breaks in the output.

        Returns
        -------
        OCRResponse
            Parsed OCR result from the API.

        Raises
        ------
        ValueError
            If the image format is unrecognized or unsupported.
        TypeError
            If ``image`` is not bytes, bytearray, or pathlib.Path.
        """
        files_payload: dict = {}

        if isinstance(image, (bytes, bytearray)):
            data = bytes(image)
            ext, mime = self._detect_format(data)
            files_payload["file"] = (f"image.{ext}", BytesIO(data), mime)
        elif isinstance(image, pathlib.Path):
            files_payload["file"] = (image.name, image.open("rb"))
        else:
            raise TypeError(
                f"Unsupported image source type: {type(image).__name__}. "
                "Expected bytes, bytearray, or pathlib.Path."
            )

        return await self._request(
            "POST",
            "/image",
            schema=OCRResponse,
            data={
                "pipeline": "classic",
                "combine_line_breaks": "true" if combine_line_breaks else "false",
            },
            files=files_payload,
        )

    @staticmethod
    def _detect_format(data: bytes) -> tuple[str, str]:
        if len(data) < 4:
            raise ValueError("Image data too short to detect format.")

        # PNG
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            return "png", "image/png"
        # JPEG
        if data[:3] == b"\xff\xd8\xff":
            return "jpg", "image/jpeg"
        # TIFF (MM*\0 or II*\0)
        if data[:4] in (b"MM\x00*", b"II*\x00"):
            return "tiff", "image/tiff"
        # BMP
        if data[:2] == b"BM":
            return "bmp", "image/bmp"
        # WEBP (RIFF....WEBP)
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "webp", "image/webp"

        raise ValueError(f"Unsupported or unrecognized image format.")