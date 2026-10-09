from __future__ import annotations

import base64
import binascii
import hashlib
import io
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlparse

import fitz
import numpy as np
import requests
from PIL import Image, ImageSequence, UnidentifiedImageError


@dataclass(frozen=True)
class LoadedDocument:
    content: bytes
    filename: str
    source: str
    sha256: str


def _positive_int(value: Any, name: str, default: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not 1 <= parsed <= maximum:
        raise ValueError(f"{name} must be between 1 and {maximum}.")
    return parsed


def _filename_from_url(url: str) -> str:
    path = PurePosixPath(unquote(urlparse(url).path))
    return path.name or "document"


def _download(url: str, maximum_bytes: int) -> bytes:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) URL.")

    headers = {"User-Agent": "docai-runpod-paddle-ocr/1.0"}
    with requests.get(url, headers=headers, stream=True, timeout=(10, 180)) as response:
        response.raise_for_status()
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > maximum_bytes:
            raise ValueError("The remote document exceeds MAX_DOCUMENT_MB.")

        chunks: list[bytes] = []
        size = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            size += len(chunk)
            if size > maximum_bytes:
                raise ValueError("The remote document exceeds MAX_DOCUMENT_MB.")
            chunks.append(chunk)
    return b"".join(chunks)


def _decode_base64(value: str, maximum_bytes: int) -> bytes:
    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        content = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("base64 is not valid Base64 data.") from exc
    if len(content) > maximum_bytes:
        raise ValueError("The decoded document exceeds MAX_DOCUMENT_MB.")
    return content


def load_document(job_input: dict[str, Any], maximum_bytes: int) -> LoadedDocument:
    url = job_input.get("url")
    encoded = job_input.get("base64")
    if bool(url) == bool(encoded):
        raise ValueError("Provide exactly one of input.url or input.base64.")

    if url:
        if not isinstance(url, str):
            raise ValueError("url must be a string.")
        content = _download(url, maximum_bytes)
        default_name = _filename_from_url(url)
        source = "url"
    else:
        if not isinstance(encoded, str):
            raise ValueError("base64 must be a string.")
        content = _decode_base64(encoded, maximum_bytes)
        default_name = "document"
        source = "base64"

    if not content:
        raise ValueError("The document is empty.")
    filename = str(job_input.get("filename") or default_name)
    return LoadedDocument(
        content=content,
        filename=filename,
        source=source,
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _render_pdf(content: bytes, dpi: int, maximum_pages: int) -> list[np.ndarray]:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise ValueError("The PDF could not be opened.") from exc

    try:
        if document.needs_pass:
            raise ValueError("Password-protected PDFs are not supported.")
        if document.page_count < 1:
            raise ValueError("The PDF has no pages.")
        if document.page_count > maximum_pages:
            raise ValueError(f"The PDF exceeds the {maximum_pages}-page limit.")

        scale = dpi / 72.0
        matrix = fitz.Matrix(scale, scale)
        pages: list[np.ndarray] = []
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            pixels = np.frombuffer(pixmap.samples, dtype=np.uint8)
            pages.append(pixels.reshape(pixmap.height, pixmap.width, 3).copy())
        return pages
    finally:
        document.close()


def _render_image(content: bytes, maximum_pages: int) -> list[np.ndarray]:
    try:
        source = Image.open(io.BytesIO(content))
    except UnidentifiedImageError as exc:
        raise ValueError("The input is neither a valid PDF nor a supported image.") from exc

    pages: list[np.ndarray] = []
    try:
        for index, frame in enumerate(ImageSequence.Iterator(source)):
            if index >= maximum_pages:
                raise ValueError(f"The image exceeds the {maximum_pages}-frame limit.")
            pages.append(np.asarray(frame.convert("RGB")).copy())
    finally:
        source.close()
    return pages


def render_pages(
    content: bytes,
    *,
    dpi: int,
    maximum_pages: int,
) -> list[np.ndarray]:
    if content.lstrip().startswith(b"%PDF-"):
        return _render_pdf(content, dpi, maximum_pages)
    return _render_image(content, maximum_pages)

