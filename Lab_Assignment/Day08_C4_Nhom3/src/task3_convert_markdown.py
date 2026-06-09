"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import base64
import hashlib
import os
import subprocess
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from markitdown import MarkItDown
except ImportError:
    MarkItDown = None

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
PROJECT_DIR = Path(__file__).parent.parent
OCR_CACHE_DIR = OUTPUT_DIR / ".ocr_cache"


OCR_PROMPT = """Trich xuat toan bo van ban trong anh trang PDF nay.
Yeu cau:
- Giu nguyen thu tu doc tu tren xuong duoi, trai sang phai.
- Tra ve Markdown sach, khong them nhan xet.
- Neu co bang, chuyen thanh bang Markdown neu co the.
- Neu trang khong co chu, tra ve chuoi rong."""


def _write_markdown(output_path: Path, content: str):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content.strip() + "\n", encoding="utf-8")
    print(f"  Saved: {output_path}")


def _log(message: str):
    print(message, flush=True)


def _load_env_file():
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _get_pdf_page_count(filepath: Path) -> int:
    result = subprocess.run(
        ["pdfinfo", str(filepath)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise RuntimeError(f"Cannot read PDF page count: {filepath}")


def _pdf_cache_dir(filepath: Path) -> Path:
    relative = str(filepath.relative_to(PROJECT_DIR))
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:12]
    return OCR_CACHE_DIR / f"{filepath.stem}-{digest}"


def _strip_markdown_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _render_pdf_page(filepath: Path, page_number: int, output_dir: Path) -> Path:
    output_prefix = output_dir / f"page_{page_number:04d}"
    subprocess.run(
        [
            "pdftoppm",
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            "160",
            "-jpeg",
            "-singlefile",
            str(filepath),
            str(output_prefix),
        ],
        check=True,
        capture_output=True,
    )
    return output_prefix.with_suffix(".jpg")


def _chat_completions_url(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _ocr_image_with_vllm(image_path: Path) -> str:
    api_base = os.getenv("OCR_VLLM_API_BASE", "").strip()
    model = os.getenv("OCR_VLLM_MODEL", "").strip()
    api_key = os.getenv("OCR_VLLM_API_KEY", "").strip()
    if not api_base or not model:
        raise RuntimeError("Missing OCR_VLLM_API_BASE or OCR_VLLM_MODEL in .env")

    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 4096,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                    },
                ],
            }
        ],
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        _chat_completions_url(api_base),
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM OCR HTTP {exc.code}: {body[:500]}") from exc

    return _strip_markdown_fence(data["choices"][0]["message"]["content"])


def _convert_pdf_with_vllm_ocr(filepath: Path) -> str:
    page_count = _get_pdf_page_count(filepath)
    pages = {}
    cache_dir = _pdf_cache_dir(filepath)
    cache_dir.mkdir(parents=True, exist_ok=True)

    concurrency = max(1, int(os.getenv("OCR_VLLM_CONCURRENCY", "4")))

    def ocr_page(page_number: int, tmp_dir: Path) -> tuple[int, str, bool]:
        cache_path = cache_dir / f"page_{page_number:04d}.md"
        if cache_path.exists():
            return page_number, cache_path.read_text(encoding="utf-8"), True
        image_path = _render_pdf_page(filepath, page_number, tmp_dir)
        page_text = _ocr_image_with_vllm(image_path)
        cache_path.write_text(page_text, encoding="utf-8")
        return page_number, page_text, False

    with tempfile.TemporaryDirectory(prefix="task3_ocr_") as tmp:
        tmp_dir = Path(tmp)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(ocr_page, page_number, tmp_dir): page_number
                for page_number in range(1, page_count + 1)
            }
            for future in as_completed(futures):
                page_number, page_text, cache_hit = future.result()
                cache_hit = " (cached)" if cache_hit else ""
                _log(f"  OCR page {page_number}/{page_count}{cache_hit}")
                pages[page_number] = page_text

    rendered_pages = []
    for page_number in range(1, page_count + 1):
        page_text = pages.get(page_number, "")
        rendered_pages.append(f"## Page {page_number}\n\n{page_text}" if page_text else f"## Page {page_number}\n\n")
    return "\n\n".join(rendered_pages)


def _convert_with_markitdown(filepath: Path) -> str:
    if MarkItDown is None:
        raise RuntimeError("markitdown is not installed")
    md = MarkItDown()
    result = md.convert(str(filepath))
    return result.text_content


def _convert_pdf_with_pdftotext(filepath: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(filepath), "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _convert_doc_with_textutil(filepath: Path) -> str:
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(filepath)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _convert_file_to_text(filepath: Path) -> str:
    if filepath.suffix.lower() == ".pdf":
        try:
            return _convert_pdf_with_vllm_ocr(filepath)
        except Exception as exc:
            print(f"  vLLM OCR unavailable/failed ({exc}); using local fallback.")

    try:
        return _convert_with_markitdown(filepath)
    except Exception as exc:
        print(f"  MarkItDown unavailable/failed ({exc}); using local fallback.")
        if filepath.suffix.lower() == ".pdf":
            return _convert_pdf_with_pdftotext(filepath)
        return _convert_doc_with_textutil(filepath)


def _fallback_note(filepath: Path) -> str:
    return (
        "No substantial text could be extracted from this source file with the "
        "available local conversion tools. The original file is kept in "
        f"`{filepath.relative_to(LANDING_DIR)}` for downstream processing or "
        "manual OCR. This markdown record preserves the source path and allows "
        "the standardized corpus to keep a one-to-one mapping with landing files."
    )


def _metadata_header(title: str, source: str, crawled: str | None = None, published: str | None = None) -> str:
    lines = [
        f"# {title}",
        "",
        f"**Source:** {source}",
    ]
    if crawled:
        lines.append(f"**Crawled:** {crawled}")
    if published:
        lines.append(f"**Published:** {published}")
    lines.extend(["", "---", ""])
    return "\n".join(lines)


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in legal_dir.iterdir():
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            text = _convert_file_to_text(filepath)
            if len(text.strip()) < 200:
                text = _fallback_note(filepath) + "\n\n" + text

            header = _metadata_header(filepath.stem, str(filepath.relative_to(LANDING_DIR)))
            _write_markdown(output_dir / f"{filepath.stem}.md", header + text)


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in news_dir.iterdir():
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            title = data.get("title", filepath.stem)
            header = _metadata_header(
                title=title,
                source=data.get("url", "N/A"),
                crawled=data.get("date_crawled"),
                published=data.get("published_at"),
            )
            content = data.get("content_markdown") or ""
            _write_markdown(output_dir / f"{filepath.stem}.md", header + content)
        elif filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            text = _convert_file_to_text(filepath)
            if len(text.strip()) < 200:
                text = _fallback_note(filepath) + "\n\n" + text

            header = _metadata_header(filepath.stem, str(filepath.relative_to(LANDING_DIR)))
            _write_markdown(output_dir / f"{filepath.stem}.md", header + text)


def convert_all():
    """Convert toàn bộ files."""
    _load_env_file()

    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\nDone! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
