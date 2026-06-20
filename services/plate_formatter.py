import re
import logging

logger = logging.getLogger(__name__)


def format_vietnamese_plate(raw_chars, is_two_lines: bool) -> str:
    """Chuẩn hoá chuỗi OCR thành định dạng biển số Việt Nam.

    Args:
        raw_chars: Nếu ``is_two_lines=True`` thì là list[str] gồm 2 dòng
                   (dòng trên, dòng dưới). Nếu False thì là str hoặc list[str]
                   một phần tử.
        is_two_lines: True khi biển số 2 dòng (xe máy).

    Returns:
        Chuỗi biển số đã được format chuẩn.
    """
    # Bước 1: Lọc sạch rác, gộp thành 1 chuỗi duy nhất,
    #         chỉ lấy chữ và số, in hoa toàn bộ.
    if is_two_lines and isinstance(raw_chars, list) and len(raw_chars) >= 2:
        text = (
            "".join(c for c in raw_chars[0] if c.isalnum())
            + "".join(c for c in raw_chars[1] if c.isalnum())
        )
    else:
        if isinstance(raw_chars, list):
            raw_chars = "".join(raw_chars)
        text = "".join(c for c in raw_chars if c.isalnum())

    text = text.upper()

    if not text:
        return ""

    # Bước 2: Dùng RegEx để khớp mẫu và format chuẩn.

    # 1. Ô tô chuẩn 5 số (VD: 51G97162) → 2 số + 1 chữ + 5 số
    if re.match(r"^\d{2}[A-Z]\d{5}$", text):
        return f"{text[:3]}-{text[3:6]}.{text[6:]}"

    # 2. Ô tô đặc biệt 5 số (VD: 51LD12345) → 2 số + cụm 2 chữ đặc biệt + 5 số
    elif re.match(r"^\d{2}(LD|KT|DA|RM|MK|CD|HC)\d{5}$", text):
        return f"{text[:4]}-{text[4:7]}.{text[7:]}"

    # 3. Xe máy chuẩn 5 số (VD: 29H112345, 29AA12345) → 2 số + 1 chữ + 1 chữ/số + 5 số
    elif re.match(r"^\d{2}[A-Z][A-Z0-9]\d{5}$", text):
        return f"{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}"

    # 4. Ô tô cũ 4 số (VD: 30K1234) → 2 số + 1 chữ + 4 số
    elif re.match(r"^\d{2}[A-Z]\d{4}$", text):
        return f"{text[:3]}-{text[3:]}"

    # 5. Xe máy cũ 4 số (VD: 29H11234) → 2 số + 1 chữ + 1 chữ/số + 4 số
    elif re.match(r"^\d{2}[A-Z][A-Z0-9]\d{4}$", text):
        return f"{text[:2]}-{text[2:4]} {text[4:]}"

    # Bước 3: FALLBACK — OCR đọc sai/thiếu ký tự → bóc tách an toàn
    else:
        match = re.search(r"\d+$", text[2:])  # Bỏ qua 2 số đầu (mã vùng)
        if match:
            suffix = match.group()
            prefix = text[: -len(suffix)]

            if len(suffix) == 5:
                # Đuôi đúng 5 ký tự → có dấu chấm
                return f"{prefix}-{suffix[:3]}.{suffix[3:]}"
            else:
                # Đuôi lệch → nối bằng gạch ngang, giữ nguyên để dễ debug
                return f"{prefix}-{suffix}"

        logger.debug("format_vietnamese_plate: không khớp mẫu nào, trả về chuỗi gốc: '%s'", text)
        return text  # Bất khả kháng thì trả về chuỗi gốc

def is_valid_format(plate: str) -> bool:
    """Kiểm tra xem biển số có đúng chuẩn cơ bản không (2 số đầu, chữ cái thứ 3, đủ độ dài)."""
    if not plate:
        return False
    # Bỏ các ký tự định dạng để xét chuỗi thuần
    text = re.sub(r"[^A-Z0-9]", "", plate.upper())
    
    # Chuẩn VN ngắn nhất là 7 ký tự (VD: 30K1234), dài nhất 9 ký tự (VD: 51LD12345)
    if len(text) < 7:
        return False
        
    # Bắt buộc: 2 ký tự đầu là số, ký tự thứ 3 là chữ cái
    return bool(re.match(r"^\d{2}[A-Z]", text))
