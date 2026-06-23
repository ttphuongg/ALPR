import re
import logging

logger = logging.getLogger(__name__)


def format_vietnamese_plate(raw_chars, is_two_lines: bool) -> str:
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

    # Dùng RegEx để khớp mẫu và format chuẩn.

    # Ô tô chuẩn 5 số: 2 số + 1 chữ + 5 số
    if re.match(r"^\d{2}[A-Z]\d{5}$", text):
        return f"{text[:3]}-{text[3:6]}.{text[6:]}"

    # Ô tô đặc biệt 5 số: 2 số + cụm 2 chữ đặc biệt + 5 số
    elif re.match(r"^\d{2}(LD|KT|DA|RM|MK|CD|HC)\d{5}$", text):
        return f"{text[:4]}-{text[4:7]}.{text[7:]}"

    # Xe máy chuẩn 5 số: 2 số + 1 chữ + 1 chữ/số + 5 số
    elif re.match(r"^\d{2}[A-Z][A-Z0-9]\d{5}$", text):
        return f"{text[:2]}-{text[2:4]} {text[4:7]}.{text[7:]}"

    # Ô tô cũ 4 số: 2 số + 1 chữ + 4 số
    elif re.match(r"^\d{2}[A-Z]\d{4}$", text):
        return f"{text[:3]}-{text[3:]}"

    # Xe máy cũ 4 số: 2 số + 1 chữ + 1 chữ/số + 4 số
    elif re.match(r"^\d{2}[A-Z][A-Z0-9]\d{4}$", text):
        return f"{text[:2]}-{text[2:4]} {text[4:]}"

    else:
        match = re.search(r"\d+$", text[2:]) 
        if match:
            suffix = match.group()
            prefix = text[: -len(suffix)]

            if len(suffix) == 5:
                # Đuôi đúng 5 ký tự có dấu chấm
                return f"{prefix}-{suffix[:3]}.{suffix[3:]}"
            else:
                # Đuôi lệch nối bằng gạch ngang, giữ nguyên để dễ debug
                return f"{prefix}-{suffix}"

        logger.debug("format_vietnamese_plate: không khớp mẫu nào, trả về chuỗi gốc: '%s'", text)
        return text 

def is_valid_format(plate: str) -> bool:
    if not plate:
        return False
    # Bỏ các ký tự định dạng để xét chuỗi thuần
    text = re.sub(r"[^A-Z0-9]", "", plate.upper())
    
    # Chuẩn VN ngắn nhất là 7 ký tự, dài nhất 9 ký tự
    if len(text) < 7:
        return False
        
    # Bắt buộc: 2 ký tự đầu là số, ký tự thứ 3 là chữ cái
    return bool(re.match(r"^\d{2}[A-Z]", text))
