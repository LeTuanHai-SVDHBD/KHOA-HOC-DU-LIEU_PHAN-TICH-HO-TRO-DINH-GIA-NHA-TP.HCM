from __future__ import annotations

import re

import pandas as pd

REQUIRED_COLUMNS = {
    "area_m2",
    "district",
    "house_type",
    "legal_status",
    "main_door_direction",
    "balcony_direction",
    "bedrooms",
    "frontage_m",
    "alley_width_m",
    "price_billion",
}

RAW_COLUMN_ALIASES = {
    "location": "location",
    "price": "price_raw",
    "type of house": "house_type",
    "land area": "land_area_raw",
    "bedrooms": "bedrooms_raw",
    "toilets": "toilets_raw",
    "main door direction": "main_door_direction",
    "balcony direction": "balcony_direction",
    "legal documents": "legal_status",
}


def _normalize_col_name(name: str) -> str:
    return str(name).strip().lower()


def _extract_first_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).lower().strip()
    text = text.replace(" ", "")
    match = re.search(r"\d+[\.,]?\d*", text)
    if not match:
        return None
    num = match.group(0).replace(",", ".")
    try:
        return float(num)
    except ValueError:
        return None


def _parse_price_to_billion(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).lower().strip()
    text = text.replace(" ", "")
    text = text.replace("vnđ", "đ")

    if "tỷ" in text:
        values = [float(v.replace(",", ".")) for v in re.findall(r"\d+[\.,]?\d*", text)]
        if not values:
            return None
        if len(values) == 1:
            return values[0]
        # Example: "6 tỷ 500 triệu" -> 6.5
        return values[0] + (values[1] / 1000.0)

    if "triệu" in text:
        value_million = _extract_first_number(text)
        return None if value_million is None else value_million / 1000.0

    # Example: "990.000.000 đ"
    digits = re.sub(r"[^0-9]", "", text)
    if digits:
        raw_vnd = float(digits)
        return raw_vnd / 1_000_000_000.0

    return None


def _parse_area_m2(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).lower().strip().replace(" ", "")
    text = text.replace(".", "") if "m²" in text and text.count(".") > 1 else text
    # Area values can be like "93,9 m² (5,0x18,8)". The first number is the area.
    numbers = re.findall(r"\d+[\.,]?\d*", text)
    if not numbers:
        return None
    candidate = numbers[0].replace(",", ".")
    try:
        return float(candidate)
    except ValueError:
        return None


def _parse_frontage_from_land_area(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).lower().replace(" ", "")
    dim_match = re.search(r"\((\d+[\.,]?\d*)x(\d+[\.,]?\d*)\)", text)
    if not dim_match:
        return None
    try:
        return float(dim_match.group(1).replace(",", "."))
    except ValueError:
        return None


def _parse_bedrooms(value: object) -> float | None:
    return _extract_first_number(value)


def _parse_toilets(value: object) -> float | None:
    return _extract_first_number(value)


def _infer_alley_width_from_house_type(value: object) -> float:
    text = str(value).lower()
    if "hẻm" in text or "hem" in text or "ngõ" in text or "ngo" in text:
        return 3.5
    if "mặt tiền" in text or "mat tien" in text:
        return 8.0
    if "villa" in text or "biệt thự" in text or "biet thu" in text:
        return 10.0
    return 5.0


def _extract_district_from_location(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().lower()
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[1]
    return text or "unknown"


def _convert_raw_schema_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    lowered = {_normalize_col_name(c): c for c in df.columns}
    has_required = REQUIRED_COLUMNS.issubset(set(df.columns))
    if has_required:
        return df.copy()

    has_raw_signature = all(key in lowered for key in RAW_COLUMN_ALIASES)
    if not has_raw_signature:
        return df.copy()

    mapped = pd.DataFrame()
    location_col = lowered["location"]
    price_col = lowered["price"]
    house_type_col = lowered["type of house"]
    land_area_col = lowered["land area"]
    bedrooms_col = lowered["bedrooms"]
    toilets_col = lowered["toilets"]
    main_door_col = lowered["main door direction"]
    balcony_col = lowered["balcony direction"]
    legal_col = lowered["legal documents"]

    mapped["district"] = df[location_col].map(_extract_district_from_location)
    mapped["house_type"] = df[house_type_col].astype(str)
    mapped["legal_status"] = df[legal_col].astype(str)
    mapped["area_m2"] = df[land_area_col].map(_parse_area_m2)
    mapped["price_billion"] = df[price_col].map(_parse_price_to_billion)
    mapped["bedrooms"] = df[bedrooms_col].map(_parse_bedrooms)
    mapped["toilets"] = df[toilets_col].map(_parse_toilets)
    mapped["frontage_m"] = df[land_area_col].map(_parse_frontage_from_land_area)
    mapped["alley_width_m"] = df[house_type_col].map(_infer_alley_width_from_house_type)
    mapped["main_door_direction"] = df[main_door_col].astype(str)
    mapped["balcony_direction"] = df[balcony_col].astype(str)

    return mapped


def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing columns: {missing_text}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    prepared = _convert_raw_schema_if_needed(df)
    validate_columns(prepared)
    cleaned = prepared.copy()

    if "toilets" not in cleaned.columns:
        cleaned["toilets"] = pd.NA

    numeric_cols = ["area_m2", "bedrooms", "toilets", "frontage_m", "alley_width_m", "price_billion"]
    for col in numeric_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    frontage_default = cleaned["frontage_m"].median() if not cleaned["frontage_m"].dropna().empty else 4.0
    bedrooms_default = cleaned["bedrooms"].median() if not cleaned["bedrooms"].dropna().empty else 2.0
    toilets_default = cleaned["toilets"].median() if not cleaned["toilets"].dropna().empty else 2.0

    cleaned["frontage_m"] = cleaned["frontage_m"].fillna(frontage_default)
    cleaned["bedrooms"] = cleaned["bedrooms"].fillna(bedrooms_default)
    cleaned["toilets"] = cleaned["toilets"].fillna(toilets_default)

    cleaned = cleaned.dropna(subset=numeric_cols)

    for col in ["district", "house_type", "legal_status", "main_door_direction", "balcony_direction"]:
        cleaned[col] = cleaned[col].astype(str).str.strip().str.lower()
        cleaned[col] = cleaned[col].replace({"": "unknown", "nan": "unknown"}).fillna("unknown")

    cleaned = cleaned[cleaned["area_m2"] > 0]
    cleaned = cleaned[cleaned["price_billion"] > 0]
    cleaned = cleaned.reset_index(drop=True)

    return cleaned
