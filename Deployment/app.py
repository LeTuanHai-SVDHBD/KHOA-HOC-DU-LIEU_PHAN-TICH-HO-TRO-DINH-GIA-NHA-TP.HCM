from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from src.data_prep import clean_dataset
from src.modeling import FEATURES, predict_cluster, predict_price, train_models

DEFAULT_DATA_PATH = "data/real_estate_listings.csv"
DEFAULT_REGION_XLSX_PATH = "data/DataClearn_with_clusters.xlsx"

st.set_page_config(
    page_title="Nền tảng định giá bất động sản AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top right, #f8f4e7 0%, #eef5f4 45%, #edf1fb 100%);
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12343b 0%, #1d4e57 100%);
    }
    div[data-testid="stSidebar"] * {
        color: #f6f7f8 !important;
    }
    .hero-card {
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        margin-bottom: 0.8rem;
        background: linear-gradient(135deg, rgba(18,52,59,0.95) 0%, rgba(35,111,121,0.9) 100%);
        box-shadow: 0 12px 30px rgba(18, 52, 59, 0.2);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .hero-title {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: 0.2px;
    }
    .hero-sub {
        color: #d9f4f5;
        font-size: 1rem;
        margin: 0;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #12343b;
        margin-bottom: 0.4rem;
    }
    .chip {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        background: #e8f4f2;
        color: #0f4b4b;
        font-weight: 600;
        font-size: 0.84rem;
    }
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.75);
        border: 1px solid rgba(18,52,59,0.08);
        padding: 0.6rem 0.8rem;
        border-radius: 14px;
    }
    div.stButton > button {
        background: linear-gradient(135deg, #0f766e 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.55rem 1rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">Nền tảng định giá bất động sản bằng AI</div>
        <p class="hero-sub">Nhập thông tin nhà đất, hệ thống sẽ phân cụm, dự đoán giá tham chiếu và đối chiếu theo vùng thị trường.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    if path.lower().endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)


def _extract_first_number(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).lower().strip().replace(" ", "")
    match = re.search(r"\d+[\.,]?\d*", text)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", "."))
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


def _parse_area_input(text: str) -> float | None:
    if text is None:
        return None
    raw = str(text).strip()
    if raw == "":
        return None
    raw = raw.replace(",", ".")
    match = re.search(r"\d+(?:\.\d+)?", raw)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _normalize_text_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def infer_region_from_district(district_text: str) -> str:
    text = (district_text or "").strip().lower()

    central_keywords = {
        "quận 1",
        "quận 3",
        "quận 5",
        "quận 10",
    }
    near_center_keywords = {
        "quận 4",
        "quận 6",
        "quận 7",
        "quận 8",
        "quận 11",
        "quận phú nhuận",
        "quận bình thạnh",
        "quận tân bình",
        "quận tân phú",
        "quận gò vấp",
    }

    if any(k in text for k in central_keywords):
        return "trung tâm"
    if any(k in text for k in near_center_keywords):
        return "ven trung tâm"

    return "ngoại ô"


def build_location_hierarchy(raw_df: pd.DataFrame) -> pd.DataFrame:
    location_col = None
    for col in raw_df.columns:
        if str(col).strip().lower() == "location":
            location_col = col
            break

    if location_col is None:
        return pd.DataFrame(columns=["cap_tinh_tp", "cap_quan_huyen", "cap_phuong_xa"])

    records = []
    for raw_loc in raw_df[location_col].dropna().astype(str):
        parts = [p.strip() for p in raw_loc.split(",") if p.strip()]
        if not parts:
            continue

        cap_phuong_xa = parts[0].lower()
        cap_quan_huyen = parts[1].lower() if len(parts) >= 2 else parts[0].lower()

        if len(parts) >= 3:
            cap_tinh_tp = parts[2].lower()
        else:
            cap_tinh_tp = "tp.hcm"

        records.append(
            {
                "cap_tinh_tp": cap_tinh_tp,
                "cap_quan_huyen": cap_quan_huyen,
                "cap_phuong_xa": cap_phuong_xa,
            }
        )

    if not records:
        return pd.DataFrame(columns=["cap_tinh_tp", "cap_quan_huyen", "cap_phuong_xa"])

    return pd.DataFrame(records).drop_duplicates().reset_index(drop=True)


@st.cache_data
def load_region_price_table(xlsx_path: str) -> pd.DataFrame:
    try:
        df = pd.read_excel(xlsx_path)
    except Exception:
        return pd.DataFrame()

    if not {"Vi_tri", "Phan_Loai_Gia"}.issubset(df.columns):
        return pd.DataFrame()

    region_df = df[["Vi_tri", "Phan_Loai_Gia"]].dropna().copy()
    region_df["Vi_tri"] = _normalize_text_series(region_df["Vi_tri"])
    region_df["Phan_Loai_Gia"] = region_df["Phan_Loai_Gia"].astype(str).str.strip()

    count_table = (
        region_df.groupby(["Vi_tri", "Phan_Loai_Gia"]).size().reset_index(name="so_luong")
    )
    total_by_region = count_table.groupby("Vi_tri")["so_luong"].transform("sum")
    count_table["ti_le_phan_tram"] = (count_table["so_luong"] / total_by_region * 100).round(1)

    return count_table.sort_values(["Vi_tri", "so_luong"], ascending=[True, False]).reset_index(drop=True)


@st.cache_data
def load_house_legal_options(xlsx_path: str) -> tuple[list[str], list[str]]:
    try:
        df = pd.read_excel(xlsx_path)
    except Exception:
        return [], []

    if not {"Loai_Nha", "Phap_Ly"}.issubset(df.columns):
        return [], []

    house_values = _normalize_text_series(df["Loai_Nha"]).dropna().tolist()
    legal_values = _normalize_text_series(df["Phap_Ly"]).dropna().tolist()

    house_values = sorted(set(house_values))
    legal_values = sorted(set(legal_values))

    return house_values, legal_values


@st.cache_resource
def load_artifacts(data_path: str):
    raw_df = load_data(data_path)
    cleaned = clean_dataset(raw_df)
    artifacts = train_models(cleaned)
    return raw_df, cleaned, artifacts


def get_sorted_unique_numbers(values: pd.Series, default_values: list[float]) -> list[float]:
    parsed = [_extract_first_number(v) for v in values.dropna().tolist()]
    parsed = [float(v) for v in parsed if v is not None]
    unique = sorted(set(parsed))
    return unique if unique else default_values


with st.sidebar:
    st.header("Thiết lập dữ liệu")
    uploaded = st.file_uploader("Tải lên file CSV/XLSX", type=["csv", "xlsx"])
    st.write("Các cột cần có (định dạng chuẩn hóa):")
    st.code(
        "area_m2,district,house_type,legal_status,bedrooms,frontage_m,alley_width_m,price_billion",
        language="text",
    )
    st.write("Hoặc có thể tải file gốc với cột như: Location, Price, Type of House, Land Area...")
    st.markdown("---")
    st.markdown("### Gợi ý dùng nhanh")
    st.markdown("- Điền đầy đủ cấp Quận/Huyện và Phường/Xã")
    st.markdown("- Diện tích nên trong khoảng dữ liệu huấn luyện")
    st.markdown("- Chọn số phòng ngủ và nhà vệ sinh lớn hơn 0")

if uploaded is not None:
    if uploaded.name.lower().endswith(".xlsx"):
        temp_df = pd.read_excel(uploaded)
    else:
        temp_df = pd.read_csv(uploaded)
    raw_data = temp_df.copy()
    data = clean_dataset(temp_df)
    artifacts = train_models(data)
else:
    raw_data, data, artifacts = load_artifacts(DEFAULT_DATA_PATH)

region_price_table = load_region_price_table(DEFAULT_REGION_XLSX_PATH)
xlsx_house_values, xlsx_legal_values = load_house_legal_options(DEFAULT_REGION_XLSX_PATH)

left, right = st.columns([1, 1])

with left:
    st.markdown('<div class="section-title">Biểu mẫu nhập liệu</div>', unsafe_allow_html=True)
    st.markdown('<span class="chip">Bước 1: Nhập thông tin</span>', unsafe_allow_html=True)

    hierarchy_df = build_location_hierarchy(raw_data)
    selected_tinh = ""
    selected_quan = ""
    selected_phuong = ""

    if hierarchy_df.empty:
        cap_tinh_options = ["tp.hcm"]
        selected_tinh = st.selectbox("Cấp cao nhất (Tỉnh/Thành phố)", cap_tinh_options)

        cap_quan_real = sorted(data["district"].unique().tolist())
        cap_quan_options = ["-- Chọn Quận/Huyện --", *cap_quan_real]
        selected_quan_ui = st.selectbox("Cấp Quận/Huyện", cap_quan_options, index=0)
        selected_quan = "" if selected_quan_ui.startswith("--") else selected_quan_ui

        if selected_quan:
            cap_phuong_options = ["(không có dữ liệu phường/xã trong file)"]
            selected_phuong = st.selectbox("Cấp Phường/Xã", cap_phuong_options, disabled=True)
        else:
            st.selectbox("Cấp Phường/Xã", ["-- Chọn Quận/Huyện trước --"], disabled=True)
    else:
        cap_tinh_options = sorted(hierarchy_df["cap_tinh_tp"].unique().tolist())
        selected_tinh = st.selectbox("Cấp cao nhất (Tỉnh/Thành phố)", cap_tinh_options)

        quan_df = hierarchy_df[hierarchy_df["cap_tinh_tp"] == selected_tinh]
        cap_quan_real = sorted(quan_df["cap_quan_huyen"].unique().tolist())
        cap_quan_options = ["-- Chọn Quận/Huyện --", *cap_quan_real]
        selected_quan_ui = st.selectbox("Cấp Quận/Huyện", cap_quan_options, index=0)
        selected_quan = "" if selected_quan_ui.startswith("--") else selected_quan_ui

        if selected_quan:
            phuong_df = quan_df[quan_df["cap_quan_huyen"] == selected_quan]
            cap_phuong_real = sorted(phuong_df["cap_phuong_xa"].unique().tolist())
            cap_phuong_options = ["-- Chọn Phường/Xã --", *cap_phuong_real]
            selected_phuong_ui = st.selectbox("Cấp Phường/Xã", cap_phuong_options, index=0)
            selected_phuong = "" if selected_phuong_ui.startswith("--") else selected_phuong_ui
        else:
            st.selectbox("Cấp Phường/Xã", ["-- Chọn Quận/Huyện trước --"], disabled=True)

    raw_cols = {str(c).strip().lower(): c for c in raw_data.columns}

    type_col = raw_cols.get("type of house")
    legal_col = raw_cols.get("legal documents")
    bedrooms_col = raw_cols.get("bedrooms")
    toilets_col = raw_cols.get("toilets")
    land_area_col = raw_cols.get("land area")
    main_door_col = raw_cols.get("main door direction")
    balcony_col = raw_cols.get("balcony direction")

    house_type_values = (
        xlsx_house_values
        if xlsx_house_values
        else (
            sorted(_normalize_text_series(raw_data[type_col]).dropna().unique().tolist())
            if type_col is not None
            else sorted(data["house_type"].unique().tolist())
        )
    )
    legal_values = (
        xlsx_legal_values
        if xlsx_legal_values
        else (
            sorted(_normalize_text_series(raw_data[legal_col]).dropna().unique().tolist())
            if legal_col is not None
            else sorted(data["legal_status"].unique().tolist())
        )
    )
    main_door_values = (
        sorted(
            _normalize_text_series(raw_data[main_door_col])
            .replace({"": pd.NA, "nan": pd.NA})
            .dropna()
            .unique()
            .tolist()
        )
        if main_door_col is not None
        else sorted(data["main_door_direction"].unique().tolist())
    )
    balcony_values = (
        sorted(
            _normalize_text_series(raw_data[balcony_col])
            .replace({"": pd.NA, "nan": pd.NA})
            .dropna()
            .unique()
            .tolist()
        )
        if balcony_col is not None
        else sorted(data["balcony_direction"].unique().tolist())
    )
    main_door_values = main_door_values or ["unknown"]
    balcony_values = balcony_values or ["unknown"]

    area_min = float(max(20.0, data["area_m2"].min()))
    area_max = float(min(600.0, data["area_m2"].max()))
    area_default = float(max(area_min, min(80.0, area_max)))

    bedroom_values = (
        get_sorted_unique_numbers(raw_data[bedrooms_col], [1, 2, 3, 4, 5])
        if bedrooms_col is not None
        else sorted(data["bedrooms"].astype(float).round().astype(int).unique().tolist())
    )
    bedroom_values = [int(v) for v in bedroom_values if v >= 0]
    bedroom_values = sorted(set(bedroom_values)) or [1, 2, 3, 4, 5]

    toilet_values = (
        get_sorted_unique_numbers(raw_data[toilets_col], [1, 2, 3, 4, 5])
        if toilets_col is not None
        else sorted(data["toilets"].astype(float).round().astype(int).unique().tolist())
    )
    toilet_values = [int(v) for v in toilet_values if v >= 0]
    toilet_values = sorted(set(toilet_values)) or [1, 2, 3, 4, 5]

    frontage_values = (
        sorted(
            {
                round(v, 1)
                for v in raw_data[land_area_col].map(_parse_frontage_from_land_area).dropna().tolist()
            }
        )
        if land_area_col is not None
        else sorted(data["frontage_m"].round(1).unique().tolist())
    )
    frontage_values = frontage_values or [3.0, 4.0, 5.0, 6.0, 8.0]

    alley_values = sorted({round(v, 1) for v in data["alley_width_m"].dropna().tolist()})
    alley_values = alley_values or [3.5, 5.0, 8.0, 10.0]

    area_input_text = st.text_input("Diện tích (m2)", value="", placeholder="Nhập diện tích, ví dụ: 81.5")
    area_m2 = _parse_area_input(area_input_text)
    area_valid = area_m2 is not None and area_min <= area_m2 <= area_max
    if area_m2 is not None and not area_valid:
        st.warning(f"Diện tích phải trong khoảng {area_min:.1f} - {area_max:.1f} m2.")

    if selected_quan and selected_phuong:
        st.caption(f"Địa chỉ đã chọn: {selected_phuong}, {selected_quan}, {selected_tinh}")
    elif selected_quan:
        st.caption(f"Địa chỉ đã chọn: (chưa chọn phường/xã), {selected_quan}, {selected_tinh}")
    else:
        st.caption("Địa chỉ đã chọn: chưa hoàn tất chọn cấp địa lý")

    if selected_quan:
        selected_region = infer_region_from_district(selected_quan)
        st.caption(f"Phân loại vùng theo vị trí: {selected_region.title()}")

    house_type = st.selectbox("Loại hình nhà", house_type_values, index=None, placeholder="-- Chọn loại hình nhà --")
    legal_status = st.selectbox("Tình trạng pháp lý", legal_values, index=None, placeholder="-- Chọn tình trạng pháp lý --")
    main_door_direction = st.selectbox(
        "Hướng cửa chính",
        main_door_values,
        index=None,
        placeholder="-- Chọn hướng cửa chính --",
    )
    balcony_direction = st.selectbox(
        "Hướng ban công",
        balcony_values,
        index=None,
        placeholder="-- Chọn hướng ban công --",
    )

    bedroom_max = int(max(bedroom_values)) if bedroom_values else 10
    toilet_max = int(max(toilet_values)) if toilet_values else 10
    bedrooms = st.slider("Số phòng ngủ", min_value=0, max_value=max(1, bedroom_max), value=0, step=1)
    toilets = st.slider("Số nhà vệ sinh", min_value=0, max_value=max(1, toilet_max), value=0, step=1)
    frontage_m = st.selectbox(
        "Chiều ngang mặt tiền (m)",
        frontage_values,
        index=None,
        placeholder="-- Chọn chiều ngang mặt tiền --",
    )
    alley_width_m = st.selectbox(
        "Độ rộng hẻm/đường (m)",
        alley_values,
        index=None,
        placeholder="-- Chọn độ rộng hẻm/đường --",
    )

    row = pd.DataFrame(
        [
            {
                "area_m2": area_m2 if area_m2 is not None else 0.0,
                "district": selected_quan,
                "house_type": house_type,
                "legal_status": legal_status,
                "main_door_direction": main_door_direction or "unknown",
                "balcony_direction": balcony_direction or "unknown",
                "bedrooms": bedrooms,
                "toilets": toilets,
                "frontage_m": frontage_m,
                "alley_width_m": alley_width_m,
            }
        ]
    )

    can_predict = (
        bool(selected_quan)
        and (hierarchy_df.empty or bool(selected_phuong))
        and bool(house_type)
        and bool(legal_status)
        and area_valid
        and bedrooms > 0
        and toilets > 0
        and frontage_m is not None
        and alley_width_m is not None
    )
    if not can_predict:
        st.warning(
            "Vui lòng nhập diện tích, chọn Loại hình nhà/Pháp lý, chọn đúng cấp địa lý, và chỉnh số phòng ngủ/số nhà vệ sinh > 0 trước khi dự đoán."
        )

    run_btn = st.button("Dự đoán bằng AI", type="primary", disabled=not can_predict, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Kết quả AI</div>', unsafe_allow_html=True)
    st.markdown('<span class="chip">Bước 2: Đọc kết quả</span>', unsafe_allow_html=True)
    if run_btn:
        cluster_id = predict_cluster(artifacts, row[FEATURES])
        predicted_price = predict_price(artifacts, row[FEATURES])
        region_est_avg = None
        selected_region = infer_region_from_district(selected_quan)

        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.metric("Cụm dự đoán", f"Cụm {cluster_id}")
        with mcol2:
            st.metric("Giá tham chiếu", f"{predicted_price:.2f} tỷ VND")
        with mcol3:
            st.metric("Phân vùng", selected_region.title())

        cluster_df = data.copy()
        cluster_df["cluster"] = artifacts.clustering_pipeline.predict(cluster_df[FEATURES])
        peer = cluster_df[cluster_df["cluster"] == cluster_id]

        st.write("Giá trung bình trong cụm:", f"{peer['price_billion'].mean():.2f} tỷ VND")
        st.write("Số mẫu trong cụm:", int(peer.shape[0]))

        if not region_price_table.empty:
            rg = region_price_table[region_price_table["Vi_tri"] == selected_region]

            if not rg.empty:
                price_score = {"Dưới 5 tỷ": 2.5, "5-10 tỷ": 7.5, "Trên 10 tỷ": 12.5}
                top_band = rg.sort_values("so_luong", ascending=False).iloc[0]["Phan_Loai_Gia"]
                total = rg["so_luong"].sum()
                est_avg = 0.0
                for _, rr in rg.iterrows():
                    est_avg += price_score.get(rr["Phan_Loai_Gia"], 0.0) * rr["so_luong"]
                est_avg = est_avg / total if total else 0.0
                region_est_avg = est_avg

                st.markdown("### Đối chiếu theo vùng (từ file XLSX)")
                st.write(f"Vị trí đã chọn thuộc vùng: **{selected_region.title()}**")
                st.write(f"Mức giá phổ biến của vùng: **{top_band}**")
                st.write(f"Giá ước lượng trung bình của vùng: **{est_avg:.2f} tỷ VND**")
                st.write(f"Số mẫu vùng trong file XLSX: **{int(total)}**")

        chart_rows = [
            {"Chi_so": "Giá dự đoán", "Ty_VND": float(predicted_price)},
            {"Chi_so": "TB trong cụm", "Ty_VND": float(peer["price_billion"].mean())},
        ]
        if region_est_avg is not None:
            chart_rows.append({"Chi_so": "TB theo vùng", "Ty_VND": float(region_est_avg)})

        chart_df = pd.DataFrame(chart_rows)
        st.markdown("### Biểu đồ tròn so sánh nhanh")
        st.vega_lite_chart(
            chart_df,
            {
                "mark": {"type": "arc", "outerRadius": 125, "innerRadius": 55},
                "encoding": {
                    "theta": {"field": "Ty_VND", "type": "quantitative"},
                    "color": {
                        "field": "Chi_so",
                        "type": "nominal",
                        "scale": {
                            "range": ["#1f8ef1", "#20bf6b", "#f39c12"]
                        },
                        "legend": {"title": "Chỉ số"},
                    },
                    "tooltip": [
                        {"field": "Chi_so", "type": "nominal", "title": "Chỉ số"},
                        {
                            "field": "Ty_VND",
                            "type": "quantitative",
                            "title": "Tỷ VND",
                            "format": ".2f",
                        },
                    ],
                },
                "view": {"stroke": None},
            },
            width="stretch",
        )

        st.markdown("### Mẫu tham khảo gần nhất từ dữ liệu của bạn")
        cols_to_show = [
            "district",
            "house_type",
            "legal_status",
            "area_m2",
            "bedrooms",
            "toilets",
            "price_billion",
        ]
        view_peer = peer[cols_to_show].head(8).rename(
            columns={
                "district": "khu_vuc",
                "house_type": "loai_nha",
                "legal_status": "phap_ly",
                "area_m2": "dien_tich_m2",
                "bedrooms": "so_phong_ngu",
                "toilets": "so_nha_ve_sinh",
                "price_billion": "gia_ty_vnd",
            }
        )
        st.dataframe(view_peer, width="stretch")
    else:
        st.info("Nhập thông tin bên trái và bấm 'Dự đoán bằng AI' để xem kết quả trực quan.")

st.divider()
tab_data, tab_clean, tab_method = st.tabs([
    "Tổng quan dữ liệu",
    "Dữ liệu sau làm sạch",
    "Phương pháp AI",
])

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng huấn luyện", f"{len(data):,}")
    c2.metric("Số quận/huyện", f"{data['district'].nunique():,}")
    c3.metric("Loại hình nhà", f"{data['house_type'].nunique():,}")
    st.dataframe(data.head(20), width="stretch")

with tab_clean:
    st.write(f"Số dòng dữ liệu gốc: {len(raw_data)}")
    st.write(f"Số dòng sau làm sạch: {len(data)}")
    cleaned_preview = data.head(30).copy()
    st.dataframe(cleaned_preview, width="stretch")
    csv_bytes = data.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        label="Tải về dữ liệu đã làm sạch (CSV)",
        data=csv_bytes,
        file_name="du_lieu_da_lam_sach.csv",
        mime="text/csv",
        use_container_width=True,
    )

with tab_method:
    st.markdown(
        "- Theo nội dung file Word: quy trình chuẩn là MCA + K-Means để phân khúc dữ liệu bất động sản.\n"
        "- Trong web app hiện tại: dữ liệu được làm sạch và mã hóa đặc trưng, sau đó dùng K-Means để phân cụm và RandomForestRegressor để dự đoán giá tham chiếu.\n"
        "- Phần kết quả vùng Trung tâm/Ven trung tâm/Ngoại ô được tổng hợp từ file XLSX để đối chiếu định giá."
    )
