import os
import re
import math
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use('Agg') # 确保无 GUI 环境下正常运行
import matplotlib.pyplot as plt
import rasterio

OUTPUT_DIR = "indices_output"
MAX_RENDER_PIXELS = 4_000_000  # 热力图只按显示需要读取，整景全尺寸读入太慢

PROCESSED_PATH = "processed.tif"
MASK_PATH = "mask.npy"
MASK_NDWI_PATH = "mask_ndwi.npy"
CALIBRATED_PATH = "_calibrated.tif"

# 每次请求独占一个运行目录，避免并发/多实例互相覆盖中间文件
RUN_ROOT = os.path.join(OUTPUT_DIR, "runs")
KEEP_RUNS = 10


def new_run_dir():
    """为一次请求创建独立的输出目录。"""
    os.makedirs(RUN_ROOT, exist_ok=True)
    return tempfile.mkdtemp(prefix="run_", dir=RUN_ROOT)


def prune_run_dirs(keep=KEEP_RUNS, min_age_seconds=300):
    """只保留最近几次运行，删除更早的；近期目录不动，避免误删正在跑的请求。"""
    try:
        entries = [e for e in os.scandir(RUN_ROOT) if e.is_dir()]
    except OSError:
        return
    entries.sort(key=lambda e: e.stat().st_mtime, reverse=True)
    now = time.time()
    for old in entries[keep:]:
        try:
            if now - old.stat().st_mtime < min_age_seconds:
                continue
            shutil.rmtree(old.path, ignore_errors=True)
        except OSError:
            pass


def _is_water_index(index_type):
    name = str(index_type or "").upper()
    return name in {"NDWI", "NDTI", "MNDWI", "CHL_PROXY"} or name.endswith("WI")

# 指数 -> (标准色带, 是否以 0 为中心对称拉伸)
# NDVI/NDBI/RVI/SAVI 用红-黄-绿；NDWI 用红-蓝（水体为正、偏蓝）
INDEX_STYLE = {
    "NDVI": ("RdYlGn", False),
    "NDWI": ("RdBu", True),
    "NDBI": ("RdYlGn", False),
    "RVI": ("RdYlGn", False),
    "SAVI": ("RdYlGn", False),
    "NBR": ("RdYlGn", False),
    "BSI": ("YlOrBr", False),
    "NDTI": ("RdBu", True),
    "NBR2": ("RdYlGn", False),
    "LAI": ("YlGn", False),
    "chl_proxy": ("YlGn", False),
}

# 非标准指数在图标题上打标（matplotlib 未配置中文字体，这里用英文）
INDEX_TITLE_SUFFIX = {
    "chl_proxy": "  [proxy, not a standard index]",
    "LAI": "  [empirical estimate, not standard]",
}

ALL_INDICES = tuple(INDEX_STYLE.keys())

# 从文件名里识别波段号：_B4 / B04 / band4 / 4.tif
BAND_PATTERNS = [
    re.compile(r"_B(\d{1,2})A?(?=[._]|$)", re.I),
    re.compile(r"\bB(\d{1,2})A?\b", re.I),
    re.compile(r"band[ _-]?(\d{1,2})", re.I),
    re.compile(r"(\d{1,2})\.tiff?$", re.I),
]


def _band_number(path, fallback):
    name = os.path.basename(path)
    for pattern in BAND_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return fallback


def _build_sources(paths):
    """返回 {波段号: (文件路径, 文件内波段索引)}。单文件多波段则按 1..N 编号。"""
    if len(paths) == 1:
        with rasterio.open(paths[0]) as src:
            if src.count > 1:
                return {i: (paths[0], i) for i in range(1, src.count + 1)}
    sources = {}
    for pos, path in enumerate(paths, 1):
        sources[_band_number(path, pos)] = (path, 1)
    return sources


def _read_band(sources, band, out_shape=None):
    if band is None:
        return None
    source = sources.get(band)
    if source is None:
        return None
    with rasterio.open(source[0]) as src:
        return src.read(source[1], out_shape=out_shape).astype('float32')


def _stretch(arr, symmetric):
    """按 2%~98% 分位数拉伸，保证指数高低区域在色彩上有明显区分。"""
    sample = arr[::4, ::4]
    valid = sample[np.isfinite(sample)]
    if valid.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(valid, [2, 98])
    if symmetric:
        half = float(max(abs(lo), abs(hi)))
        if half < 1e-6:
            half = 1.0
        return -half, half
    if hi - lo < 1e-6:
        return float(lo) - 0.05, float(hi) + 0.05
    return float(lo), float(hi)


def _save_heatmap(arr, name, out_path, subtitle):
    cmap = plt.get_cmap(INDEX_STYLE[name][0]).copy()
    cmap.set_bad(color="lightgray")

    arr = np.where(np.isfinite(arr), arr, np.nan)
    vmin, vmax = _stretch(arr, INDEX_STYLE[name][1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    ax.set_title(f"{name}  {subtitle}{INDEX_TITLE_SUFFIX.get(name, '')}", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#dddddd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    plt.close(fig)


METADATA_BAND_MAP = {
    "landsat8": {"blue_band": 2, "green_band": 3, "red_band": 4, "nir_band": 5,
                 "swir_band": 6, "swir2_band": 7},
    "landsat9": {"blue_band": 2, "green_band": 3, "red_band": 4, "nir_band": 5,
                 "swir_band": 6, "swir2_band": 7},
    "sentinel2": {"blue_band": 2, "green_band": 3, "red_band": 4, "nir_band": 8,
                  "swir_band": 11, "swir2_band": 12},
}


def _role_from_description(text):
    value = str(text or "").lower()
    if re.search(r"swir[ _-]?2|short[ _-]?wave[ _-]?infrared[ _-]?2|短波红外[ _-]?2", value):
        return "swir2_band"
    if re.search(r"swir|short[ _-]?wave|短波红外", value):
        return "swir_band"
    if re.search(r"nir|near[ _-]?infrared|近红外", value):
        return "nir_band"
    if re.search(r"\bred\b|红光|红波段", value):
        return "red_band"
    if re.search(r"green|绿光|绿波段", value):
        return "green_band"
    if re.search(r"\bblue\b|蓝光|蓝波段", value):
        return "blue_band"
    return None


def _mapping_from_descriptions(descriptions):
    mapping = {}
    for band_index, description in enumerate(descriptions or [], 1):
        role = _role_from_description(description)
        if role and role not in mapping:
            mapping[role] = band_index
    if "red_band" in mapping and "nir_band" in mapping:
        mapping.setdefault("blue_band", None)
        mapping.setdefault("green_band", None)
        mapping.setdefault("swir_band", None)
        mapping.setdefault("swir2_band", None)
        return mapping
    return None


def _valid_metadata_mapping(mapping, band_count):
    if not mapping:
        return None
    red = mapping.get("red_band")
    nir = mapping.get("nir_band")
    if not red or not nir or red > band_count or nir > band_count:
        return None
    result = dict(mapping)
    for key in ("blue_band", "green_band", "swir_band", "swir2_band"):
        if result.get(key) and result[key] > band_count:
            result[key] = None
    return result


def _find_mtl(directory):
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for name in names:
        if name.upper().endswith("_MTL.TXT") or name.upper() == "MTL.TXT":
            return os.path.join(directory, name)
    return None


def _sensor_from_text(text):
    value = text.upper()
    if re.search(r"LANDSAT[_ -]?9|LC0?9", value):
        return "landsat9"
    if re.search(r"LANDSAT[_ -]?8|LC0?8", value):
        return "landsat8"
    return None


def _read_landsat_mtl(directory):
    mtl = _find_mtl(directory)
    if not mtl:
        return None
    values = {}
    try:
        with open(mtl, "r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip().upper()] = value.strip().strip('"')
    except OSError:
        return None

    sensor = _sensor_from_text(values.get("SPACECRAFT_ID", "")) or _sensor_from_text(os.path.basename(mtl))
    if not sensor:
        return None
    mapping = _valid_metadata_mapping(METADATA_BAND_MAP[sensor], 99)
    return sensor, mapping, "landsat_mtl"


def _read_sentinel_xml(directory):
    try:
        names = os.listdir(directory)
    except OSError:
        return None
    for name in names:
        if not name.lower().endswith(".xml"):
            continue
        path = os.path.join(directory, name)
        try:
            root = ET.parse(path).getroot()
            tag = root.tag.lower()
            header = name.upper()
        except (ET.ParseError, OSError):
            continue
        if ("msil" in header or "sentinel-2" in header
                or "level-2a" in tag or "level-1c" in tag):
            mapping = _valid_metadata_mapping(METADATA_BAND_MAP["sentinel2"], 99)
            return "sentinel2", mapping, "sentinel2_xml"
    return None


@lru_cache(maxsize=128)
def _parse_metadata_cached(image_path, size, mtime_ns):
    result = {"sensor": None, "source": None, "mapping": None, "band_count": None}
    try:
        with rasterio.open(image_path) as src:
            band_count = src.count
            descriptions = src.descriptions
            tags = src.tags()
    except Exception:
        return result

    result["band_count"] = band_count

    description_mapping = _mapping_from_descriptions(descriptions)
    if description_mapping:
        result.update({"sensor": "generic", "source": "band_descriptions",
                       "mapping": description_mapping})
        return result

    tag_text = " ".join(str(value) for value in tags.values())
    tag_mapping = _mapping_from_descriptions([tag_text]) if tag_text.strip() else None
    if tag_mapping:
        result.update({"sensor": "generic", "source": "band_descriptions",
                       "mapping": tag_mapping})
        return result

    directory = os.path.dirname(image_path)

    landsat = _read_landsat_mtl(directory)
    if landsat:
        sensor, mapping, source = landsat
        mapping = _valid_metadata_mapping(mapping, band_count)
        if mapping:
            result.update({"sensor": sensor, "source": source, "mapping": mapping})
            return result

    sentinel = _read_sentinel_xml(directory)
    if sentinel:
        sensor, mapping, source = sentinel
        mapping = _valid_metadata_mapping(mapping, band_count)
        if mapping:
            result.update({"sensor": sensor, "source": source, "mapping": mapping})
            return result

    return result


def parse_metadata(image_path):
    """本地元数据优先解析：波段描述、Landsat MTL、Sentinel-2 XML。"""
    path = os.path.abspath(image_path)
    stat = os.stat(path)
    cached = _parse_metadata_cached(path, stat.st_size, stat.st_mtime_ns)
    mapping = dict(cached["mapping"]) if cached.get("mapping") else None
    return {**cached, "mapping": mapping}


@lru_cache(maxsize=128)
def _check_data_level_cached(image_path, size, mtime_ns):
    with rasterio.open(image_path) as src:
        first = src.read(1)
        nodata = src.nodata
    valid = first[np.isfinite(first)]
    if nodata is not None:
        valid = valid[valid != nodata]
    if valid.size == 0:
        raise ValueError("影像没有有效像元，无法判断数据级别")

    max_value = float(valid.max())
    if max_value > 255:
        return "L0"
    if max_value > 1.0:
        return "L1C"
    return "L2"


def check_data_level(image_path):
    """按首波段最大值判断数据级别；按文件大小和修改时间缓存结果。"""
    path = os.path.abspath(image_path)
    stat = os.stat(path)
    return _check_data_level_cached(path, stat.st_size, stat.st_mtime_ns)


def radiometric_calibration(image_path, output_path=CALIBRATED_PATH):
    """将整数 DN 按位深量级归一化到 0~1，作为无定标系数时的近似辐射定标。"""
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        profile.update(dtype='float32', nodata=np.nan, compress='lzw')
        dtype = np.dtype(src.dtypes[0])
        scale = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0

        with rasterio.open(output_path, 'w', **profile) as dst:
            for index in range(1, src.count + 1):
                raw = src.read(index)
                arr = raw.astype('float32') / scale
                if src.nodata is not None:
                    arr[raw == src.nodata] = np.nan
                dst.write(arr, index)
    return output_path


def atmospheric_correction_dos(image_path, output_path=PROCESSED_PATH):
    """轻量 DOS 大气校正：逐波段减去暗物体值，抑制路径辐射影响。"""
    with rasterio.open(image_path) as src:
        profile = src.profile.copy()
        profile.update(dtype='float32', nodata=np.nan, compress='lzw')

        with rasterio.open(output_path, 'w', **profile) as dst:
            for index in range(1, src.count + 1):
                raw = src.read(index)
                arr = raw.astype('float32')
                valid = arr[np.isfinite(arr) & (arr > 0)]
                dark = float(np.percentile(valid, 1)) if valid.size else 0.0
                corrected = np.where(np.isfinite(arr), arr - dark, np.nan)
                corrected = np.where(corrected > 0, corrected, 0.0)
                if src.nodata is not None:
                    corrected[raw == src.nodata] = np.nan
                dst.write(corrected.astype('float32'), index)
    return output_path


def _default_band_roles(count):
    if count >= 6:
        return {"green_band": 3, "red_band": 4, "nir_band": 5}
    if count == 4:
        return {"green_band": 2, "red_band": 3, "nir_band": 4}
    if count == 3:
        return {"green_band": 2, "red_band": 1, "nir_band": None}
    return {"green_band": 1, "red_band": 1, "nir_band": None}


def _norm_params(arr):
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(valid, [2, 98])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return float(lo), float(hi)


def generate_mask(image_path, index_type="NDVI", output_path=MASK_PATH, band_indices=None):
    """按指数类型生成掩膜：陆地指数剔除水体、云和阴影；NDWI 只剔除云和阴影。"""
    with rasterio.open(image_path) as src:
        height, width = src.height, src.width
        count = src.count
        roles = dict(band_indices or _default_band_roles(count))
        roles = {k: v for k, v in roles.items() if v and 1 <= v <= count}
        is_water_index = _is_water_index(index_type)

        used = {1, 2, 3}
        used.update(roles.values())
        used = {b for b in used if 1 <= b <= count}

        params = {}
        for band in used:
            sample = src.read(band, out_shape=(min(height, 512), min(width, 512))).astype('float32')
            if src.nodata is not None:
                sample[sample == src.nodata] = np.nan
            params[band] = _norm_params(sample)

        mask = np.ones((height, width), dtype=bool)
        water_count = cloud_count = shadow_count = 0
        nodata = src.nodata

        for _, window in src.block_windows(1):
            arrays = {}
            for band in used:
                raw = src.read(band, window=window).astype('float32')
                if nodata is not None:
                    raw[raw == nodata] = np.nan
                lo, hi = params[band]
                arrays[band] = (np.clip((raw - lo) / (hi - lo), 0, 1), raw)

            norm = {b: v[0] for b, v in arrays.items()}
            raw = {b: v[1] for b, v in arrays.items()}

            brightness = np.nanmean(np.stack([norm[b] for b in sorted(used)[:3]]), axis=0)
            valid = np.isfinite(brightness)
            for band in used:
                valid &= np.isfinite(raw[band])
                valid &= raw[band] > 0

            g = roles.get("green_band")
            r = roles.get("red_band")
            n = roles.get("nir_band")
            ndwi = None
            ndvi = None
            if g and n:
                den = raw[g] + raw[n]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndwi = np.where(den > 0, (raw[g] - raw[n]) / den, 0.0)
            if r and n:
                den = raw[r] + raw[n]
                with np.errstate(divide='ignore', invalid='ignore'):
                    ndvi = np.where(den > 0, (raw[n] - raw[r]) / den, 0.0)

            water = ndwi > 0.1 if ndwi is not None else np.zeros_like(valid)
            if ndvi is not None:
                cloud = (brightness > 0.75) & (ndvi < 0.2)
            else:
                cloud = brightness > 0.85
            shadow = (brightness < 0.10) & ~water

            if is_water_index:
                block = valid & ~cloud & ~shadow
            else:
                block = valid & ~water & ~cloud & ~shadow
            mask[window.row_off:window.row_off + window.height,
                 window.col_off:window.col_off + window.width] = block

            water_count += int(np.count_nonzero(water & valid))
            cloud_count += int(np.count_nonzero(cloud & valid))
            shadow_count += int(np.count_nonzero(shadow & valid))

    np.save(output_path, mask)
    total = int(mask.size)
    stats = {
        "index_type": index_type,
        "valid_ratio": float(mask.sum()) / total if total else 0.0,
        "water_ratio": water_count / total if total else 0.0,
        "cloud_ratio": cloud_count / total if total else 0.0,
        "shadow_ratio": shadow_count / total if total else 0.0,
    }
    return output_path, stats


def preprocess_pipeline(image_path, index_type="NDVI", work_dir=None):
    """级别检测 -> 定标/大气校正 -> 按指数类型生成陆地与 NDWI 两套掩膜。"""
    result = {"success": False, "level": None, "processed_path": None,
              "mask_path": None, "mask_paths": {}, "message": ""}
    work_dir = work_dir or new_run_dir()
    processed_path = os.path.join(work_dir, "processed.tif")
    mask_path = os.path.join(work_dir, "mask.npy")
    mask_ndwi_path = os.path.join(work_dir, "mask_ndwi.npy")
    calibrated_path = os.path.join(work_dir, "_calibrated.tif")
    try:
        if not image_path or not os.path.exists(image_path):
            raise FileNotFoundError("影像文件不存在")

        level = check_data_level(image_path)
        result["level"] = level

        if level == "L0":
            calibrated = radiometric_calibration(image_path, calibrated_path)
            atmospheric_correction_dos(calibrated, processed_path)
            try:
                os.remove(calibrated)
            except OSError:
                pass
        elif level == "L1C":
            atmospheric_correction_dos(image_path, processed_path)
        else:
            shutil.copyfile(image_path, processed_path)

        land_mask, land_stats = generate_mask(processed_path, "NDVI", mask_path)
        water_mask, water_stats = generate_mask(processed_path, "NDWI", mask_ndwi_path)
        if land_stats["valid_ratio"] <= 0 and water_stats["valid_ratio"] <= 0:
            raise ValueError("掩膜后没有有效像元")

        is_water = _is_water_index(index_type)
        selected_mask = water_mask if is_water else land_mask
        selected_stats = water_stats if is_water else land_stats

        result.update({
            "success": True,
            "processed_path": os.path.abspath(processed_path),
            "mask_path": os.path.abspath(selected_mask),
            "mask_paths": {
                "land": os.path.abspath(land_mask),
                "water": os.path.abspath(water_mask),
            },
            "mask_stats": selected_stats,
            "land_mask_stats": land_stats,
            "water_mask_stats": water_stats,
            "message": (
                f"预处理完成，影像路径为 {os.path.basename(processed_path)}，掩膜文件已生成。"
                f"数据级别：{level}；陆地指数掩膜有效率约 {land_stats['valid_ratio']:.1%}，"
                f"水体指数掩膜（NDWI/NDTI）有效率约 {water_stats['valid_ratio']:.1%}。"
            ),
        })
    except Exception as e:
        result["message"] = f"预处理失败：{e}"
    return result


def band_numbers(image_paths):
    """返回影像可用的波段号列表（多文件按文件名解析，单文件按 1..N）。"""
    paths = [image_paths] if isinstance(image_paths, str) else list(image_paths)
    paths = [p for p in paths if p and os.path.exists(p)]
    if not paths:
        return []
    try:
        return sorted(_build_sources(paths))
    except Exception:
        return []


def _load_mask(mask_path, shape):
    if not mask_path or not os.path.exists(mask_path):
        return None
    mask = np.load(mask_path).astype(bool)
    if mask.shape == shape:
        return mask
    height, width = shape
    mask_h, mask_w = mask.shape
    rows = (np.arange(height) * mask_h // height).clip(0, mask_h - 1)
    cols = (np.arange(width) * mask_w // width).clip(0, mask_w - 1)
    return mask[np.ix_(rows, cols)]


def _index_renderers(red, nir, green, swir, blue, swir2):
    """波段只读一次，各指数复用同一份数组（读入时已按显示尺寸降采样）。"""

    def ndvi():
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((nir + red) > 0, (nir - red) / (nir + red), np.nan)

    def savi():
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((nir + red) > 0, 1.5 * (nir - red) / (nir + red + 0.5), np.nan)

    def rvi():
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(red > 0, nir / red, np.nan)

    def ndwi():
        with np.errstate(divide='ignore', invalid='ignore'):
            return (green - nir) / (green + nir)

    def ndbi():
        with np.errstate(divide='ignore', invalid='ignore'):
            return (swir - nir) / (swir + nir)

    def nbr():
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((nir + swir2) > 0, (nir - swir2) / (nir + swir2), np.nan)

    def bsi():
        with np.errstate(divide='ignore', invalid='ignore'):
            num = (swir + red) - (nir + blue)
            den = (swir + red) + (nir + blue)
            return np.where(den > 0, num / den, np.nan)

    def ndti():
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((red + green) > 0, (red - green) / (red + green), np.nan)

    def nbr2():
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((swir + swir2) > 0, (swir - swir2) / (swir + swir2), np.nan)

    def lai():
        with np.errstate(divide='ignore', invalid='ignore'):
            value = np.where(red > 0, 0.5 * (nir / red) - 1, np.nan)
        return np.maximum(np.nan_to_num(value, nan=np.nan, posinf=np.nan, neginf=np.nan), 0)

    def chl_proxy():
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where((green + blue) > 0, (green - blue) / (green + blue), np.nan)

    return {
        "NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi, "RVI": rvi, "SAVI": savi,
        "NBR": nbr, "BSI": bsi, "NDTI": ndti, "NBR2": nbr2, "LAI": lai,
        "chl_proxy": chl_proxy,
    }


def _summary_for_index(name, arr):
    """统计有效像元均值、四分位数及分类占比，生成顾问可用的文本摘要。"""
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return f"{name}：掩膜后没有有效像元，数据不足。"

    mean = float(np.mean(valid))
    q25, q50, q75 = (float(v) for v in np.percentile(valid, [25, 50, 75]))

    if name == "NDVI":
        high = float(np.count_nonzero(valid > 0.6)) / valid.size * 100
        medium = float(np.count_nonzero((valid >= 0.2) & (valid <= 0.6))) / valid.size * 100
        low = float(np.count_nonzero(valid < 0.2)) / valid.size * 100
        return (f"NDVI均值为{mean:.2f}，25%/50%/75%分位数为"
                f"{q25:.2f}/{q50:.2f}/{q75:.2f}；"
                f"高覆盖（>0.6）占比{high:.1f}%，"
                f"中覆盖（0.2—0.6）占比{medium:.1f}%，"
                f"低覆盖/裸地（<0.2）占比{low:.1f}%。")

    if name == "NDWI":
        water = float(np.count_nonzero(valid > 0)) / valid.size * 100
        return (f"NDWI均值为{mean:.2f}，25%/50%/75%分位数为"
                f"{q25:.2f}/{q50:.2f}/{q75:.2f}；"
                f"水体（NDWI>0）面积占比约{water:.1f}%。")

    if name == "NDBI":
        built = float(np.count_nonzero(valid > 0)) / valid.size * 100
        return (f"NDBI均值为{mean:.2f}，25%/50%/75%分位数为"
                f"{q25:.2f}/{q50:.2f}/{q75:.2f}；"
                f"建成区/高反射像元（NDBI>0）占比约{built:.1f}%。")

    if name == "chl_proxy":
        green_dominant = float(np.count_nonzero(valid > 0)) / valid.size * 100
        return (f"chl_proxy均值为{mean:.2f}，25%/50%/75%分位数为"
                f"{q25:.2f}/{q50:.2f}/{q75:.2f}；"
                f"绿光相对占优（chl_proxy>0）的水面像元占比约{green_dominant:.1f}%。"
                f"（注意：这只是叶绿素代理指标，不是叶绿素a浓度。）")

    return (f"{name}均值为{mean:.2f}，"
            f"四分位数为{q25:.2f}/{q50:.2f}/{q75:.2f}。")


def calculate_indices(image_paths, red_band: int, nir_band: int,
                      green_band: int = None, swir_band: int = None,
                      blue_band: int = None, swir2_band: int = None,
                      mask_path: str = None, mask_paths: dict = None,
                      output_dir: str = OUTPUT_DIR, indices: list = None):
    """计算十种指数并输出热力图；水体指数与陆地指数分别使用对应掩膜。"""
    paths = [image_paths] if isinstance(image_paths, str) else list(image_paths)
    paths = [p for p in paths if p]
    if not paths:
        return "错误：没有收到影像文件。", []
    missing_files = [p for p in paths if not os.path.exists(p)]
    if missing_files:
        return f"错误：文件不存在 {missing_files[0]}", []

    try:
        os.makedirs(output_dir, exist_ok=True)
        sources = _build_sources(paths)

        if red_band not in sources or nir_band not in sources:
            available = ", ".join(str(b) for b in sorted(sources))
            return (f"错误：缺少红光(第{red_band}波段)或近红外(第{nir_band}波段)数据。"
                    f"当前可用的波段号为：{available}。", [])

        with rasterio.open(paths[0]) as first:
            height, width = first.height, first.width
        out_shape = None
        if height * width > MAX_RENDER_PIXELS:
            scale = math.sqrt(MAX_RENDER_PIXELS / (height * width))
            out_shape = (max(1, int(height * scale)), max(1, int(width * scale)))

        has_green = green_band in sources
        has_swir = swir_band in sources
        has_blue = blue_band in sources
        has_swir2 = swir2_band in sources

        red = _read_band(sources, red_band, out_shape)
        nir = _read_band(sources, nir_band, out_shape)
        green = _read_band(sources, green_band, out_shape) if has_green else None
        swir = _read_band(sources, swir_band, out_shape) if has_swir else None
        blue = _read_band(sources, blue_band, out_shape) if has_blue else None
        swir2 = _read_band(sources, swir2_band, out_shape) if has_swir2 else None

        renderers = _index_renderers(red, nir, green, swir, blue, swir2)
        common_mask = None
        land_mask = None
        water_mask = None
        shape = red.shape if red is not None else None
        if mask_paths and shape:
            land_mask = _load_mask(mask_paths.get("land"), shape)
            water_mask = _load_mask(mask_paths.get("water"), shape)
        elif mask_path and shape:
            common_mask = _load_mask(mask_path, shape)

        sub = f"(R:B{red_band} NIR:B{nir_band})"
        specs = [
            ("NDVI", sub, True),
            ("NDWI", f"(G:B{green_band} NIR:B{nir_band})" if has_green else None, has_green),
            ("NDBI", f"(SWIR1:B{swir_band} NIR:B{nir_band})" if has_swir else None, has_swir),
            ("RVI", sub, True),
            ("SAVI", sub, True),
            ("NBR", f"(NIR:B{nir_band} SWIR2:B{swir2_band})" if has_swir2 else None, has_swir2),
            ("BSI", f"(B:B{blue_band} R:B{red_band} NIR:B{nir_band} SWIR1:B{swir_band})"
             if (has_blue and has_swir) else None, has_blue and has_swir),
            ("NDTI", f"(R:B{red_band} G:B{green_band})" if has_green else None, has_green),
            ("NBR2", f"(SWIR1:B{swir_band} SWIR2:B{swir2_band})"
             if (has_swir and has_swir2) else None, has_swir and has_swir2),
            ("LAI", sub, True),
            ("chl_proxy", f"(B:B{blue_band} G:B{green_band})"
             if (has_blue and has_green) else None, has_blue and has_green),
        ]
        if indices is not None:
            wanted = set(indices)
            specs = [s for s in specs if s[0] in wanted]

        paths_out = []
        summaries = []
        for name, subtitle, available in specs:
            out_path = os.path.join(output_dir, f"{name.lower()}.png")
            if not available:
                continue
            arr = renderers[name]()
            mask = water_mask if (mask_paths and name in ("NDWI", "NDTI", "chl_proxy")) else land_mask
            if mask is None:
                mask = common_mask
            if mask is not None:
                arr = np.where(mask, arr, np.nan)
            summaries.append(_summary_for_index(name, arr))
            _save_heatmap(arr, name, out_path, subtitle)
            paths_out.append(os.path.abspath(out_path))

        missing = [name for name, _, available in specs if not available]
        summary_text = "\n".join(summaries)
        if missing:
            summary_text += f"\n因缺少所需波段，未计算：{', '.join(missing)}。"
        return summary_text, paths_out

    except Exception as e:
        return f"处理失败，错误信息：{str(e)}", []


def calculate_index(image_path, red_band: int, nir_band: int,
                    green_band: int = None, swir_band: int = None,
                    blue_band: int = None, swir2_band: int = None,
                    mask_path: str = None, mask_paths: dict = None,
                    output_dir: str = OUTPUT_DIR, indices: list = None):
    """单文件版本的指数计算入口，按 indices 只计算路由出的指数，供大模型工具调用。"""
    return calculate_indices([image_path], red_band=red_band, nir_band=nir_band,
                             green_band=green_band, swir_band=swir_band,
                             blue_band=blue_band, swir2_band=swir2_band,
                             mask_path=mask_path, mask_paths=mask_paths,
                             output_dir=output_dir, indices=indices)
