import os
import json
import re
from functools import lru_cache
from openai import OpenAI
from dotenv import load_dotenv
from tools import (calculate_index, preprocess_pipeline, band_numbers, parse_metadata,
                   new_run_dir, prune_run_dirs)
from prompts import SYSTEM_PROMPT, CONSULTANT_PROMPT, ROUTING_SYSTEM_PROMPT

load_dotenv()

API_KEY = os.getenv("MOONSHOT_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("KIMI_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL") or "https://api.moonshot.cn/v1"
MODEL = os.getenv("LLM_MODEL") or "kimi-k2.6"
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE") or 1)
USE_LLM = os.getenv("USE_LLM", "true").strip().lower() not in {"0", "false", "no", "off"}

LOCAL_BLOCK_MESSAGE = ("[本地规则拦截] 无法识别该影像的传感器类型，请手动指定红光和近红外波段，"
                       "或开启大模型解析开关。")

# 本地字典未命中、且大模型也没给出指数时，直接反问而不是默认算 NDVI
ASK_BACK_FALLBACK = ("我不太确定您想了解哪方面。请问您是想看植被长势、水域范围、城市建筑，"
                     "还是土地退化/裸地？")

# 自制/经验式指数必须显式打标，不能与标准指数混为一谈
INDEX_LABELS = {
    "chl_proxy": "chl_proxy（自制代理，非标准指数）",
    "LAI": "LAI（经验式估算，非标准指数）",
}

DISCLAIMER = ("本结果仅反映该期影像的光谱特征，未做定量反演与精度验证，"
              "结论仅供初步参考，不能作为决策或评估依据。")

# 变化/趋势类问题至少需要两个时相，单期影像无法回答
CHANGE_INTENT_PATTERN = re.compile(
    r"变化|变好|变差|改善|恶化|趋势|前后|恢复情况|退化了吗|是否退化|有没有退化|退化没|退化严重")
CHANGE_NEEDS_TWO_DATES = (
    "判断“退化/改善/变化”需要至少两个时相的影像，当前只有一期数据，无法回答这类问题。\n\n"
    "您可以选择：① 再提供同一区域的另一期影像；② 改问当前状态，"
    "例如“这里植被覆盖怎么样”“裸地多不多”“水域面积多大”。"
)


def _label(name):
    return INDEX_LABELS.get(name, name)

client = OpenAI(api_key=API_KEY, base_url=BASE_URL) if API_KEY else None

# ⚙️ 预处理与指数计算工具；路径由系统内部注入，模型只负责波段参数
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "preprocess_pipeline",
            "description": (
                "对当前上传的多波段影像执行数据级别检测、辐射定标/轻量大气校正（DOS）"
                "并生成掩膜。计算任何指数前必须先调用本工具。"
                "调用时必须传入本次主要计算的指数类型 index_type；系统会同时生成"
                "陆地指数掩膜（剔除水体、云、阴影）和水体指数掩膜（保留水体，仅剔除云、阴影，适用于 NDWI/NDTI）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "index_type": {
                        "type": "string",
                        "enum": ["NDVI", "NDWI", "NDBI", "RVI", "SAVI",
                                 "NBR", "BSI", "NDTI", "NBR2", "LAI"],
                        "description": "本次主要计算的指数类型，例如 NDVI、NDWI 或 NDTI"
                    }
                },
                "required": ["index_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_index",
            "description": (
                "基于预处理后的影像，按系统路由出的指数子集计算并输出热力图。"
                "必须在 preprocess_pipeline 成功之后调用；掩膜由系统自动应用。"
                "indices 必须严格来自系统的指数路由（见系统提示词中的【指数路由规则】），"
                "不得自行增删指数，多场景冲突时最多 2 个核心指数，其余作为建议在文本中说明。"
                "需要传入蓝、绿、红、近红外、短波红外1、短波红外2六个波段编号。"
                "Landsat 8/9：蓝2、绿3、红4、近红外5、短波红外1为6、短波红外2为7；"
                "Sentinel-2：蓝2、绿3、红4、近红外8、短波红外1为11、短波红外2为12；"
                "高分一号/二号 4 波段（蓝绿红近红）：蓝1、绿2、红3、近红外4，无短波红外（填 null）；"
                "普通 4 波段影像：蓝1、绿2、红3、近红外4，无短波红外（填 null）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["NDVI", "NDWI", "NDBI", "RVI", "SAVI",
                                     "NBR", "BSI", "NDTI", "NBR2", "LAI", "chl_proxy"]
                        },
                        "description": "系统路由决定要计算的指数名列表（已应用数量约束），例如 ['NDVI','NDWI']"
                    },
                    "blue_band": {"type": "integer", "description": "蓝光波段编号，没有则填 null"},
                    "green_band": {"type": "integer", "description": "绿光波段编号"},
                    "red_band": {"type": "integer", "description": "红光波段编号"},
                    "nir_band": {"type": "integer", "description": "近红外波段编号"},
                    "swir_band": {"type": "integer", "description": "短波红外1波段编号，没有则填 null"},
                    "swir2_band": {"type": "integer", "description": "短波红外2波段编号，没有则填 null"}
                },
                "required": ["green_band", "red_band", "nir_band"]
            }
        }
    }
]

FAIL_MESSAGE = "请提供大气校正后的多波段文件（L1C或L2级别）"

DEFAULT_BANDS = {"blue_band": 2, "green_band": 3, "red_band": 4,
                 "nir_band": 5, "swir_band": 6, "swir2_band": 7}  # Landsat 8

SENSOR_BANDS = {
    "sentinel2": {"blue_band": 2, "green_band": 3, "red_band": 4,
                  "nir_band": 8, "swir_band": 11, "swir2_band": 12},
    "landsat": {"blue_band": 2, "green_band": 3, "red_band": 4,
                "nir_band": 5, "swir_band": 6, "swir2_band": 7},
    "gaofen": {"blue_band": 1, "green_band": 2, "red_band": 3,
               "nir_band": 4, "swir_band": None, "swir2_band": None},
    "generic4": {"blue_band": 1, "green_band": 2, "red_band": 3,
                 "nir_band": 4, "swir_band": None, "swir2_band": None},
}

SENSOR_HINTS = [
    ("sentinel2", r"sentinel[ _-]?2|sentinel2|s2a|s2b|msil2a|哨兵[ _-]?2|哨兵二号|哨兵2号"),
    ("landsat", r"landsat[ _-]?(8|9)|landsat[89]|lc0?[89]|陆地卫星[ _-]?[89]|\boli\b|\btirs\b"),
    ("gaofen", r"gf[ _-]?[12]|gaofen[ _-]?[12]|高分[一二12]"),
]

BAND_ROLE_ALIASES = {
    "blue_band": r"(?:\bblue\b|蓝光|蓝波段)",
    "green_band": r"(?:\bgreen\b|绿光|绿波段)",
    "red_band": r"(?:\bred\b|红光|红波段)",
    "nir_band": r"(?:\bnir\b|near[ _-]?infrared|近红外)",
    "swir2_band": r"(?:\bswir[ _-]?2\b|short[ _-]?wave[ _-]?infrared[ _-]?2|短波红外[ _-]?2)",
    "swir_band": r"(?:\bswir\b(?!2)|short[ _-]?wave(?!.?2)|短波红外(?!.?2))",
}

CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
             "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _to_band_number(token):
    token = token.strip()
    if token.isdigit():
        return int(token)
    if token in CN_DIGITS:
        return CN_DIGITS[token]
    if token.startswith("十"):
        return 10 + CN_DIGITS.get(token[1:], 0)
    if token.endswith("十"):
        return CN_DIGITS.get(token[:-1], 1) * 10
    return None


def _parse_explicit_bands(text):
    """支持“红光4”“第四波段是红光”“Band 4 is red”等表达。"""
    result = {}
    number = r"([0-9]{1,2}|[一二三四五六七八九十]{1,3})"
    for key, alias in BAND_ROLE_ALIASES.items():
        match = re.search(alias + r"\s*(?:波段)?\s*(?:是|为|对应|=|:|：)?\s*(?:第)?\s*"
                          + number, text, re.IGNORECASE)
        if not match:
            match = re.search(r"(?:第)?" + number
                              + r"\s*(?:波段|band)?\s*(?:是|为|对应|is|=|:|：)?\s*"
                              + alias, text, re.IGNORECASE)
        if match:
            value = _to_band_number(match.group(1))
            if value:
                result[key] = value
    return result


def _prune_mapping(bands, available, enforce_order=True):
    """把语义波段映射裁剪到实际存在的波段，并剔除明显不合理的短波红外赋值。"""
    red = bands.get("red_band")
    nir = bands.get("nir_band")
    if red not in available or nir not in available:
        return None

    green = bands.get("green_band")
    if green not in available:
        green = None

    swir = bands.get("swir_band")
    if swir not in available:
        swir = None
    if enforce_order and swir is not None and swir <= nir:
        swir = None

    blue = bands.get("blue_band")
    if blue not in available:
        blue = None

    swir2 = bands.get("swir2_band")
    if swir2 not in available:
        swir2 = None
    if enforce_order and swir2 is not None:
        if swir2 <= nir or (swir is not None and swir2 <= swir):
            swir2 = None

    return {"blue_band": blue, "green_band": green, "red_band": red,
            "nir_band": nir, "swir_band": swir, "swir2_band": swir2}


def _local_bands_impl(user_message, image_paths):
    """明确波段 > 本地元数据 > 卫星别名 > 普通4波段模板。"""
    names = " ".join(os.path.basename(p) for p in image_paths)
    hint = f"{user_message} {names}"
    available = set(band_numbers(image_paths))

    explicit = _parse_explicit_bands(hint)
    if "red_band" in explicit and "nir_band" in explicit:
        mapping = {
            "blue_band": explicit.get("blue_band"),
            "green_band": explicit.get("green_band"),
            "red_band": explicit["red_band"],
            "nir_band": explicit["nir_band"],
            "swir_band": explicit.get("swir_band"),
            "swir2_band": explicit.get("swir2_band"),
        }
        # 用户只给了红/近红外时，未指定的波段可由已识别卫星的模板补全
        for sensor, pattern in SENSOR_HINTS:
            if re.search(pattern, hint, re.IGNORECASE):
                template = _prune_mapping(SENSOR_BANDS[sensor], available, enforce_order=False)
                if template:
                    for key in ("blue_band", "green_band", "swir_band", "swir2_band"):
                        if mapping[key] is None:
                            mapping[key] = template.get(key)
                break
        if mapping["swir_band"] is not None and mapping["swir_band"] <= mapping["nir_band"]:
            mapping["swir_band"] = None
        if mapping["swir2_band"] is not None:
            if (mapping["swir2_band"] <= mapping["nir_band"]
                    or (mapping["swir_band"] is not None
                        and mapping["swir2_band"] <= mapping["swir_band"])):
                mapping["swir2_band"] = None
        return mapping, "user_explicit"

    if image_paths:
        try:
            metadata = parse_metadata(image_paths[0])
        except OSError:
            metadata = {}
        meta_mapping = metadata.get("mapping")
        if meta_mapping:
            mapping = _prune_mapping(meta_mapping, available)
            if mapping:
                return mapping, f"metadata:{metadata.get('source')}"

    for sensor, pattern in SENSOR_HINTS:
        if re.search(pattern, hint, re.IGNORECASE):
            mapping = _prune_mapping(SENSOR_BANDS[sensor], available)
            if mapping:
                return mapping, f"sensor:{sensor}"

    if re.search(r"(4|四)[ _-]?波段|4[ _-]?band", hint, re.IGNORECASE):
        if available and max(available) <= 4 and {2, 3, 4} <= available:
            mapping = _prune_mapping(SENSOR_BANDS["generic4"], available)
            if mapping:
                return mapping, "generic4"
    return None, None


@lru_cache(maxsize=256)
def _local_bands_cached(user_message, image_path, size, mtime_ns):
    return _local_bands_impl(user_message, [image_path])


def _local_bands(user_message, image_paths):
    if len(image_paths) == 1:
        try:
            path = os.path.abspath(image_paths[0])
            stat = os.stat(path)
            mapping, source = _local_bands_cached(
                user_message, path, stat.st_size, stat.st_mtime_ns)
            return (dict(mapping), source) if mapping else (None, None)
        except OSError:
            pass
    return _local_bands_impl(user_message, image_paths)


SOURCE_LABELS = {
    "user_explicit": "已按您指定的波段处理",
    "metadata:band_descriptions": "已按影像自带的波段描述识别波段",
    "metadata:landsat_mtl": "已按 Landsat MTL 元数据识别波段",
    "metadata:sentinel2_xml": "已按 Sentinel-2 XML 元数据识别波段",
    "sensor:landsat": "已按 Landsat 8/9 的波段规则处理",
    "sensor:sentinel2": "已按 Sentinel-2 的波段规则处理",
    "sensor:gaofen": "已按高分一号/二号的波段规则处理",
    "generic4": "已按普通 4 波段（蓝、绿、红、近红外）规则处理",
    "llm": "已根据文件名和描述推断波段",
    "default": "无法判断传感器，已按默认的 Landsat 8 波段处理",
}


def _build_reply(mapping, source, images, pre=None, summary=None, routing=None):
    blue = mapping.get("blue_band")
    green = mapping.get("green_band")
    red = mapping.get("red_band")
    nir = mapping.get("nir_band")
    swir = mapping.get("swir_band")
    swir2 = mapping.get("swir2_band")

    lines = []
    if pre and pre.get("success"):
        lines.append(pre.get("message", "预处理完成，掩膜文件已生成。"))
        lines.append("掩膜已生效：陆地指数剔除水体、阴影和云；NDWI/NDTI 保留水体，仅剔除阴影和云。")

    lines.append(
        f"{SOURCE_LABELS.get(source, '已按推断结果处理')}："
        f"蓝光={blue or '无'}，绿光={green or '无'}，红光={red}，近红外={nir}，"
        f"短波红外1={swir or '无'}，短波红外2={swir2 or '无'}。"
    )

    missing = []
    if green is None:
        missing += ["NDWI", "NDTI"]
    if swir is None:
        missing += ["NDBI"]
    if swir2 is None:
        missing += ["NBR"]
    if blue is None or swir is None:
        missing += ["BSI"]
    if swir is None or swir2 is None:
        missing += ["NBR2"]
    if blue is None or green is None:
        missing += ["chl_proxy"]

    formulas = {
        "NDWI": "NDWI = (G − NIR) / (G + NIR)",
        "NDTI": "NDTI = (R − G) / (R + G)",
        "NDBI": "NDBI = (SWIR1 − NIR) / (SWIR1 + NIR)",
        "NBR": "NBR = (NIR − SWIR2) / (NIR + SWIR2)",
        "BSI": "BSI = ((SWIR1 + R) − (NIR + B)) / ((SWIR1 + R) + (NIR + B))",
        "NBR2": "NBR2 = (SWIR1 − SWIR2) / (SWIR1 + SWIR2)",
        "chl_proxy": "chl_proxy = (G − B) / (G + B)",
    }

    # 只对本次路由到的指数提示缺波段，避免无关指数刷屏
    if routing:
        wanted = set(routing.get("compute") or [])
        missing = [name for name in dict.fromkeys(missing) if name in wanted]

    for name in dict.fromkeys(missing):
        lines.append(f"未输出 {name} 热力图，因为缺少所需波段。\n{formulas[name]}")
    if missing:
        lines.append("如需这些指数，请补充对应的波段数据后重新计算。")

    if summary:
        lines.append("【统计摘要】\n" + summary)

    if routing:
        compute = routing.get("compute") or []
        suggest = routing.get("suggest") or []
        lines.append("【指数路由】本次计算：" + "、".join(_label(name) for name in compute) + "。")
        if suggest:
            lines.append("【建议】还可补充计算：" + "、".join(_label(name) for name in suggest)
                         + "（为控制算力，本次未展开，其余未列指数同理不自动计算）。")
        if "chl_proxy" in compute:
            lines.append("【数据局限性】" + CHL_PROXY_LIMITATION)

    lines.append(f"本次共生成 {len(images)} 张指数热力图。")
    lines.append("【说明】" + DISCLAIMER)
    return "\n\n".join(lines)


# =====================================================================
# 指数路由映射（写死在系统内，大模型路由必须严格遵循，不得自行增删）
#
# 第一类：直接明确触发 —— 用户直接喊指数名字，或说出特征关键词，命中即固定映射
# 第二类：模糊业务场景 —— 预设组合，最多不超过组合个数
# 关键约束：多个场景同时命中时，最多只计算最核心的 2 个指数，
#           其余全部在文本里作为“建议”提示，避免一次跑满十个公式。
# =====================================================================

# 全局优先级：决定“最核心的 2 个”怎么选；生态基本面三指数排最前
INDEX_PRIORITY = [
    "NDVI", "NDWI", "NDBI",
    "SAVI", "BSI", "NBR",
    "NDTI", "RVI", "LAI",
    "chl_proxy", "NBR2",
]
ALL_INDEX_WHITELIST = set(INDEX_PRIORITY)

# 大模型智能路由工具（Function Calling Schema）：本地字典未命中时，让模型从白名单选指数
ROUTING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "select_indices",
            "description": "从 11 种遥感指数中选择与用户问题最匹配、需要实际计算的指数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "indices": {
                        "type": "array",
                        "items": {"type": "string", "enum": INDEX_PRIORITY},
                        "description": "需要实际计算的指数（最多2个核心，单一综合体检最多3个）"
                    },
                    "suggest": {
                        "type": "array",
                        "items": {"type": "string", "enum": INDEX_PRIORITY},
                        "description": "建议但本次不计算的指数"
                    }
                },
                "required": ["indices"]
            }
        }
    }
]

# 用户直接喊指数名字时的匹配（按长度降序，避免 NBR2 被 NBR 先匹配）
_CANONICAL = {name.upper(): name for name in INDEX_PRIORITY}
_EXPLICIT_TOKENS = sorted(_CANONICAL, key=len, reverse=True)
EXPLICIT_INDEX_PATTERN = re.compile("|".join(map(re.escape, _EXPLICIT_TOKENS)), re.I)

# 第一类：直接明确触发（特征关键词 -> 指数）
# base：命中即必须触发；modifiers：命中后追加的指数
DIRECT_RULES = [
    {
        "id": "vegetation",
        "label": "植被/绿化/长势/农作物",
        "pattern": r"植被|绿化|长势|农作物|作物|农田|植被覆盖|森林|林地|叶面积",
        "base": ["NDVI"],
        "modifiers": [
            (r"干旱|稀疏", ["SAVI"]),   # 加了“干旱、稀疏” -> 追调 SAVI
            (r"茂密", ["RVI"]),         # 加了“茂密森林” -> 追调 RVI（或 LAI）
            (r"叶面积|LAI", ["LAI"]),   # 明确提叶面积 -> LAI
        ],
    },
    {
        "id": "water",
        "label": "水体/湖泊/河流/淹没/水域面积",
        "pattern": r"水体|湖泊|河流|淹没|水域面积|水域|湖",
        "base": ["NDWI"],
        "modifiers": [],
    },
    {
        "id": "water_quality",
        "label": "水质/浑浊/泥沙/污染",
        "pattern": r"水质|浑浊|泥沙|污染|浊度",
        "base": ["NDTI"],
        "modifiers": [],
    },
    {
        "id": "bloom",
        "label": "水华/藻类/富营养化",
        "pattern": r"水华|藻类|富营养化|叶绿素",
        "base": ["chl_proxy"],
        "modifiers": [],
    },
    {
        "id": "urban",
        "label": "城市/建筑/房屋/违建/不透水面",
        "pattern": r"城市|建筑|房屋|违建|不透水面|建成区|建设用地|城镇",
        "base": ["NDBI"],
        "modifiers": [
            (r"区分裸土", ["BSI"]),   # 强调“区分裸土” -> 配套触发 BSI
        ],
    },
    {
        "id": "bare",
        "label": "裸地/沙化/土地退化/开挖",
        "pattern": r"裸地|沙化|土地退化|开挖",
        "base": ["BSI"],
        "modifiers": [],
    },
    {
        "id": "fire",
        "label": "火灾/着火/火烧迹地/过火面积",
        "pattern": r"火灾|着火|火烧|火烧迹地|过火面积|燃烧",
        "base": ["NBR"],
        "modifiers": [],
    },
]

# 第二类：模糊业务场景（预设组合，需要大模型做语义翻译，但组合已写死）
FUZZY_RULES = [
    {
        "id": "eco_health",
        "label": "生态环境怎么样/全面体检",
        "pattern": r"生态环境|生态怎么样|全面体检|体检|生态基本面|生态状况",
        "indices": ["NDVI", "NDWI", "NDBI"],
    },
    {
        "id": "degradation",
        "label": "这片地退化了吗",
        "pattern": r"退化了吗|是否退化|有没有退化|退化没|退化严重",
        "indices": ["BSI", "NDVI"],
    },
    {
        "id": "suitability",
        "label": "这里适合建房子吗",
        "pattern": r"适合建|建房|盖房|建房子|开发建设|建楼|建厂",
        "indices": ["NDBI", "NDWI", "SAVI"],
    },
    {
        "id": "water_env",
        "label": "水环境怎么样",
        "pattern": r"水环境|水情|水状况|水体状况",
        "indices": ["NDWI", "NDTI"],
    },
]

MAX_CORE_COMPUTE = 2    # 多场景冲突时，最多计算的最核心指数个数
MAX_SINGLE_COMPUTE = 3  # 单场景（如生态体检/建房子）最多计算个数

CHL_PROXY_LIMITATION = (
    "chl_proxy 仅为蓝光/绿光波段的归一化差值代理，反映水面“绿光相对强度”，"
    "只能作为水华/富营养化的粗略线索；它不能换算为叶绿素 a 绝对浓度（μg/L），"
    "也无法区分藻类、悬浮泥沙与有色溶解有机物。判断真实水华需要原位采样，"
    "或使用 MCI/FAI 等专用波段、更高光谱分辨率的数据。"
)


def _priority(index):
    try:
        return INDEX_PRIORITY.index(index)
    except ValueError:
        return len(INDEX_PRIORITY)


def _dedup_by_priority(indices):
    return sorted(set(indices), key=_priority)


def route_indices(user_message):
    """把用户问题路由为 {compute, suggest, scenarios, matched_direct, matched_fuzzy}。"""
    text = user_message or ""

    # 0) 第一类“直接喊名字”：显式写出指数名
    explicit = []
    for match in EXPLICIT_INDEX_PATTERN.finditer(text):
        explicit.append(_CANONICAL[match.group(0).upper()])
    if re.search(r"MNDWI", text, re.I):
        explicit.append("NDWI")
    explicit = _dedup_by_priority(explicit)

    # 1) 第一类“说特征” + 2) 第二类“模糊场景”
    scenario_compute = []
    scenarios = []
    matched_direct = []
    matched_fuzzy = []

    for rule in DIRECT_RULES:
        if re.search(rule["pattern"], text):
            matched_direct.append(rule["label"])
            scenarios.append(rule["id"])
            scenario_compute.extend(rule["base"])
            for mod_pat, mod_idxs in rule.get("modifiers", []):
                if re.search(mod_pat, text):
                    scenario_compute.extend(mod_idxs)

    for rule in FUZZY_RULES:
        if re.search(rule["pattern"], text):
            matched_fuzzy.append(rule["label"])
            scenarios.append(rule["id"])
            scenario_compute.extend(rule["indices"])

    scenario_compute = _dedup_by_priority(scenario_compute)

    # 3) 应用关键约束
    if explicit:
        compute = explicit
        suggest = [i for i in scenario_compute if i not in set(compute)]
    elif len(scenarios) >= 2:
        compute = scenario_compute[:MAX_CORE_COMPUTE]
        suggest = scenario_compute[MAX_CORE_COMPUTE:]
    else:
        compute = scenario_compute[:MAX_SINGLE_COMPUTE]
        suggest = scenario_compute[MAX_SINGLE_COMPUTE:]

    fallback = not explicit and not scenarios

    return {
        "compute": compute,
        "suggest": suggest,
        "scenarios": scenarios,
        "matched_direct": matched_direct,
        "matched_fuzzy": matched_fuzzy,
        "fallback": fallback,
    }


def _detect_index_type(user_message):
    """返回预处理用的 index_type（取路由结果的首个计算指数）。"""
    return route_indices(user_message)["compute"][0]


def _finalize_indices(candidates):
    """把候选指数收束为白名单内、按优先级、最多3个的列表。"""
    result = []
    for name in _dedup_by_priority(candidates or []):
        if name in ALL_INDEX_WHITELIST and name not in result:
            result.append(name)
    return result[:MAX_SINGLE_COMPUTE]


def _llm_route(user_message):
    """本地字典未命中时，用大模型 + Function Calling 做一次智能路由。"""
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ROUTING_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            tools=ROUTING_TOOLS,
            tool_choice="auto",
            temperature=TEMPERATURE,
        )
        message = response.choices[0].message

        if message.tool_calls:
            for call in message.tool_calls:
                if call.function.name != "select_indices":
                    continue
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                compute = _finalize_indices(args.get("indices"))
                if not compute:
                    continue
                suggest = [i for i in _finalize_indices(args.get("suggest"))
                           if i not in set(compute)]
                return {
                    "compute": compute,
                    "suggest": suggest,
                    "scenarios": ["llm_semantic"],
                    "matched_direct": [],
                    "matched_fuzzy": ["大模型语义路由"],
                    "fallback": False,
                }

        # 没有调用工具 -> 视为“太模糊，需要反问”
        content = (message.content or "").strip()
        if content:
            return {"ask_back": content}
    except Exception as e:
        print(f"[Agent] 大模型语义路由失败: {e}")
    return None


def _consult_steps(user_message, summary, mapping, source, images, pre, routing=None):
    """生成器：先吐出图文，再等顾问结论。调用方可以先把热力图显示出来。"""
    base = _build_reply(mapping, source, images, pre, summary, routing)
    yield base, images
    if not USE_LLM or client is None:
        return

    route_note = ""
    if routing:
        route_note = (f"系统指数路由：本次已计算 {', '.join(routing.get('compute') or [])}；"
                      f"建议指数：{', '.join(routing.get('suggest') or []) or '无'}。")
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": CONSULTANT_PROMPT},
                {"role": "user", "content": (
                    f"用户问题：{user_message}\n\n"
                    f"{route_note}\n\n"
                    f"工具统计摘要：\n{summary}\n\n"
                    "请严格依据以上数值给出简明评价；不得使用等级或类别词，"
                    "只做数值引用和本景内的相对比较，并点明数据边界。"
                    "若路由中有建议指数，可在结尾以“建议”形式提及。"
                )},
            ],
            temperature=TEMPERATURE,
        )
        answer = (response.choices[0].message.content or "").strip()
        if answer:
            yield base + "\n\n【分析结论】\n" + answer, images
            return
    except Exception as e:
        print(f"[Agent] 顾问结论生成失败: {e}")

    yield base + "\n\n【分析结论】\n模型分析暂不可用，以上为统计摘要。", images


def _preprocessed_steps(user_message, mapping, source, pre, routing, run_dir):
    status, images = calculate_index(
        pre["processed_path"], mask_paths=pre.get("mask_paths"),
        indices=routing.get("compute"), output_dir=run_dir, **mapping)
    if not images:
        yield status, images
        return
    yield from _consult_steps(user_message, status, mapping, source, images, pre, routing)


def _check_image(image_path):
    try:
        bands = band_numbers([image_path])
    except Exception:
        bands = []
    if not bands:
        return "无法读取该文件，请确认上传的是有效的 GeoTIFF 影像。"
    return None


def run_agent_steps(user_message: str, image_paths=None):
    """生成器：先产出热力图与统计摘要，再产出顾问结论，便于前端边出图边等文字。"""
    image_paths = [p for p in (image_paths or []) if p]
    if not image_paths:
        yield "请先上传一个多波段 GeoTIFF 文件。", []
        return
    image_path = image_paths[0]

    problem = _check_image(image_path)
    if problem:
        yield problem, []
        return

    # 变化/趋势类问题需要两个时相，单期影像直接拒答，避免给出没有依据的结论
    if len(image_paths) < 2 and CHANGE_INTENT_PATTERN.search(user_message or ""):
        yield CHANGE_NEEDS_TWO_DATES, []
        return

    prune_run_dirs()
    routing = route_indices(user_message)

    # 本地字典未命中 -> 大模型智能路由兜底（Function Calling）
    if routing.get("fallback") and USE_LLM and client is not None:
        llm_routing = _llm_route(user_message)
        if llm_routing:
            if "ask_back" in llm_routing:
                yield llm_routing["ask_back"], []   # 终极兜底：太模糊，反问用户
                return
            routing = llm_routing

    # 本地未命中且大模型没有给出指数（未配置/调用失败/模型也判断对不上）：反问
    if not routing.get("compute"):
        yield ASK_BACK_FALLBACK, []
        return

    print(f"[Agent] 指数路由: 计算={routing['compute']}，建议={routing['suggest']}，"
          f"命中场景={routing['scenarios']}")

    run_dir = new_run_dir()

    local, source = _local_bands(user_message, [image_path])
    if local:
        index_type = routing["compute"][0]
        print(f"[Agent] 波段映射（来源={source}）: {local}，指数类型={index_type}，进入预处理")
        try:
            pre = preprocess_pipeline(image_path, index_type=index_type, work_dir=run_dir)
        except Exception as e:
            yield f"预处理出错：{e}", []
            return
        if not pre.get("success"):
            print(f"[Agent] 预处理失败: {pre.get('message')}")
            yield FAIL_MESSAGE, []
            return
        yield from _preprocessed_steps(user_message, local, source, pre, routing, run_dir)
        return

    if not USE_LLM:
        print("[Agent] USE_LLM=false，本地规则未命中，拒绝云端调用")
        yield LOCAL_BLOCK_MESSAGE, []
        return
    if client is None:
        yield ("[本地规则拦截] 未配置大模型 API Key，请配置 MOONSHOT_API_KEY，"
               "或在 .env 中设置 USE_LLM=false 使用纯本地模式。"), []
        return

    file_list = os.path.basename(image_path)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{user_message}\n\n[系统信息] 用户上传了一个多波段文件：{file_list}\n"
            f"[指数路由] 本次应计算：{', '.join(routing['compute'])}；"
            f"建议指数：{', '.join(routing['suggest']) or '无'}。"
            "按要求先调用 preprocess_pipeline；预处理成功后再调用 calculate_index，"
            "且 calculate_index 的 indices 参数必须严格等于上述路由结果。"
            "如果无法判断传感器或波段，请直接用中文询问用户。"
        )}
    ]

    pre = None
    mapping = None
    source = "llm"

    try:
        for _ in range(4):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools_schema,
                tool_choice="auto",
                temperature=TEMPERATURE
            )
            response_message = response.choices[0].message

            if not response_message.tool_calls:
                content = (response_message.content or "").strip()
                if content:
                    if pre is None:
                        pre = preprocess_pipeline(
                            image_path, index_type=routing["compute"][0],
                            work_dir=run_dir)
                        if not pre.get("success"):
                            print(f"[Agent] 预处理失败: {pre.get('message')}")
                            yield FAIL_MESSAGE, []
                            return
                        content = (pre.get("message", "") + "\n\n" + content
                                   + "\n\n掩膜已生效：陆地指数剔除水体、阴影和云；"
                                     "NDWI 保留水体，仅剔除阴影和云。")
                    elif pre.get("success") and "掩膜" not in content:
                        content += ("\n\n掩膜已生效：陆地指数剔除水体、阴影和云；"
                                    "NDWI 保留水体，仅剔除阴影和云。")
                    yield content, []
                    return
                break

            messages.append(response_message)

            for tool_call in response_message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                if name == "preprocess_pipeline":
                    if pre is None:
                        requested_type = str(routing["compute"][0])
                        pre = preprocess_pipeline(image_path, index_type=requested_type,
                                                  work_dir=run_dir)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps(pre, ensure_ascii=False, default=str),
                    })
                    if not pre.get("success"):
                        print(f"[Agent] 预处理失败: {pre.get('message')}")
                        yield FAIL_MESSAGE, []
                        return

                elif name == "calculate_index":
                    auto_preprocessed = False
                    if pre is None or not pre.get("success"):
                        pre = preprocess_pipeline(image_path,
                                                  index_type=routing["compute"][0],
                                                  work_dir=run_dir)
                        auto_preprocessed = True
                        if not pre.get("success"):
                            print(f"[Agent] 预处理失败: {pre.get('message')}")
                            yield FAIL_MESSAGE, []
                            return

                    available = set(band_numbers(pre["processed_path"]))
                    candidate = {key: args.get(key) for key in DEFAULT_BANDS}
                    mapping = _prune_mapping(candidate, available)
                    source = "llm"
                    if mapping is None:
                        mapping = _prune_mapping(DEFAULT_BANDS, available)
                        source = "default"

                    if mapping is None:
                        content = ("无法确定可用波段，当前可用波段号为："
                                   + (", ".join(str(b) for b in sorted(available)) or "无"))
                        messages.append({
                            "role": "tool", "tool_call_id": tool_call.id,
                            "name": name, "content": content,
                        })
                        continue

                    status, images = calculate_index(
                        pre["processed_path"], mask_paths=pre.get("mask_paths"),
                        indices=routing["compute"], output_dir=run_dir, **mapping)
                    if auto_preprocessed:
                        status += "（系统已自动完成预处理）"
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call.id,
                        "name": name, "content": status,
                    })
                    print(f"[Agent] 波段映射（来源={source}）: {mapping}，预处理级别={pre.get('level')}")
                    if images:
                        yield from _consult_steps(
                            user_message, status, mapping, source, images, pre, routing)
                        return

                else:
                    messages.append({
                        "role": "tool", "tool_call_id": tool_call.id,
                        "name": name, "content": "未知工具",
                    })

        if pre and pre.get("success"):
            available = set(band_numbers(pre["processed_path"]))
            mapping = _prune_mapping(DEFAULT_BANDS, available)
            if mapping:
                source = "default"
                status, images = calculate_index(
                    pre["processed_path"], mask_paths=pre.get("mask_paths"),
                    indices=routing["compute"], output_dir=run_dir, **mapping)
                if images:
                    yield from _consult_steps(
                        user_message, status, mapping, source, images, pre, routing)
                    return
                yield status, []
                return

        yield "无法完成计算，请补充传感器信息或各光谱的波段编号。", []
        return

    except Exception as e:
        yield f"Agent 运行出错: {str(e)}", []
        return


def run_agent(user_message: str, image_paths=None):
    """一次性拿最终结果的便捷入口。"""
    last = ("", [])
    for last in run_agent_steps(user_message, image_paths):
        pass
    return last
