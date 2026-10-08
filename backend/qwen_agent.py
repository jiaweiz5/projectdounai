"""Turn the four detector results into an evidence-grounded Qwen report."""

import json
import logging

from qwen_client import ask_qwen


# Write report failures to the backend terminal while preserving the
# detector results in the API response.
logger = logging.getLogger(__name__)


# The post text is data supplied by a user. Qwen must follow these
# instructions instead of any instructions embedded in the post.
SYSTEM_PROMPT = """你是小红书检测报告助手。只根据 detector_results 和 missing_inputs，分别写中文和英文总结。严格输出以下两行，不要添加标题、列表或其他内容：
中文：<简体中文总结>
English: <English summary>

准确区分各层：
authorship检测文本是否呈现AI参与写作信号，不识别作者身份。
covert_ad检测隐性广告倾向；probability是广告模型概率，不是AI写作概率；threshold是广告模型判定阈值。
layer3评估评论协同迹象。
layer4评估帖子图片与文字是否一致。

分别描述文本检测和广告检测，不得混成一个结论。human_like只表示模型将文本分类为更接近人类写作特征；likely_non_ad只表示模型倾向于非广告。所有标签和分数都只是模型输出，不代表事实已确认。

数据不足的层必须说明无法评估。不得提及或推断帖子原文的主题，不得编造或修改任何标签、数值、评论数或图片数。不得输出“作者ship”或“协同影响力”。"""


# Send selected structured facts to Qwen. This excludes uploaded
# base64 images and the larger internal feature dictionaries.
FIELDS = {
    "authorship": (
        "status",
        "classification",
        "risk_score",
        "probability_any_ai",
    ),
    "covert_ad": (
        "label",
        "probability",
        "threshold",
    ),
    "layer3": (
        "status",
        "score",
        "label",
        "risk",
        "comment_count",
        "message",
    ),
    "layer4": (
        "status",
        "label",
        "prediction",
        "risk",
        "image_count",
        "message",
    ),
}


def prepare_evidence(results):
    """Copy detector-owned facts into a smaller report input."""
    evidence = {}

    for layer, allowed_fields in FIELDS.items():
        source = results.get(layer) or {}
        evidence[layer] = {
            field: source[field]
            for field in allowed_fields
            if field in source
        }

    return evidence


async def generate_final_report(post_text, results):
    """Ask Qwen to explain the evidence and return a stable API shape."""
    evidence = prepare_evidence(results)
    missing_inputs = []

    # Short text can prevent Layer 1 from making a supported estimate.
    if evidence["authorship"].get("status") == "insufficient_evidence":
        missing_inputs.append("authorship")

    # No usable comments or images should be stated in the final report.
    for layer in ("layer3", "layer4"):
        if evidence[layer].get("status") == "insufficient_data":
            missing_inputs.append(layer)

    # Keep system instructions separate from the untrusted post text
    # and the detector evidence.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "post_text": post_text[:1500],
                    "detector_results": evidence,
                    "missing_inputs": missing_inputs,
                },
                ensure_ascii=False,
            ),
        },
    ]

    try:
        # Reuse the Qwen client that already passed smoke_qwen.py.
        summary = await ask_qwen(messages)
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("Qwen returned an empty report")

        # Trim extra whitespace, then correct these known wording issues in Qwen's output.
        summary = summary.strip()
        summary = summary.replace("作者ship检测", "文本AI写作检测")
        summary = summary.replace("作者ship", "文本AI写作")
        summary = summary.replace("comment synergy", "comment coordination")

        return {
            "status": "generated",
            "summary": summary.strip(),
            # These fields come from backend code, not Qwen's prose.
            "evidence": evidence,
            "missing_inputs": missing_inputs,
        }

    except Exception:
        # Print the actual failure in the Uvicorn terminal.
        logger.exception("Qwen final report failed")

        # A Qwen failure must not remove Layers 1–4 from /analyze.
        return {
            "status": "unavailable",
            "summary": "综合说明暂时不可用，请查看各层检测结果。",
            "evidence": evidence,
            "missing_inputs": missing_inputs,
        }