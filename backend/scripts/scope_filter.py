import re


# ============================================================
# GENERAL XHS / REDNOTE ZHONGCAO TERMS
# ============================================================

RECOMMENDATION_TERMS = {
    "种草",
    "拔草",
    "推荐",
    "测评",
    "实测",
    "亲测",
    "体验",
    "使用感",
    "使用体验",
    "真实体验",
    "分享",
    "好用",
    "难用",
    "回购",
    "无限回购",
    "避雷",
    "踩雷",
    "平替",
    "值得买",
    "值得入",
    "值得",
    "入手",
    "购买",
    "买了",
    "用了",
    "试用",
    "试了",
    "性价比",
    "必买",
    "宝藏",
    "安利",
    "不推荐",
    "推荐指数",
}


COMMERCE_TERMS = {
    "价格",
    "优惠",
    "折扣",
    "活动",
    "链接",
    "旗舰店",
    "官方",
    "同款",
    "新品",
    "品牌",
    "店铺",
    "购买",
    "下单",
    "入手",
    "性价比",
    "便宜",
    "贵",
}


EXPERIENCE_TERMS = {
    "用了",
    "吃了",
    "住了",
    "去了",
    "试了",
    "体验了",
    "买了",
    "穿了",
    "玩了",
    "看了",
    "喝了",
    "试用",
    "上手",
    "开箱",
}


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if text is None:
        return ""

    text = str(text)

    # Replace repeated whitespace/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# TERM MATCHING
# ============================================================

def matched(text, terms):
    return [
        term
        for term in terms
        if term in text
    ]


# ============================================================
# XHS ZHONGCAO ANALYSIS
# ============================================================

def analyze_scope(text):
    """
    Analyze whether text looks like general XHS / RedNote
    recommendation / zhongcao content.

    This is NOT an AI-authorship classifier.
    It only determines whether the post is in our target scope.
    """

    text = clean_text(text)

    recommendation = matched(
        text,
        RECOMMENDATION_TERMS
    )

    commerce = matched(
        text,
        COMMERCE_TERMS
    )

    experience = matched(
        text,
        EXPERIENCE_TERMS
    )

    score = 0

    if recommendation:
        score += 2

    if commerce:
        score += 1

    if experience:
        score += 1

    if len(recommendation) >= 2:
        score += 1

    # Require a minimum amount of text.
    # Then either:
    # 1. Explicit recommendation/review language
    # OR
    # 2. Both commerce + personal-experience language
    accepted = (
        len(text) >= 20
        and (
            len(recommendation) >= 1
            or (
                len(commerce) >= 1
                and len(experience) >= 1
            )
        )
    )

    return {
        "accepted": accepted,
        "score": score,
        "recommendation_terms": recommendation,
        "commerce_terms": commerce,
        "experience_terms": experience,
    }


# ============================================================
# BOOLEAN HELPER
# ============================================================

def is_xhs_zhongcao(text):
    """
    Simple True / False helper.

    This function exists so prepare_datasets.py can call:

        is_xhs_zhongcao(text)

    without causing the ImportError you saw earlier.
    """

    result = analyze_scope(text)

    return result["accepted"]