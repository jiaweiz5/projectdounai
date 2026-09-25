import os
import sys
from pprint import pprint

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comment_detector import analyze_comment_coordination
comments = [
    {
        "id": "c01",
        "text": "这个真的太好用了！强烈推荐",
        "timestamp": None,
    },
    {
        "id": "c02",
        "text": "这个真的太好用了，强烈推荐！",
        "timestamp": None,
    },
    {
        "id": "c03",
        "text": "真的很好用，强烈推荐大家购买",
        "timestamp": None,
    },
    {
        "id": "c04",
        "text": "我觉得颜色一般，没有图片那么好看",
        "timestamp": None,
    },
    {
        "id": "c05",
        "text": "包装今天到了，还没开始用",
        "timestamp": None,
    },
    {
        "id": "c06",
        "text": "这个真的太好用了！强烈推荐",
        "timestamp": None,
    },
]


result = analyze_comment_coordination(comments)

pprint(result)