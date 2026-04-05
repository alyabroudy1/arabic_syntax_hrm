"""
Arabiya Engine — Arabic Linguistic Intelligence System

Transforms raw undiacritized Arabic text into fully diacritized,
prosodically annotated output using syntax-driven analysis.

Usage:
    from arabiya import ArabiyaEngine
    engine = ArabiyaEngine.create_with_mock()
    result = engine.process("ذهب الطالب الى المدرسة")
    print(result.diacritized)
"""

__version__ = "0.1.0"
__author__ = "Arabiya Team"

from arabiya.engine import ArabiyaEngine
from arabiya.core import ArabiyaResult, WordInfo

__all__ = ["ArabiyaEngine", "ArabiyaResult", "WordInfo"]
