"""citation-guard: local, validated citation-faithfulness guard (verify / re-attribute / flag)."""
from .core import guard, supported, p_attributable, ATTR_MODEL, PROMPT

__version__ = "0.1.0"
__all__ = ["guard", "supported", "p_attributable", "ATTR_MODEL", "PROMPT"]
