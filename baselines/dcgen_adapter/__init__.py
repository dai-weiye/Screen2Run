"""Offline Android XML adaptation layer for DCGenGrid."""

from .config import AdapterConfig, UPSTREAM_COMMIT
from .xml_assembler import assemble_android_xml, count_leaves, expected_call_count
from .xml_parser import AndroidXMLParseError, parse_android_fragment

__all__ = [
    "AdapterConfig",
    "AndroidXMLParseError",
    "UPSTREAM_COMMIT",
    "assemble_android_xml",
    "count_leaves",
    "expected_call_count",
    "parse_android_fragment",
]
