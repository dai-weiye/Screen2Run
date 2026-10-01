"""Android XML target adapter for LayoutCoder."""

from .structure_renderer import render_structure
from .xml_extract import XMLExtractionError, extract_android_xml

__all__ = ["XMLExtractionError", "extract_android_xml", "render_structure"]
