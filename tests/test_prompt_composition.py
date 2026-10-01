"""Check the exported prompt composer without API calls."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prompt_composer", ROOT / "scripts/compose_prompt.py")
composer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(composer)


class PromptCompositionTests(unittest.TestCase):
    def test_direct_exact(self):
        self.assertEqual(composer.compose("direct", {}), (ROOT / "prompts/baselines/direct.txt").read_text())

    def test_refinement_preserves_input_xml(self):
        xml = '<TextView android:text="A &amp; B"/>\n'
        self.assertEqual(composer.compose("self_refine", {"xml": xml}),
                         (ROOT / "prompts/baselines/self_refine.txt").read_text() + "\n\nCURRENT XML:\n" + xml)

    def test_s1_requires_actual_dimensions(self):
        with self.assertRaises(ValueError):
            composer.compose("s1_layout_extraction", {"width": 100})
        with self.assertRaises(ValueError):
            composer.compose("s1_layout_extraction", {"width": True, "height": 200})

    def test_bounded_xml_retry(self):
        context = {"s2": "{}", "hint": ""}
        suffix = (ROOT / "prompts/shared/xml_retry_suffix.txt").read_text()
        self.assertEqual(composer.compose("s3_xml_translation", context, xml_retry=2),
                         composer.compose("s3_xml_translation", context) + suffix * 2)
        with self.assertRaises(ValueError):
            composer.compose("direct", {}, xml_retry=1)

    def test_ordered_dcgen_children(self):
        children = ["<TextView/>", "<ImageView/>"]
        expected = (ROOT / "prompts/baselines/dcgen_assembly.txt").read_text().replace(
            "[PARTS]", "Child fragment 0:\n<TextView/>\n\nChild fragment 1:\n<ImageView/>")
        self.assertEqual(composer.compose("dcgen_assembly", {"child_codes": children}), expected)


if __name__ == "__main__":
    unittest.main()
