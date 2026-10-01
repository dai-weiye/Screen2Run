"""Offline checks for the portable runtime and frozen prompt contracts."""
from __future__ import annotations
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import release_paths

class RuntimeReleaseTests(unittest.TestCase):
    def test_modules_import_without_sdk_or_credentials(self):
        for name in ("model_sessions", "run_model_sessions", "image_filling",
                     "run_image_filling", "content_guard", "device_session",
                     "batch_render", "postprocess", "prompt_baselines_run",
                     "baselines.dcgen_adapter.online_cli", "baselines.layoutcoder_adapter.cli"):
            with self.subTest(module=name):
                importlib.import_module(name)

    def test_python_sources_parse(self):
        for directory in ("screen2run", "baselines", "android_harness"):
            for path in (ROOT / directory).rglob("*.py"):
                with self.subTest(path=str(path.relative_to(ROOT))):
                    ast.parse(path.read_text(encoding="utf-8"))

    def test_cli_help_has_no_external_side_effect(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for relative in ("screen2run/model_sessions.py", "screen2run/run_model_sessions.py",
                         "screen2run/run_image_filling.py", "screen2run/localize_elements.py",
                         "screen2run/content_guard.py", "baselines/run_baseline.py",
                         "baselines/prompt_baselines_run.py", "baselines/postprocess.py",
                         "android_harness/parallel_render.py", "android_harness/batch_render.py",
                         "android_harness/device_session.py"):
            with self.subTest(entry=relative):
                result = subprocess.run([sys.executable, str(ROOT / relative), "--help"],
                    cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_generation_prompts(self):
        import model_sessions
        expected = {
            "PROMPTS": "bf9f1e95098145d147e0cd5bdc3c49718be474e4f1384b2814d75a210a894334",
            "GEOMETRY_NOTE": "f370d4e72044b30e445c61128c6493bead897619237e7b7491b5c111365fd157",
            "IDENTITY_MAPPING_INSTRUCTIONS": "38f86b490f5475027599c5c2bb9f5a2b2785fc149549041dae4a5f920270d796",
            "VISUAL_STYLE_INSTRUCTIONS": "af16bcb5191b3c6a6c860ea1e0465dfcc2b27e82530ca06278038c46db38dce7",
        }
        for name, checksum in expected.items():
            raw = json.dumps(getattr(model_sessions, name), sort_keys=True,
                             ensure_ascii=False).encode()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), checksum, name)

    def test_reference_to_device_coordinates(self):
        from image_filling import Frame
        for width, height in ((688, 1070), (1080, 2340), (1200, 1920)):
            frame = Frame(width, height)
            self.assertAlmostEqual(frame.dp(width), 1080 / 2.625)
            self.assertEqual(frame.screen_h, round(height * 1080 / width))

    def test_portable_reference_path(self):
        self.assertEqual(release_paths.resolve_screenshot("easy/sample.png"),
                         release_paths.SCREENSHOTS / "easy/sample.png")

    def test_custom_screenshot_list(self):
        from unittest.mock import patch
        import run_image_filling
        with tempfile.TemporaryDirectory() as directory:
            cohort = Path(directory) / "screens.txt"
            cohort.write_text("# Custom screenshots\ncustom/example.png\n")
            with patch.dict(os.environ, SCREEN2RUN_SCREENSHOT_LIST=str(cohort)):
                self.assertEqual(run_image_filling.shot_map()["example"],
                                 release_paths.SCREENSHOTS / "custom/example.png")

    def test_prepare_and_ground_outside_the_release(self):
        from argparse import Namespace
        from unittest.mock import patch
        from PIL import Image
        import image_filling
        import run_image_filling
        with tempfile.TemporaryDirectory(prefix="screen2run_portability_") as directory:
            root = Path(directory)
            reference = root / "sample.png"
            Image.new("RGB", (128, 256), "white").save(reference)
            source = root / "source" / "full" / "sample"
            source.mkdir(parents=True)
            xml = ('<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android" '
                   'android:id="@+id/root" android:layout_width="match_parent" '
                   'android:layout_height="match_parent" android:background="#FFFFFF"/>')
            (source / "final_unbound.xml").write_text(xml)
            cohort = root / "cohort.txt"
            cohort.write_text("sample\n")
            with patch.object(run_image_filling, "shot_map", return_value={"sample": reference}), \
                 patch("resource_sanitizer._public_drawables", return_value=frozenset()), \
                 patch("attribute_filter.drop_unknown_android_attrs", side_effect=lambda xml: (xml, [])):
                run_image_filling.cmd_prepare(Namespace(sids=cohort, out=root / "prepared",
                    incremental=False, src=[root / "source"]))
            prepared = root / "prepared" / "full" / "sample" / "final.xml"
            hierarchy = root / "hierarchy.xml"
            hierarchy.write_text('<hierarchy><node resource-id="com.example.myapplication:id/root" '
                'class="android.widget.FrameLayout" bounds="[0,0][1080,2160]" text=""/></hierarchy>')
            with patch.object(image_filling, "load_ocr", return_value=[]), \
                 patch.object(image_filling, "load_components", return_value=[]):
                grounder = image_filling.Grounder("sample", reference, prepared.read_text(), hierarchy, None)
                report = grounder.run(root / "grounded")
            self.assertTrue((root / "grounded" / "final.xml").is_file())
            self.assertEqual(report["live_nodes"], 1)
            self.assertEqual(release_paths.record_path(source), str(source.resolve()))

    def test_host_template_is_source_only(self):
        root = release_paths.HOST_APP
        for name in ("settings.gradle", "build.gradle", "gradlew", "app/build.gradle",
                     "app/src/main/AndroidManifest.xml",
                     "app/src/main/java/com/example/myapplication/MainActivity.java",
                     "app/src/main/res/values/styles.xml"):
            self.assertTrue((root / name).is_file(), name)
        self.assertFalse(any(root.rglob("*.apk")))
        self.assertNotIn("/Users/", (root / "gradlew").read_text())
        self.assertTrue(release_paths.PLACEHOLDER_DRAWABLE.is_file())

    def test_both_original_fonts_are_distributed(self):
        from PIL import ImageFont
        from image_filling import FONT_PATH, VAR_FONT_PATH
        for path in (FONT_PATH, VAR_FONT_PATH):
            self.assertTrue(path.is_file(), str(path))
        variable = ImageFont.truetype(str(VAR_FONT_PATH), 20)
        variable.set_variation_by_axes([700, 100, 0])
        self.assertGreater(variable.getlength("Screen2Run"), 0)

    def test_dcgen_candidate_copy_is_byte_identical(self):
        from baselines.run_baseline import materialize_dcgen_output
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            source = output / "sample"
            source.mkdir()
            xml = b'<FrameLayout android:layout_width="match_parent"/>\n'
            (source / "sample.xml").write_bytes(xml)
            (source / "img.xml").write_text('<shape/>\n')
            materialize_dcgen_output(output, "sample")
            self.assertEqual((output / "final.xml").read_bytes(), xml)
            self.assertEqual((output / "drawables/img.xml").read_bytes(),
                             (source / "img.xml").read_bytes())

    def test_baseline_prepare_has_no_historical_directory_dependency(self):
        from argparse import Namespace
        from unittest.mock import patch
        import postprocess
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            screen = root / "generated/full/sample"
            screen.mkdir(parents=True)
            xml = ('<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android" '
                   'android:layout_width="match_parent" android:layout_height="match_parent"/>')
            (screen / "final.xml").write_text(xml)
            cohort = root / "sample.txt"
            cohort.write_text(str(root / "sample.png") + "\n")
            with patch.dict(os.environ), \
                 patch("resource_sanitizer.PUBLIC_DRAWABLES", frozenset()), \
                 patch.object(postprocess, "drop_unknown_android_attrs", side_effect=lambda xml: (xml, [])):
                result = postprocess.cmd_prepare(Namespace(source=root / "generated", cohort=cohort,
                                                          out=root / "ready"))
            self.assertEqual(result, 0)
            self.assertEqual((root / "ready/full/sample/generated.xml").read_text(), xml)
            self.assertTrue((root / "ready/full/sample/drawables/img.png").is_file())
            self.assertEqual((root / "ready/screenshots.txt").read_text(), str(root / "sample.png") + "\n")

    def test_render_worker_initializes_host_before_locking(self):
        from unittest.mock import patch
        import batch_render
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def device(*args):
                project = root / "projects" / "worker_one"
                self.assertTrue((project / "gradlew").is_file())
                self.assertTrue((project / ".s2r_render.lock").is_file())
                raise RuntimeError("Stop before device access")
            with patch.object(release_paths, "HOST_PROJECTS", root / "projects"), \
                 patch.object(batch_render, "Device", side_effect=device), \
                 self.assertRaisesRegex(RuntimeError, "Stop before device access"):
                batch_render.worker(("emulator-test", "worker_one", "gradle_one"),
                                    root / "candidate", [], root / "render", False, 1)

    def test_prompt_baseline_call_counts_and_payloads(self):
        from prompt_baselines_run import run_direct_arm, load_prompt
        xml = ('<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android" '
               'android:layout_width="match_parent" android:layout_height="match_parent"/>')
        class FakeClient:
            def __init__(self): self.calls = []
            def complete(self, prompt, screenshot, expect):
                self.calls.append((prompt, screenshot, expect))
                return xml
        for arm, count in (("direct", 1), ("cot", 1), ("self_refine", 2)):
            with tempfile.TemporaryDirectory() as directory:
                client = FakeClient()
                board = run_direct_arm(client, Path("input.png"), Path(directory),
                    self_refine=arm == "self_refine", cot=arm == "cot")
                self.assertEqual(len(client.calls), count)
                self.assertEqual(client.calls[0][0],
                    load_prompt("cot_agent.txt" if arm == "cot" else "direct_agent.txt"))
                self.assertEqual(board["variant"], arm)
                self.assertEqual((Path(directory) / "final.xml").read_text(), xml + "\n")

    def test_s4_archive_keeps_raw_review(self):
        from run_model_sessions import _parse_critique
        report = "MISSING IN XML:\n- Search button\nBUILD ERRORS: none"
        parsed = _parse_critique(report)
        self.assertEqual(parsed["raw"], report)
        self.assertEqual(parsed["issues"][0]["msg"], "Search button")

    def test_missing_upstream_is_not_replaced_with_another_algorithm(self):
        from baselines.dcgen_adapter import segmentation
        previous = release_paths.DCGEN
        try:
            with tempfile.TemporaryDirectory() as directory:
                release_paths.DCGEN = Path(directory)
                segmentation._upstream_class.cache_clear()
                with self.assertRaises(FileNotFoundError):
                    segmentation._upstream_class()
        finally:
            release_paths.DCGEN = previous
            segmentation._upstream_class.cache_clear()

if __name__ == "__main__":
    unittest.main()
