"""Portable dataset assembly and value-preservation contracts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_screenshots", ROOT / "scripts/prepare_screenshots.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class ReleaseDataTests(unittest.TestCase):
    def test_fixed_cohort(self):
        records = json.loads((ROOT / "data/screen_lists/screens.json").read_text())["screens"]
        self.assertEqual(len(records), 600)
        self.assertEqual(len({r["screen_id"] for r in records}), 600)
        self.assertEqual([sum(r["split"] == k for r in records) for k in ("Easy", "Real", "Unseen")], [250, 250, 100])
        self.assertEqual([sum(r["split"] == k and r["backbone_subset"] for r in records) for k in ("Easy", "Real", "Unseen")], [50, 50, 20])
        for record in records:
            self.assertFalse(Path(record["file"]).is_absolute())
            measured = json.loads((ROOT / "data/element_measurements" / record["screen_id"] / "measured_elements.json").read_text())
            self.assertEqual(measured["image"], record["file"])
            self.assertEqual((measured["width"], measured["height"]), (record["width"], record["height"]))

    def test_offline_assembly_refuses_overwrite(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "input/android/all_data/test.png"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (12, 24), "blue").save(source)
            manifest = root / "manifest.json"
            item = {"file": "Easy/test.png", "screen_id": "test", "split": "Easy", "width": 12,
                    "height": 24, "sha256": prepare.sha(source.read_bytes()),
                    "upstream": {"member": "android/all_data/test.png"}}
            manifest.write_text(json.dumps({"screens": [item]}))
            args = types.SimpleNamespace(manifest=manifest, out=root / "output", pix2code_root=root / "input",
                                         redraw_root=None, mobileviews_root=None, verify_only=False)
            self.assertEqual(prepare.prepare(args), 0)
            target = root / "output/Easy/test.png"
            self.assertEqual(target.read_bytes(), source.read_bytes())
            target.write_bytes(b"different existing file")
            self.assertEqual(prepare.prepare(args), 1)
            self.assertEqual(target.read_bytes(), b"different existing file")

    def test_reference_targets_not_bundled(self):
        self.assertFalse(list((ROOT / "data").rglob("*.gui")))
        self.assertEqual(len(list((ROOT / "data/element_measurements").glob("*/measured_elements.json"))), 600)

    def test_manifest_cannot_escape_selected_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("../escape.png", "/absolute.png"):
                with self.assertRaises(ValueError):
                    prepare.beneath(root, name)
                with self.assertRaises(ValueError):
                    prepare.locate(root, name)
            (root / "link").symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                prepare.beneath(root, "link/escape.png")


if __name__ == "__main__":
    unittest.main()
