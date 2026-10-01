"""Read-only public-data validation; no private paths or participant metadata."""
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/human_study'
VDIMS = {'overall', 'layout', 'text', 'image', 'style', 'replace', 'rank'}
CDIMS = {'readability', 'maintainability', 'practice', 'usability', 'rank'}


class HumanReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ratings = json.loads((DATA / 'ratings.json').read_text())
        cls.expected = json.loads((DATA / 'expected_results.json').read_text())
        cls.stimuli = json.loads((DATA / 'stimuli_manifest.json').read_text())
        cls.rows = cls.ratings['records']

    def test_exact_public_schema_and_analysis_orders(self):
        self.assertEqual(set(self.ratings), {'schema', 'raters', 'methods', 'items', 'records'})
        self.assertEqual(self.ratings['schema'], 'screen2run-human-ratings/1')
        self.assertEqual(self.ratings['raters'], [f'R{i}' for i in range(1, 7)])
        self.assertEqual(self.ratings['methods'], ['layoutcoder', 'dcgen', 'ours'])
        self.assertEqual(self.ratings['items']['v'], self.ratings['items']['c'])
        self.assertEqual(len(set(self.ratings['items']['v'])), 60)
        self.assertEqual(len(self.rows), 2160)

    def test_complete_unique_design_and_value_ranges(self):
        seen = set()
        for row in self.rows:
            self.assertEqual(set(row), {'part', 'rater', 'item_id', 'screen_id', 'dataset', 'method', 'values', 'collection_round'})
            key = tuple(row[k] for k in ('part', 'rater', 'item_id', 'method'))
            self.assertNotIn(key, seen); seen.add(key)
            self.assertIn(row['part'], ('v', 'c'))
            self.assertIn(row['rater'], self.ratings['raters'])
            self.assertIn(row['method'], self.ratings['methods'])
            self.assertIn(row['item_id'], self.ratings['items'][row['part']])
            self.assertEqual(set(row['values']), VDIMS if row['part'] == 'v' else CDIMS)
            for field, value in row['values'].items():
                self.assertIs(type(value), int)
                self.assertIn(value, (0, 1) if field == 'replace' else range(1, 4) if field == 'rank' else range(1, 6))
        for part in ('v', 'c'):
            for rater in self.ratings['raters']:
                for uid in self.ratings['items'][part]:
                    self.assertTrue(all((part, rater, uid, m) in seen for m in self.ratings['methods']))

    def test_rank_permutations(self):
        groups = collections.defaultdict(list)
        for row in self.rows:
            groups[(row['part'], row['rater'], row['item_id'])].append(row['values']['rank'])
        self.assertEqual(len(groups), 720)
        self.assertTrue(all(sorted(v) == [1, 2, 3] for v in groups.values()))

    def test_screen_roster_and_collection_rounds(self):
        roster = {x['item_id']: x for x in self.stimuli['items']}
        self.assertEqual(list(roster), self.ratings['items']['v'])
        self.assertEqual(collections.Counter(x['dataset'] for x in roster.values()), {'Easy': 25, 'Real': 25, 'Unseen': 10})
        self.assertEqual(collections.Counter(x['code_collection_round'] for x in roster.values()), {'initial': 12, 'code_supplement': 48})
        for row in self.rows:
            item = roster[row['item_id']]
            self.assertEqual((row['screen_id'], row['dataset']), (item['screen_id'], item['dataset']))
            self.assertEqual(row['collection_round'], 'initial' if row['part'] == 'v' else item['code_collection_round'])

    def test_all_36_means_match_frozen_targets(self):
        values = {(x['part'], x['rater'], x['item_id'], x['method']): x['values'] for x in self.rows}
        checks = 0
        for slot in ('primary', 'secondary'):
            for outcome, target in self.expected[slot].items():
                part, dimension = outcome.split(':')
                self.assertEqual(target['n_items'], 60)
                self.assertEqual(target['n_raters'], 6)
                for method in self.ratings['methods']:
                    means = [math.fsum(values[(part, r, uid, method)][dimension] for r in self.ratings['raters']) / 6 for uid in self.ratings['items'][part]]
                    self.assertAlmostEqual(math.fsum(means) / 60, target['mean'][method], places=12)
                    checks += 1
        self.assertEqual(checks, 36)

    def test_code_round_means_match(self):
        for public_round, target_round in [('initial', 'frozen12'), ('code_supplement', 'recovery48')]:
            target = self.expected['code_rounds'][target_round]
            for field, methods in target['mean'].items():
                for method, mean in methods.items():
                    values = [x['values'][field] for x in self.rows if x['part'] == 'c' and x['method'] == method and x['collection_round'] == public_round]
                    self.assertEqual(len(values), target['n_items'] * 6)
                    self.assertAlmostEqual(math.fsum(values) / len(values), mean, places=12)

    def test_exact_displayed_code_and_no_image_bundles(self):
        listed = set()
        for item in self.stimuli['items']:
            self.assertEqual(set(item['stimuli']), set(self.ratings['methods']))
            for info in item['stimuli'].values():
                relative = Path(info['display_xml'])
                self.assertFalse(relative.is_absolute())
                self.assertNotIn('..', relative.parts)
                path = DATA / relative
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), info['display_xml_sha256'])
                ET.fromstring(path.read_bytes())
                listed.add(path)
        self.assertEqual(len(listed), 180)
        self.assertEqual(listed, set((DATA / 'code').rglob('*.xml')))
        self.assertFalse(self.stimuli['images_included'])
        self.assertFalse(self.stimuli['drawable_assets_included'])
        self.assertFalse(any(p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp') for p in DATA.rglob('*')))

    def test_numeric_exports_have_no_private_metadata(self):
        forbidden = {'user_agent', 'exported_at', 'started', 'updated', 'consent', 'bg', 'background', 'end',
                     'email', 'phone', 'contact', 'source_file', 'source_id', 'source_key_id', 'json_pointer',
                     'round_metadata', 'collection_rounds', 'key_sha256', 'code_labels', 'comment', 'comments'}
        def walk(obj):
            if isinstance(obj, dict):
                self.assertFalse(set(obj) & forbidden)
                for value in obj.values(): walk(value)
            elif isinstance(obj, list):
                for value in obj: walk(value)
            elif isinstance(obj, str):
                self.assertNotIn('/Users/', obj)
                self.assertNotIn('artifacts/tsc_method', obj)
                self.assertNotIn('Mozilla/', obj)
        for obj in (self.ratings, self.expected, self.stimuli): walk(obj)

    def test_english_metadata_and_blank_template(self):
        for name in ('ratings.json', 'expected_results.json', 'stimuli_manifest.json', 'ratings.schema.json', 'questionnaire_template.html'):
            text = (DATA / name).read_text()
            self.assertIsNone(re.search('[\u3400-\u9fff]', text), name)
        html = (DATA / 'questionnaire_template.html').read_text()
        self.assertIn('<html lang="en">', html)
        self.assertIn('screen2run-replication-return/1', html)
        self.assertNotIn('fetch(', html)
        self.assertNotIn('XMLHttpRequest', html)
        self.assertNotIn('ratings.json', html)
        self.assertNotIn('ours', html)


if __name__ == '__main__':
    unittest.main()
