"""Lower explicitly declared S2 surfaces to native Android XML resources.

No screenshots, OCR, model calls, geometry inference or bitmap output. Call after
generic drawable sanitization and before compilation; record the returned report.
The layout/resource pair must be kept together. Missing style is not a square or
zero-radius prediction and never causes a fallback shape to be invented.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

ANDROID = '{http://schemas.android.com/apk/res/android}'
PREFIX = 's2r_native_shape_'
SURFACE_FIELDS = {'background', 'shape', 'stroke'}
STATEFUL = {'Switch', 'CheckBox', 'RadioButton', 'ToggleButton', 'SeekBar',
            'ProgressBar', 'RatingBar'}
TARGETS = {'Button', 'EditText', 'TextView', 'View', 'LinearLayout', 'FrameLayout',
           'RelativeLayout', 'ScrollView', 'HorizontalScrollView'}


class NativeShapeDeclarationError(ValueError):
    pass


def _object(value, allowed, label):
    if not isinstance(value, dict) or set(value) - set(allowed):
        raise NativeShapeDeclarationError(f'{label}: expected object with only {sorted(allowed)}')
    return value


def _number(value, label, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise NativeShapeDeclarationError(f'{label}: expected finite number')
    if value < 0 or (positive and value == 0):
        raise NativeShapeDeclarationError(f'{label}: expected {"positive" if positive else "nonnegative"} number')
    return float(value)


def _color(value, label):
    if not isinstance(value, str) or not re.fullmatch(r'#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?', value):
        raise NativeShapeDeclarationError(f'{label}: expected literal #RRGGBB or #AARRGGBB')
    return value.upper()


def _argb(color):
    return '#FF'+color[1:].upper() if len(color) == 7 else color.upper()


def _dp(quantity, ratio, label, *, positive=False):
    _object(quantity, {'value', 'unit'}, label)
    if quantity.get('unit') not in {'reference_px', 'dp'}:
        raise NativeShapeDeclarationError(f'{label}: explicit reference_px or dp unit required')
    value = _number(quantity.get('value'), label, positive=positive)
    return value / ratio if quantity['unit'] == 'reference_px' else value


def compile_declared_shape(style: dict, *, reference_px_per_dp: float) -> tuple[str, dict]:
    """Return shape XML plus canonical metadata; invalid declarations raise.

Non-surface visual_style fields (typography/padding/widget state) are left to
their own consumers. No fill is inferred from node.color or the screenshot.
"""
    ratio = _number(reference_px_per_dp, 'reference_px_per_dp', positive=True)
    if not isinstance(style, dict):
        raise NativeShapeDeclarationError('visual_style: expected object')
    background = _object(style.get('background'),
                         {'kind', 'color', 'start_color', 'end_color', 'angle_degrees'}, 'background')
    kind = background.get('kind')
    if kind == 'solid':
        if set(background) != {'kind', 'color'}:
            raise NativeShapeDeclarationError('solid background requires only kind and color')
        fill = f'  <solid android:color="{_color(background["color"], "background.color")}" />'
    elif kind == 'gradient':
        if set(background) != {'kind', 'start_color', 'end_color', 'angle_degrees'}:
            raise NativeShapeDeclarationError('gradient requires both colors and explicit angle_degrees')
        angle = _number(background['angle_degrees'], 'angle_degrees')
        if angle not in range(0, 360, 45):
            raise NativeShapeDeclarationError('gradient angle_degrees must be 0,45,...315')
        fill = (f'  <gradient android:type="linear" android:startColor="{_color(background["start_color"], "start_color")}" '
                f'android:endColor="{_color(background["end_color"], "end_color")}" android:angle="{angle:g}" />')
    else:
        raise NativeShapeDeclarationError('background kind must be solid or gradient')
    shape = style.get('shape')
    if shape is None:
        raise NativeShapeDeclarationError('surface_shape_undeclared')
    shape = _object(shape, {'kind', 'corner_radius'}, 'shape')
    shape_kind = shape.get('kind')
    if shape_kind not in {'rectangle', 'oval', 'circle'}:
        raise NativeShapeDeclarationError('shape.kind must be rectangle, oval or circle')
    lines = [f'<shape xmlns:android="{ANDROID[1:-1]}" android:shape="{"oval" if shape_kind == "circle" else shape_kind}">', fill]
    metadata = {'shape_kind': shape_kind, 'background_kind': kind}
    if shape.get('corner_radius') is not None:
        if shape_kind != 'rectangle':
            raise NativeShapeDeclarationError('corner_radius applies only to rectangle')
        radius = _dp(shape['corner_radius'], ratio, 'corner_radius')
        lines.append(f'  <corners android:radius="{radius:.8g}dp" />')
        metadata['corner_radius_dp'] = radius
    stroke = style.get('stroke')
    if stroke is not None:
        _object(stroke, {'width', 'color'}, 'stroke')
        width = _dp(stroke.get('width'), ratio, 'stroke.width', positive=True)
        border = _color(stroke.get('color'), 'stroke.color')
        lines.append(f'  <stroke android:width="{width:.8g}dp" android:color="{border}" />')
        metadata['stroke_width_dp'] = width
    return '\n'.join([*lines, '</shape>', '']), metadata


def is_native_shape_resource(path: Path, expectedname: str | None = None) -> bool:
    """Validate content-addressed native shape resources, never names alone.

Only the generated scalar-color geometry vocabulary is accepted. Bitmaps,
references, nested drawables, arbitrary XML tags/attributes and mismatched file
hashes are rejected. `expectedname` is the resource name without extension.
"""
    path = Path(path)
    name = path.stem
    if (path.suffix != '.xml' or not re.fullmatch(PREFIX+r'[0-9a-f]{20}', name) or
            (expectedname is not None and expectedname != name)):
        return False
    try:
        if path.stat().st_size > 65536:
            return False
        raw = path.read_bytes()
        if name != PREFIX+hashlib.sha256(raw).hexdigest()[:20]:
            return False
        text = raw.decode('utf-8')
        if '<!' in text or '@' in text or '?' in text:
            return False
        root = ET.fromstring(text)
        if root.tag != 'shape' or set(root.attrib) != {ANDROID+'shape'}:
            return False
        if root.get(ANDROID+'shape') not in {'rectangle', 'oval'} or (root.text or '').strip():
            return False
        children = list(root)
        kinds = Counter(child.tag for child in children)
        if (set(kinds)-{'solid','gradient','corners','stroke'} or any(n != 1 for n in kinds.values()) or
                kinds['solid']+kinds['gradient'] != 1 or
                (root.get(ANDROID+'shape') == 'oval' and kinds['corners'])):
            return False
        def dimension(value, *, positive=False):
            if not isinstance(value, str) or not re.fullmatch(r'\d+(?:\.\d+)?(?:e[+-]?\d+)?dp', value):
                return False
            number = float(value[:-2])
            return math.isfinite(number) and (number > 0 if positive else number >= 0)
        for child in children:
            if list(child) or (child.text or '').strip() or (child.tail or '').strip():
                return False
            attrs = {key.removeprefix(ANDROID):value for key,value in child.attrib.items()}
            if any(not key.startswith(ANDROID) for key in child.attrib):
                return False
            if child.tag == 'solid':
                if set(attrs) != {'color'}:
                    return False
                _color(attrs['color'], 'color')
            elif child.tag == 'gradient':
                if set(attrs) != {'type','startColor','endColor','angle'} or attrs['type'] != 'linear':
                    return False
                _color(attrs['startColor'], 'startColor'); _color(attrs['endColor'], 'endColor')
                if float(attrs['angle']) not in range(0, 360, 45):
                    return False
            elif child.tag == 'corners':
                if set(attrs) != {'radius'} or not dimension(attrs['radius']):
                    return False
            elif child.tag == 'stroke':
                if set(attrs) != {'width','color'} or not dimension(attrs['width'], positive=True):
                    return False
                _color(attrs['color'], 'color')
        return True
    except (OSError, UnicodeDecodeError, ValueError, ET.ParseError):
        return False


def _source_ids(element):
    ids = set()
    rid = element.get(ANDROID+'id', '').rsplit('/', 1)[-1]
    if rid.startswith('s2r_n_'):
        if not re.fullmatch(r's2r_n_\d+(?:_\d+)*', rid):
            return None
        ids.add(rid)
    tag = element.get(ANDROID+'tag', '')
    if tag.startswith('s2r_nodes='):
        values = [item.strip() for item in tag.removeprefix('s2r_nodes=').split(',')]
        if any(not re.fullmatch(r's2r_n_\d+(?:_\d+)*', value) for value in values):
            return None
        ids.update(values)
    return ids


def _explicit_dp(element, name):
    text = element.get(ANDROID+name, '')
    match = re.fullmatch(r'(\d+(?:\.\d+)?)dp', text)
    return float(match.group(1)) if match else None


def bind_declared_native_shapes(xml: str, tree: dict, drawables: Path,
                                *, reference_px_per_dp: float) -> tuple[str, dict]:
    """Bind declared surfaces through one-to-one source identity, preserving UI.

Returns `(new_xml, report)`; all unresolved declarations stay pending. Equal
literal target dimensions are required for a declared circle, not inferred from
screen IDs or forced onto layout geometry. Existing bitmap/unknown resources
and stateful widgets are not repurposed. The caller supplies the verified source
pixel / Android-dp transform; this function never estimates it from image size.
"""
    _number(reference_px_per_dp, 'reference_px_per_dp', positive=True)
    root = ET.fromstring(xml)
    elements = list(root.iter())
    report = {'schema_version': 1, 'status': 'no_declared_native_surfaces', 'applied': [],
              'pending': [], 'source_styles': 0, 'bitmap_outputs': 0,
              'reference_px_per_dp': reference_px_per_dp}
    targets = defaultdict(list)
    android_ids = Counter(node.get(ANDROID+'id') for node in elements if node.get(ANDROID+'id'))
    for target in elements:
        declarations = _source_ids(target)
        if declarations:
            for source_id in declarations:
                targets[source_id].append((target, declarations))
    sources = []
    source_counts = Counter()
    def walk(node, path='$', chrome=False):
        if not isinstance(node, dict):
            return
        chrome = chrome or node.get('system_chrome') in {'status_bar', 'navigation_bar'}
        if node.get('node_id'):
            source_counts[node['node_id']] += 1
        style = node.get('visual_style')
        if ((style is not None and not isinstance(style, dict)) or
                (isinstance(style, dict) and any(style.get(key) is not None for key in SURFACE_FIELDS))):
            sources.append((node, style, path, chrome))
        for i, child in enumerate(node.get('children') or []):
            walk(child, f'{path}.children[{i}]', chrome)
    walk(tree)
    report['source_styles'] = len(sources)
    changed = False
    for source, style, path, chrome in sources:
        sid = source.get('node_id')
        item = {'source_node_id': sid, 'source_path': path}
        def pending(reason):
            report['pending'].append({**item, 'reason': reason})
        if chrome:
            pending('system_chrome_not_app_surface'); continue
        if not sid or source_counts[sid] != 1:
            pending('missing_or_ambiguous_source_identity'); continue
        matches = targets.get(sid, [])
        if len(matches) != 1:
            pending('missing_or_repeated_target_identity'); continue
        target, declarations = matches[0]
        if len(declarations) != 1 or android_ids[target.get(ANDROID+'id')] != 1:
            pending('composite_or_ambiguous_target_identity'); continue
        tag = target.tag.rsplit('}', 1)[-1].rsplit('.', 1)[-1]
        if tag in STATEFUL:
            pending('stateful_widget_requires_native_stateful_style'); continue
        if tag not in TARGETS:
            pending('target_is_not_native_surface'); continue
        old = target.get(ANDROID+'background', '')
        if old.startswith(('@drawable/', '@mipmap/')) and old != '@drawable/img' and not old.startswith('@drawable/'+PREFIX):
            pending('existing_nonshape_background_preserved'); continue
        try:
            resource, metadata = compile_declared_shape(style, reference_px_per_dp=reference_px_per_dp)
        except NativeShapeDeclarationError as exc:
            pending(str(exc)); continue
        # A later XML repair may have corrected the fill. Do not silently undo
        # that accepted change with an older S2 prediction; expose the conflict.
        if (metadata['background_kind'] == 'solid' and re.fullmatch(r'#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?', old)
                and _argb(old) != _argb(style['background']['color'])):
            pending('surface_fill_conflicts_with_final_xml'); continue
        if metadata['shape_kind'] == 'circle':
            width, height = _explicit_dp(target, 'layout_width'), _explicit_dp(target, 'layout_height')
            if width is None or height is None or width <= 0 or abs(width-height) > 1e-6:
                pending('circle_requires_equal_explicit_target_dimensions'); continue
        name = PREFIX + hashlib.sha256(resource.encode()).hexdigest()[:20]
        drawables = Path(drawables)
        drawables.mkdir(parents=True, exist_ok=True)
        file = drawables/f'{name}.xml'
        if file.exists() and file.read_text(encoding='utf-8') != resource:
            raise RuntimeError(f'Native resource hash collision: {file}')
        if not file.exists():
            file.write_text(resource, encoding='utf-8')
        reference = '@drawable/'+name
        changed = changed or target.get(ANDROID+'background') != reference
        target.set(ANDROID+'background', reference)
        if tag == 'Button':
            changed = changed or target.get(ANDROID+'backgroundTint') != '@null'
            target.set(ANDROID+'backgroundTint', '@null')
        report['applied'].append({**item, **metadata, 'android_id': target.get(ANDROID+'id'),
                                  'resource': file.name, 'resource_sha256': hashlib.sha256(resource.encode()).hexdigest()})
    if report['pending']:
        report['status'] = 'pending_declared_surfaces'
    elif sources:
        report['status'] = 'declared_surfaces_materialized_not_render_verified'
    if not changed:
        return xml, report
    ET.register_namespace('android', ANDROID[1:-1])
    return ET.tostring(root, encoding='unicode'), report
