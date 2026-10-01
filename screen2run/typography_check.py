"""Read-only S2 -> XML typography declaration audit, never visual acceptance.

Only explicit source observations and uniquely mapped native text views count.
No OCR, bitmap text, model calls, font guessing, layout edits or size repair.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET

A = '{http://schemas.android.com/apk/res/android}'
TEXT_VIEWS = {'TextView','Button','EditText','CheckBox','RadioButton','Switch','ToggleButton'}


def _finite(value, positive=False):
    return (type(value) in (int,float) and math.isfinite(value) and
            (value > 0 if positive else True))


def _identity(element):
    ids = set()
    name = element.get(A+'id','').rsplit('/',1)[-1]
    if name.startswith('s2r_n_'):
        if not re.fullmatch(r's2r_n_\d+(?:_\d+)*',name): return None
        ids.add(name)
    tag = element.get(A+'tag','')
    if tag.startswith('s2r_nodes='):
        declared = [s.strip() for s in tag.removeprefix('s2r_nodes=').split(',')]
        if any(not re.fullmatch(r's2r_n_\d+(?:_\d+)*',s) for s in declared): return None
        ids.update(declared)
    return ids


def _conversion(frame, font_scale):
    if not _finite(font_scale,True) or font_scale != 1:
        return None, 'requires_explicit_font_scale_1'
    if frame.get('font_scale',font_scale) != font_scale:
        return None, 'conflicting_declared_font_scale'
    scale = frame.get('source_to_screen_dp_scale_xy')
    density = frame.get('px_per_dp')
    if (not isinstance(scale,(list,tuple)) or len(scale)!=2 or
            not all(_finite(x,True) for x in scale) or not _finite(density,True)):
        return None, 'missing_or_invalid_declared_frame_scale'
    sx,sy = scale
    if not math.isclose(sx,sy,rel_tol=1e-9,abs_tol=1e-12):
        size = frame.get('source_frame',{}).get('size_px')
        if (not isinstance(size,(list,tuple)) or len(size)!=2 or
                not all(_finite(x,True) for x in size) or abs(sx-sy)*size[1]*density > .500001):
            return None, 'anisotropic_frame_has_no_single_font_scale'
    return {'sx':sx,'sy':sy,'density':density,'font_scale':font_scale,
            'basis':'declared source x-scale; y difference only allowed for <=0.5 target-pixel canvas rounding; not runtime measured'}, None


def _literal_dimension(value):
    if not isinstance(value,str): return None
    match = re.fullmatch(r'([+]?(?:\d+(?:\.\d*)?|\.\d+))(sp|dp|px)', value)
    if not match: return None
    number = float(match.group(1))
    return (number,match.group(2)) if math.isfinite(number) and number>0 else None


def _argb(value):
    if not isinstance(value,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?',value): return None
    return '#FF'+value[1:].upper() if len(value)==7 else value.upper()


def audit_declared_typography(xml: str, tree: dict, target_frame: dict,
                              *, font_scale: float) -> dict:
    """Return inspectable matched/mismatch/pending rows without changing inputs.

reference_px font sizes use the declared near-isotropic x scale to target dp,
then sp at explicit font_scale=1. Actual XML sp/dp/px values are compared in the
integer pixel bins decoded by Android TextView. Resource/theme values are not
resolved or assumed. Every undeclared field remains unassessed.
"""
    report = {'schema_version':1,'scope':'S2/XML declaration consistency only; no visual or runtime fit pass',
              'xml_sha256':hashlib.sha256(xml.encode()).hexdigest(),
              'tree_sha256':hashlib.sha256(json.dumps(tree,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
              'target_frame':target_frame,'font_scale':font_scale,'nodes':[], 'unassessed_node_ids':[],
              'issues':[], 'acceptance_claim':'none'}
    conversion,error = _conversion(target_frame,font_scale)
    report['conversion'] = conversion
    if error: report['issues'].append({'kind':'pending_conversion','reason':error})
    try: root = ET.fromstring(xml)
    except ET.ParseError as exc:
        report.update(status='pending_typography_contract',issues=[{'kind':'invalid_xml','reason':str(exc)}])
        return report
    targets = defaultdict(list)
    android_ids = Counter(n.get(A+'id').rsplit('/',1)[-1] for n in root.iter() if n.get(A+'id'))
    for target in root.iter():
        ids = _identity(target)
        for identity in ids or ():
            targets[identity].append((target,ids))
    all_sources=[]
    def walk(node,path='$',chrome=False):
        if not isinstance(node,dict): return
        chrome=chrome or node.get('system_chrome') in {'status_bar','navigation_bar'}
        all_sources.append((node,path,chrome))
        for i,child in enumerate(node.get('children') or []): walk(child,f'{path}.children[{i}]',chrome)
    walk(tree)
    source_counts=Counter(n.get('node_id') for n,_,_ in all_sources if n.get('node_id'))
    for node,path,chrome in all_sources:
        visual=node.get('visual_style')
        typo=visual.get('typography') if isinstance(visual,dict) else None
        if not isinstance(typo,dict) or not any(value is not None for value in typo.values()):
            report['unassessed_node_ids'].append(node.get('node_id',path)); continue
        row={'source_node_id':node.get('node_id'),'source_path':path,'checks':[],
             'unassessed_fields':[key for key in ('font_size','color','font_weight','font_style','font_family',
                 'max_lines','include_font_padding','line_height','letter_spacing','alignment') if typo.get(key) is None]}
        report['nodes'].append(row)
        def add(field,expected,actual,status,reason,tolerance=None):
            row['checks'].append({'field':field,'expected':expected,'actual':actual,'status':status,
                                  'reason':reason,'tolerance':tolerance or {'mode':'exact_declared_value'}})
        sid=node.get('node_id'); matches=targets.get(sid,[])
        if chrome or not sid or source_counts[sid]!=1 or len(matches)!=1:
            add('identity',sid,len(matches),'pending','source_system_chrome_or_nonunique_identity'); continue
        target,ids=matches[0]
        row['android_id']=target.get(A+'id')
        tag=target.tag.rsplit('}',1)[-1].rsplit('.',1)[-1]
        if len(ids)!=1 or android_ids[(target.get(A+'id') or '').rsplit('/',1)[-1]]!=1 or tag not in TEXT_VIEWS:
            add('identity',sid,{'tag':tag,'claims':sorted(ids)},'pending','not_unique_native_text_target'); continue
        external={key:target.get(key) for key in ('style',A+'textAppearance') if target.get(key)}
        # XML text is compared literally; resource and theme indirection remains unresolved.
        if isinstance(node.get('text'),str):
            expected=node['text']; actual=target.get(A+'text')
            if actual is None and expected=='': actual=''
            status='pending' if actual is None or actual.startswith(('@','?')) else ('matched' if actual==expected else 'mismatch')
            add('text',expected,actual,status,'literal text equality; no OCR or paraphrase')
        def attr_value(name):
            raw=target.get(A+name)
            if raw is None: return raw,'missing_explicit_attribute_no_theme_default_assumed'
            if raw.startswith(('@','?')): return raw,'resource_or_theme_value_unresolved'
            return raw,None
        font=typo.get('font_size')
        if font is not None:
            raw,why=attr_value('textSize')
            expected={'source_quantity':font}; actual={'xml':raw}
            if (not isinstance(font,dict) or set(font)!={'value','unit'} or
                    not _finite(font.get('value'),True) or font.get('unit') not in {'reference_px','sp'}):
                why='invalid_declared_font_size'
            elif conversion is None: why=error
            else:
                expected_sp=font['value']*conversion['sx'] if font['unit']=='reference_px' else font['value']
                expected_px=expected_sp*conversion['density']
                if not math.isfinite(expected_px):
                    why='font_size_conversion_overflow'
                else:
                    expected.update(sp=expected_sp,unrounded_target_px=expected_px,decoded_target_px=math.floor(expected_px+.5))
                parsed=_literal_dimension(raw)
                if parsed is None: why=why or 'unsupported_or_invalid_textSize_dimension'
                else:
                    value,unit=parsed
                    actual_sp=value/conversion['density'] if unit=='px' else value
                    actual_px=actual_sp*conversion['density']
                    if not math.isfinite(actual_px): why='actual_font_size_conversion_overflow'
                    else: actual.update(sp=actual_sp,unrounded_target_px=actual_px,decoded_target_px=math.floor(actual_px+.5))
            if why: add('font_size',expected,actual,'pending',why)
            else:
                status='matched' if expected['decoded_target_px']==actual['decoded_target_px'] else 'mismatch'
                add('font_size',expected,actual,status,'Android textSize pixel decoding; source pixels are not target sp',
                    {'mode':'same_decoded_integer_pixel','rounding_error_per_value_px':.5})
        style_raw,style_why=attr_value('textStyle')
        style_flags=set(style_raw.split('|')) if style_raw is not None and style_why is None else set()
        if style_flags-set(('normal','bold','italic')): style_why='unsupported_textStyle_value'
        if typo.get('font_weight') is not None:
            declared=str(typo['font_weight']); expected={'normal':400,'bold':700}.get(declared)
            if expected is None and re.fullmatch('[1-9]00',declared): expected=int(declared)
            weight_raw,weight_why=attr_value('textFontWeight')
            if weight_raw is not None and weight_why is None and re.fullmatch('[1-9]00',weight_raw):
                actual=int(weight_raw); why=None
            elif weight_raw is not None: actual=weight_raw; why=weight_why or 'invalid_explicit_font_weight'
            else: actual=700 if 'bold' in style_flags else 400; why=style_why
            if expected is None: why='invalid_declared_font_weight'
            add('font_weight',expected,actual,'pending' if why else ('matched' if expected==actual else 'mismatch'),
                why or 'explicit weight/style flags; actual font-file resolution not verified')
        if typo.get('font_style') is not None:
            actual='italic' if 'italic' in style_flags else 'normal'
            expected=typo['font_style']; why=style_why
            if expected not in ('normal','italic'): why='invalid_declared_font_style'
            add('font_style',expected,actual,'pending' if why else ('matched' if actual==expected else 'mismatch'),
                why or 'explicit XML textStyle flags')
        for field,attribute in [('font_family','fontFamily'),('max_lines','maxLines'),
                                ('include_font_padding','includeFontPadding'),('color','textColor')]:
            expected=typo.get(field)
            if expected is None: continue
            actual,why=attr_value(attribute)
            if ((field=='max_lines' and (type(expected) is not int or expected<=0)) or
                    (field=='include_font_padding' and type(expected) is not bool) or
                    (field=='font_family' and (not isinstance(expected,str) or not expected.strip()))):
                why='invalid_declared_'+field
            if why is None:
                if field=='max_lines':
                    if re.fullmatch(r'[1-9]\d*',actual): actual=int(actual)
                    else: why='invalid_explicit_maxLines'
                elif field=='include_font_padding':
                    if actual in ('true','false'): actual=actual=='true'
                    else: why='invalid_explicit_includeFontPadding'
                elif field=='color':
                    actual,expected=_argb(actual),_argb(expected)
                    if actual is None or expected is None: why='unsupported_literal_color'
            add(field,expected,actual,'pending' if why else ('matched' if actual==expected else 'mismatch'),
                why or 'explicit XML attribute; no default or font-family aliases inferred')
        for field in ('line_height','letter_spacing','alignment'):
            if typo.get(field) is not None:
                # Font metrics, gravity/bidi and lineSpacing combinations require
                # a separate Android runtime audit, not invented equivalence here.
                add(field,typo[field],{name:target.get(A+name) for name in
                    {'line_height':['lineHeight','lineSpacingExtra','lineSpacingMultiplier'],
                     'letter_spacing':['letterSpacing'],'alignment':['textAlignment','gravity']}[field]},
                    'pending','declared_typography_requires_additional_layout_semantics')
        if external:
            row['external_style_attributes']=external
            add('external_style', 'statically resolved typography', external, 'pending',
                'style_or_textAppearance_inheritance_not_resolved')
    checks=[check for row in report['nodes'] for check in row['checks']]
    report['counts']=dict(Counter(check['status'] for check in checks))
    if report['issues'] or any(check['status']!='matched' for check in checks):
        report['status']='pending_typography_contract'
    elif not report['nodes']: report['status']='no_declared_typography'
    else: report['status']='declared_typography_consistent_not_render_verified'
    return report
