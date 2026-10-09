"""Export an exact historical selection and uncapped legacy-rule candidates.

This exports manifests, not motion features or newly measured FK labels.
Only archived path prefixes change in original_manifest.json. Full candidates
are indexed from official annotations and available raw AMASS NPZ headers.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

TASKS = ['raise_hand', 'reach', 'strike', 'wave', 'turn', 'sidestep',
         'back_walk', 'kick', 'jump', 'lean', 'walk']
PATTERNS = [r'(rais|lift|reach).*(hand|arm|up)', r'reach|point|extend.*arm',
            r'\bpunch|\bbox|\bjab', r'\bwav', r'\bturn|\brotat|\bspin',
            r'side.?step|sideways|side to side', r'walk.*back|step.*back|backward.*walk',
            r'\bkick', r'\bjump|\bhop', r'\blean|bend.*forward|\bbow\b', r'\bwalk']
FRAMES = [60, 60, 60, 120, 120, 120, 120, 60, 60, 60, 120]
EXCLUDE = r'crawl|cartwheel|lie down|lying|sit on|somersault'
ALIASES = {'BMLrub': 'BioMotionLab_NTroje', 'DFaust67': 'DFaust_67',
           'EyesJapanDataset': 'Eyes_Japan_Dataset', 'MPIHDM05': 'MPI_HDM05',
           'MPILimits': 'MPI_Limits', 'MPImosh': 'MPI_mosh',
           'SSMsynced': 'SSM_synced', 'Transitionsmocap': 'Transitions_mocap'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def canonical(rel):
    parts = Path(rel).with_suffix('').parts
    # Official archives can omit their dataset wrapper upon extraction.
    if parts[0] == 'ExperimentDatabase':
        parts = ('TCD_handMocap',) + parts
    elif parts[0] == 'mazen_c3d':
        parts = ('Transitions_mocap',) + parts
    return '/'.join([ALIASES.get(parts[0], parts[0]), *parts[1:]])


def inspect_raw(path):
    """Read compressed-array headers and FPS, not large pose arrays or pickle."""
    result = {'raw_path': str(path), 'bytes': path.stat().st_size,
              'mtime_ns': path.stat().st_mtime_ns}
    try:
        with zipfile.ZipFile(path) as z:
            shapes = {}
            for name in ['poses', 'trans']:
                with z.open(name + '.npy') as f:
                    version = np.lib.format.read_magic(f)
                    reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                              else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = reader(f)
                    if dtype.hasobject or not np.issubdtype(dtype, np.number):
                        raise ValueError('non_numeric_' + name)
                    shapes[name] = list(shape)
            field = next((k for k in ['mocap_framerate', 'mocap_frame_rate']
                          if k + '.npy' in z.namelist()), None)
            if field is None:
                raise ValueError('missing_fps')
            with z.open(field + '.npy') as f:
                fps = float(np.lib.format.read_array(f, allow_pickle=False).item())
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError('invalid_fps')
            poses, trans = shapes['poses'], shapes['trans']
            if len(poses) != 2 or poses[1] < 66 or len(trans) != 2 or trans[1] != 3:
                raise ValueError('invalid_array_shape')
            if poses[0] != trans[0] or poses[0] < 2:
                raise ValueError('length_mismatch_or_too_short')
            result.update(status='raw_header_valid', fps=fps, frames=poses[0],
                          shapes=shapes, expected_frames_20fps=int(poses[0] * 20 / fps))
    except Exception as exc:
        result.update(status='invalid_raw_header', error=f'{type(exc).__name__}: {exc}')
    return result


def counts(rows):
    return {s: dict(collections.Counter(r['task'] for r in rows if r['split'] == s))
            for s in ['train', 'val', 'test']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, required=True)
    ap.add_argument('--amass-root', type=Path, required=True)
    ap.add_argument('--annotations-root', type=Path, required=True)
    ap.add_argument('--historical-manifest', type=Path, required=True)
    ap.add_argument('--expected-historical-sha256', required=True)
    ap.add_argument('--raw-audit-cache', type=Path)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, raw_root, out = args.project_root.resolve(), args.amass_root.resolve(), args.output.resolve()
    assert sha(args.historical_manifest) == args.expected_historical_sha256, 'Historical source hash mismatch'
    out.mkdir(parents=True, exist_ok=False)
    old = json.loads(args.historical_manifest.read_text())
    original = []
    for row in old:
        item = dict(row)
        item['path'] = str(root / Path(row['path']).relative_to('/home/pku/frankenmotion'))
        original.append(item)
    strip_path = lambda rows: [{k: v for k, v in r.items() if k != 'path'} for r in rows]
    assert strip_path(original) == strip_path(old)
    shutil.copy2(args.historical_manifest, out / 'original_manifest.charlie.json')
    dump(out / 'original_manifest.json', original)
    print('Historical export verified:', len(original), flush=True)

    annotation_file = args.annotations_root / 'annotations.json'
    ann = json.loads(annotation_file.read_text())
    splits = {s: (args.annotations_root / 'splits' / (s + '.txt')).read_text().split()
              for s in ['train', 'val', 'test']}
    assert all(len(ids) == len(set(ids)) for ids in splits.values())
    assert not (set(splits['train']) & (set(splits['val']) | set(splits['test'])))
    families = {s: {ann[k]['path'] for k in ids} for s, ids in splits.items()}
    overlaps = {a + '_' + b: sorted(families[a] & families[b])
                for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    if any(overlaps.values()):
        dump(out / 'source_split_conflicts.json', overlaps)
        raise RuntimeError('Source split conflicts; do not silently redefine held-out membership')

    index = collections.defaultdict(list)
    inventory = []
    for p in sorted(raw_root.rglob('*.npz')):
        rel = str(p.relative_to(raw_root))
        inventory.append({'relative_path': rel, 'raw_path': str(p), 'bytes': p.stat().st_size})
        index[canonical(rel)].append(p)
    dump(out / 'raw_file_inventory.json', inventory)
    matches, unresolved = {}, {}
    for family in sorted(set().union(*families.values())):
        exact = raw_root / (family + '.npz')
        if exact.is_file():
            matches[family] = (exact, 'exact')
        else:
            candidates = index.get(canonical(family + '.npz'), [])
            if len(candidates) == 1:
                matches[family] = (candidates[0], 'dataset_alias')
            else:
                unresolved[family] = {'reason': 'missing_raw_file' if not candidates else 'ambiguous_raw_path',
                                      'matches': list(map(str, candidates))}
    print('Raw inventory:', len(inventory), 'matched annotated sources:', len(matches), flush=True)
    previous = json.loads(args.raw_audit_cache.read_text()) if args.raw_audit_cache else {}
    raw_info, pending = {}, {}
    for family, (p, method) in matches.items():
        cached = previous.get(family, {})
        st = p.stat()
        if (cached.get('raw_path') == str(p) and cached.get('bytes') == st.st_size
                and cached.get('mtime_ns') == st.st_mtime_ns):
            raw_info[family] = cached
        else:
            pending[family] = p
    reused_headers = len(raw_info)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for i, (family, info) in enumerate(zip(pending, pool.map(inspect_raw, pending.values())), 1):
            raw_info[family] = info
            if i % 1000 == 0:
                print('New raw headers inspected:', i, '/', len(pending), flush=True)
    dump(out / 'annotated_source_raw_audit.json', raw_info)
    dump(out / 'unresolved_annotated_sources.json', unresolved)

    full, all_matches, rejected = [], [], []
    for split, keys in splits.items():
        for tid, (task, pattern, n) in enumerate(zip(TASKS, PATTERNS, FRAMES)):
            seen = set()
            ordered = sorted(keys, key=lambda k: hashlib.sha256(('20261003' + task + k).encode()).hexdigest())
            for key in ordered:
                a = ann[key]; family = a['path']; caption = a.get('caption_label', '')
                if not re.search(pattern, caption, re.I):
                    continue
                reasons = []
                if 'humanact12' in family: reasons.append('excluded_humanact12')
                if not 1 <= a['end'] - a['start'] <= 20: reasons.append('duration_outside_1_20s')
                if re.search(EXCLUDE, caption, re.I): reasons.append('legacy_caption_exclusion')
                if split == 'train' and family in families['val']: reasons.append('validation_source_family')
                if family in unresolved: reasons.append(unresolved[family]['reason'])
                info = raw_info.get(family, {})
                if info and info['status'] != 'raw_header_valid': reasons.append('invalid_raw_header')
                start, end = int(a['start'] * 20), int(a['end'] * 20)
                if a['start'] < 0 or end <= start: reasons.append('invalid_annotation_interval')
                if info.get('status') == 'raw_header_valid' and start >= info['expected_frames_20fps']:
                    reasons.append('empty_crop_at_20fps')
                identity = dict(split=split, task=task, task_id=tid, key=key, family=family, caption=caption)
                if reasons:
                    rejected.append(dict(identity, reasons=reasons)); continue
                available = min(end, info['expected_frames_20fps']) - start
                row = dict(identity, raw_path=str(matches[family][0]), path_match=matches[family][1],
                           annotation_start_s=a['start'], annotation_end_s=a['end'],
                           crop_start_frame_20fps=start, annotation_end_frame_20fps=end,
                           target_frames=n, real_frames=min(n, available), pad_frames=max(0, n-available),
                           annotation_end_exceeds_raw=end > info['expected_frames_20fps'],
                           source_fps=info['fps'], source_frames=info['frames'],
                           feature_path=str(root/'datasets/motions/AMASS_20.0_fps_nh_smplrifke'/(family+'.npy')),
                           path=str(out/'cache'/f'{split}_{task}_{key}.pt'), quantity=None,
                           quantity_status='pending_RiFKE_and_FK', ready_for_training=False,
                           usage='heldout_only' if split == 'test' else ('training' if split == 'train' else 'validation'),
                           source_header_status=info['status'],
                           pending_checks=['numeric_finiteness','body_model_and_feature_conversion',
                                           'text_embedding_alignment','FK_quantity','event_inside_crop'])
                all_matches.append(row)
                if family in seen:
                    rejected.append(dict(identity, reasons=['duplicate_source_within_task_split'],
                                         preserved_in='all_matching_segments.json')); continue
                seen.add(family); full.append(row)
    dump(out/'all_matching_segments.json', all_matches)
    dump(out/'full_candidates.json', full)
    dump(out/'excluded_candidates.json', rejected)
    for split in splits:
        dump(out/f'full_candidates_{split}.json', [r for r in full if r['split'] == split])
    with (out/'full_candidates.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(full[0]) if full else ['key'])
        writer.writeheader()
        for row in full:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k,v in row.items()})

    original_availability = []
    for r in original:
        family = r['family']
        original_availability.append(dict(split=r['split'], task=r['task'], key=r['key'],
            raw_path=str(matches[family][0]) if family in matches else None,
            raw_status=raw_info.get(family, {}).get('status', unresolved.get(family, {}).get('reason')),
            historical_quantity=r['quantity'], cache_path=r['path'], cache_exists=Path(r['path']).is_file()))
    dump(out/'original_availability.json', original_availability)
    pair = lambda r: (r['split'], r['task'], r['key'])
    old_keys, full_keys = set(map(pair, old)), set(map(pair, full))
    summary = dict(historical_count=len(old), original_non_path_fields_exact=True,
        original_order_exact=True, original_non_path_sha256=digest(strip_path(old)),
        historical_sha256=sha(args.historical_manifest), original_export_sha256=sha(out/'original_manifest.json'),
        annotation_sha256=sha(annotation_file), script_sha256=sha(__file__),
        split_sha256={s:sha(args.annotations_root/'splits'/(s+'.txt')) for s in splits},
        amass_root=str(raw_root), project_root=str(root), historical_counts=counts(old),
        full_counts=counts(full), full_count=len(full), all_matching_segment_count=len(all_matches),
        raw_npz_files=len(inventory), unique_annotated_sources=len(matches)+len(unresolved),
        matched_annotated_sources=len(matches), unresolved_annotated_sources=len(unresolved),
        raw_headers_reused_with_same_path_size_mtime=reused_headers,
        flattened_layout_aliases={'ExperimentDatabase/':'TCD_handMocap/ExperimentDatabase/',
                                  'mazen_c3d/':'Transitions_mocap/mazen_c3d/'},
        raw_status_counts=dict(collections.Counter(r['status'] for r in raw_info.values())),
        original_raw_status_counts=dict(collections.Counter(r['raw_status'] for r in original_availability)),
        exclusion_counts=dict(collections.Counter(x for r in rejected for x in r['reasons'])),
        source_split_overlaps=overlaps,
        comparison={'identical_selected_ids':len(old_keys&full_keys),
                    'new_ids':len(full_keys-old_keys), 'historical_ids_not_in_full':len(old_keys-full_keys)},
        rule='Legacy cache.py caption patterns, 1-20s duration, fixed hash ordering, first task window, one source per task/split; per-task caps removed. Test is separately held out.',
        added_checks='Raw NPZ headers, positive finite FPS, array shapes and nonempty crop. Numeric-array finiteness, exact 20fps conversion, text lookup and FK are pending.',
        quantity_policy='Historical quantities preserved only in original manifest; full-candidate quantity is null until recomputed.',
        scope='All official Frankenstein-annotated candidates meeting the rules; unannotated AMASS is inventoried but receives no invented text/task label.')
    dump(out/'audit.json', summary)
    dump(out/'selection_difference.json', {'historical_not_in_full':[list(x) for x in sorted(old_keys-full_keys)],
                                           'new_in_full':[list(x) for x in sorted(full_keys-old_keys)]})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
