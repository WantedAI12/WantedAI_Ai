"""Lossless large diagnostic transport for the backend's 8 MiB JSON reader.

Recipe lines, scores, decisions and regulatory summaries remain ordinary JSON.
Only bulky lotion diagnostic subtrees are archived inside the same response.
No network fetch, persistent blob store, rounding or recomputation is needed.
"""
import base64
import hashlib
import json
import zlib

from fastapi.encoders import jsonable_encoder

FORMAT = 'lossless-lotion-diagnostics/v1'
FIELD = 'diagnostic_archive'
MAX_BYTES = 8_000_000  # Headroom below the unchanged 8 MiB backend buffer.


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf8')


def get_path(value, path):
    for key in path:
        value = value[key]
    return value


def replace_path(value, path, replacement):
    # Copy only container nodes on this path. Do not duplicate all simulation
    # material dictionaries or mutate the cached scientific response.
    result = list(value) if isinstance(value, list) else dict(value)
    key = path[0]
    result[key] = replacement if len(path) == 1 else replace_path(value[key], path[1:], replacement)
    return result


def pack_lotion_response(value):
    native = jsonable_encoder(value)
    raw = encoded(native)
    if len(raw) <= MAX_BYTES:
        return native
    if FIELD in native:
        raise ValueError('diagnostic archive cannot be nested or silently replaced')
    paths = [(key,) for key in ('simulation', 'scenario_simulations', 'fine_expression_search') if key in native]
    regulatory = native.get('regulatory') or {}
    if 'public_source_screen' in regulatory:
        paths.append(('regulatory', 'public_source_screen'))
    for index, tab in enumerate(regulatory.get('tabs') or []):
        if 'public_source_screen' in tab:
            paths.append(('regulatory', 'tabs', index, 'public_source_screen'))
    paths = [path for path in paths if len(encoded(get_path(native, path))) >= 65536]
    fields = [get_path(native, path) for path in paths]
    projected = native
    for index, path in enumerate(paths):
        original = fields[index]
        summary = ({key: item for key, item in original.items()
                    if item is None or type(item) in (str, int, float, bool)} if isinstance(original, dict) else {})
        summary.update(archived_diagnostics=True, archive_field_index=index)
        projected = replace_path(projected, path, summary)
    archive_raw = encoded(fields)
    compressed = base64.b64encode(zlib.compress(archive_raw, level=6)).decode('ascii')
    projected[FIELD] = {'format': FORMAT, 'encoding': 'zlib+base64+utf8-json',
        'paths': [list(path) for path in paths], 'data': compressed,
        'decoded_bytes': len(archive_raw), 'decoded_sha256': hashlib.sha256(archive_raw).hexdigest(),
        'original_json_bytes': len(raw), 'original_canonical_sha256': hashlib.sha256(raw).hexdigest(),
        'lossless': True, 'recipe_or_score_changed': False}
    if len(encoded(projected)) > MAX_BYTES:
        raise ValueError('required result exceeds the backend JSON transport budget')
    return projected


def unpack_lotion_response(value, *, max_decoded_bytes=128 * 1024 * 1024):
    """Offline/audit helper: reconstruct every original JSON value exactly."""
    archive = value.get(FIELD)
    if archive is None:
        return value
    if (archive.get('format') != FORMAT or archive.get('encoding') != 'zlib+base64+utf8-json'
            or type(archive.get('decoded_bytes')) is not int
            or not 0 < archive['decoded_bytes'] <= max_decoded_bytes
            or not isinstance(archive.get('data'), str) or len(archive['data']) > MAX_BYTES):
        raise ValueError('invalid or oversized diagnostic archive')
    try:
        decoder = zlib.decompressobj()
        raw = decoder.decompress(base64.b64decode(archive['data'], validate=True), archive['decoded_bytes'] + 1)
        if (len(raw) != archive['decoded_bytes'] or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail
                or hashlib.sha256(raw).hexdigest() != archive['decoded_sha256']):
            raise ValueError('diagnostic archive size or checksum mismatch')
        fields = json.loads(raw)
        paths = archive['paths']
        if not isinstance(fields, list) or not isinstance(paths, list) or len(fields) != len(paths) or len(paths) > 16:
            raise ValueError('invalid diagnostic field index')
        restored = {key: item for key, item in value.items() if key != FIELD}
        seen = set()
        for index, (path, field) in enumerate(zip(paths, fields)):
            if (not isinstance(path, list) or not 1 <= len(path) <= 4
                    or any(type(key) not in (str, int) for key in path) or tuple(path) in seen):
                raise ValueError('invalid diagnostic field path')
            seen.add(tuple(path))
            marker = get_path(restored, path)
            if marker.get('archived_diagnostics') is not True or marker.get('archive_field_index') != index:
                raise ValueError('diagnostic field marker changed')
            restored = replace_path(restored, path, field)
        if hashlib.sha256(encoded(restored)).hexdigest() != archive['original_canonical_sha256']:
            raise ValueError('reconstructed diagnostic response checksum mismatch')
        return restored
    except (KeyError, TypeError, IndexError, AttributeError, zlib.error) as error:
        raise ValueError('invalid diagnostic archive') from error
