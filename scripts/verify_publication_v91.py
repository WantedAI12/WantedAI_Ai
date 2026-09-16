"""Verify deployed-runtime AST parity without requiring private model assets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def canonical_ast(value):
    if isinstance(value, ast.AST):
        return {'node': type(value).__name__, 'fields': {
            key: canonical_ast(item) for key, item in ast.iter_fields(value)
            if key != 'type_params' or item}}
    if isinstance(value, (list, tuple)):
        return [canonical_ast(item) for item in value]
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, complex):
        return {'complex': repr(value)}
    if value is Ellipsis:
        return {'literal': 'ellipsis'}
    return value


def semantic_hash(path):
    tree = ast.parse(path.read_text(encoding='utf8'))
    # ast.dump changed its empty-field rendering in Python 3.13. Serialize
    # fields ourselves so the 3.11 deployment/CI and local 3.13 agree.
    normalized = json.dumps(canonical_ast(tree), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(normalized.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-deployed-source', type=Path)
    args = parser.parse_args()
    path = ROOT/'release/V91/source_manifest.json'
    record = json.loads(path.read_text(encoding='utf8'))
    if args.record_deployed_source:
        expected = {}
        for name, digest in record['runtime_python_files'].items():
            source = args.record_deployed_source/name
            if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
                raise ValueError('deployed source hash mismatch: ' + name)
            expected[name] = semantic_hash(source)
        record['runtime_ast_sha256'] = expected
        record['ast_normalization'] = 'canonical-json-ast/v1; no locations or empty type_params'
        path.write_text(json.dumps(record, indent=2)+'\n', encoding='utf8')
    actual_names = {item.relative_to(ROOT).as_posix() for item in (ROOT/'fragrance_ai').rglob('*.py')}
    if actual_names != set(record['runtime_ast_sha256']):
        raise ValueError('published runtime module set differs from deployed source')
    differences = [name for name, digest in record['runtime_ast_sha256'].items()
                   if semantic_hash(ROOT/name) != digest]
    if differences:
        raise ValueError('runtime implementation differs from deployed source: ' + ', '.join(differences))
    print(json.dumps({'runtime_modules_verified': len(actual_names), 'deployed_runtime_AST_equal': True,
                      'deployed_wheel_sha256': record['deployed_wheel_sha256']}))


if __name__ == '__main__':
    main()
