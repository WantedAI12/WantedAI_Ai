"""Verify deployed-runtime AST parity without requiring private model assets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def semantic_hash(path):
    tree = ast.parse(path.read_text(encoding='utf8'))
    # Empty generic parameters were added to CPython's AST in 3.12. They do
    # not change these nongeneric 3.11-compatible runtime implementations.
    normalized = ast.dump(tree, include_attributes=False).replace(', type_params=[]', '')
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
        record['ast_normalization'] = 'CPython AST without locations and empty type_params'
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
