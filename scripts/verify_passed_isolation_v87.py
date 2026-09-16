"""Check baseline source parity and verbatim pass-through of saved successes.

This does not re-run any successful recipe inference or promote stored examples
to new scientific measurements. It tests the routing/response preservation rule.
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['PERFUMERY_AI_LOCAL_PROFILE'] = 'disabled'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def function(path, name):
    tree = ast.parse(path.read_text(encoding='utf8'))
    return next(n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


class BaselineOnly(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        node.decorator_list = [d for d in node.decorator_list if not (isinstance(d, ast.Call)
            and isinstance(d.func, ast.Name) and d.func.id == 'failed_only_recovery')]
        return self.generic_visit(node)

    def visit_ImportFrom(self, node):
        return None if node.module == 'failure_recovery' else node

    def visit_If(self, node):
        if any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ('recovery_active', 'dose_correction_active')
               for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_IfExp(self, node):
        if isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name) and node.test.func.id == 'recovery_active':
            return self.visit(node.orelse)
        return self.generic_visit(node)


class BeforeDoseCorrection(ast.NodeTransformer):
    """Fold only the new, inactive V88 branch in the V87 recovery solver."""
    def visit_ImportFrom(self, node):
        return None if node.module == 'failure_recovery' else node

    def visit_Assign(self, node):
        if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                and node.targets[0].id=='corrected' and isinstance(node.value,ast.Constant)
                and node.value.value is None):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if any((isinstance(value,ast.Call) and isinstance(value.func,ast.Name)
                and value.func.id=='dose_correction_active') or
               (isinstance(value,ast.Name) and value.id=='corrected') for value in ast.walk(node.test)):
            return None
        return self.generic_visit(node)


def main():
    from fragrance_ai.recommender.failure_recovery import failed_only_recovery, recovery_active
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run', type=Path, required=True)
    p.add_argument('--baseline-package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--additional-success-runs', type=Path, nargs='*', default=[])
    p.add_argument('--v87-package', type=Path)
    a = p.parse_args()
    base = a.baseline_package / 'fragrance_ai' / 'recommender'
    now = ROOT / 'fragrance_ai' / 'recommender'
    checks = {}
    for name in ('reference_inverse.py', 'odor_space.py'):
        checks[name] = ast.dump(ast.parse((base/name).read_text(encoding='utf8'))) == ast.dump(ast.parse((now/name).read_text(encoding='utf8')))
    for name, symbol in [('service.py','create_recipe'),('hierarchical_perfume.py','native_proposals'),
                          ('lotion_estimation.py','estimate_lotion_recipe'),
                          ('lotion_reference_search.py','optimize_observed_reference')]:
        old = function(base/name,symbol)
        current = BaselineOnly().visit(function(now/name,symbol))
        checks[name+':'+symbol] = ast.dump(old) == ast.dump(current)
    for symbol in ('limit','report'):
        old = function(base/'search_budget.py',symbol)
        current = BaselineOnly().visit(function(now/'search_budget.py',symbol))
        checks['search_budget.py:'+symbol] = ast.dump(old) == ast.dump(current)
    if a.v87_package:
        reference = a.v87_package/'fragrance_ai'/'recommender'
        name = 'failure_hierarchical_v87.py'
        checks['v87:'+name] = ast.dump(ast.parse((reference/name).read_text(encoding='utf8'))) == ast.dump(ast.parse((now/name).read_text(encoding='utf8')))
        name,symbol = 'failure_inverse_v87.py','source_fixed_proposals'
        old = function(reference/name,symbol)
        if any(isinstance(value,ast.Name) and value.id=='corrected' for value in ast.walk(old)):
            raise ValueError('baseline already uses the newly introduced correction variable')
        current = BeforeDoseCorrection().visit(function(now/name,symbol))
        checks['v87:'+name+':'+symbol] = ast.dump(old)==ast.dump(current)
    if not all(checks.values()):
        raise ValueError('baseline execution path differs: '+json.dumps(checks))
    rows = [json.loads(line) for line in (a.baseline_run/'results.jsonl').read_text(encoding='utf8').splitlines()]
    selected = {(r['case_id'], r['product']):(a.baseline_run, r) for r in rows if r['target_met'] is True}
    baseline_count = len(selected)
    for directory in a.additional_success_runs:
        additional = [json.loads(line) for line in (directory/'results.jsonl').read_text(encoding='utf8').splitlines()]
        for row in additional:
            if row['target_met'] is True:
                selected[(row['case_id'], row['product'])] = directory, row
    count, calls = 0, 0
    for directory, row in selected.values():
        path = directory / 'responses' / row['response_file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['response_sha256']:
            raise ValueError('recorded response drift')
        value = json.loads(gzip.decompress(path.read_bytes()))['response']
        before = digest(value)
        invoked = []
        def baseline_response():
            invoked.append(recovery_active())
            return value
        after = failed_only_recovery(row['product'])(baseline_response)()
        if after is not value or digest(after) != before or invoked != [False]:
            raise ValueError('passed response was changed or recomputed: '+row['case_id'])
        calls += len(invoked)
        count += 1
    report = {'verified': True, 'baseline_source_path_checks': checks,
              'original_success_responses':baseline_count, 'recovered_success_responses':count-baseline_count,
              'recorded_success_responses_checked': count, 'baseline_fixture_calls': calls,
              'recovery_calls_for_successes': 0, 'response_mutations': 0,
              'fresh_recipe_inference_calls': 0, 'deployed_requests': 0,
              'scope': 'source_parity_and_saved_response_passthrough_not_new_model_accuracy'}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise ValueError('new output file required')
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
