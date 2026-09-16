"""Export the accepted model's data lineage, without source or model weights."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    temporary = output.with_suffix('.zip.partial')
    if output.exists() or temporary.exists():
        raise ValueError('Choose a new output; previous archives are preserved')
    profile = read(args.profile)
    model_path = ROOT/profile['formulation_core']['path']
    assert digest(model_path) == profile['formulation_core']['sha256']
    model = read(model_path)
    files, names = [], set()

    def add(path, group, kind, *, relative=None, expected=None):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('Data must be a file inside the selected AI workspace')
        if (path.suffix.lower() in ('.py', '.pyc', '.pt', '.pth', '.pkl', '.pickle', '.joblib', '.whl', '.gguf')
                or path.name in ('model.json', 'weights.npz') or path.name.startswith('.env')):
            raise ValueError('Code, credentials or model checkpoint selected: ' + path.name)
        hashed = digest(path)
        if expected is not None and hashed != expected:
            raise ValueError('Training source checksum mismatch: ' + str(path))
        archive_name = group + '/' + (relative or path.name).replace('\\', '/')
        if archive_name in names or '..' in Path(archive_name).parts:
            raise ValueError('Duplicate or unsafe archive name')
        names.add(archive_name)
        row = {'archive_path': archive_name, 'original_path': path.relative_to(ROOT).as_posix(),
               'kind': kind, 'bytes': path.stat().st_size, 'sha256': hashed}
        if path.suffix == '.npz':
            arrays = {}
            with zipfile.ZipFile(path) as archive:
                for entry in archive.namelist():
                    key = entry.removesuffix('.npy')
                    if key.endswith(('.weight', '.bias')) or 'state_dict' in key:
                        raise ValueError('Model parameter archive selected')
                    with archive.open(entry) as handle:
                        version = np.lib.format.read_magic(handle)
                        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                                  else np.lib.format.read_array_header_2_0)
                        shape, fortran, dtype = reader(handle)
                        if dtype.hasobject:
                            raise ValueError('Pickled object data is not allowed')
                        arrays[key] = {'shape': list(shape), 'dtype': str(dtype)}
            row['arrays'] = arrays
        if path.suffix == '.db':
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as connection:
                assert connection.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
                row['tables'] = [r[0] for r in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        files.append((path, row))

    def manifest_data(folder, group, kind, *, expected=None):
        folder = ROOT/folder
        manifest_path = folder/'manifest.json'
        manifest = read(manifest_path)
        add(manifest_path, group, 'provenance_manifest', expected=expected)
        for name, record in manifest['files'].items():
            hashed = record if isinstance(record, str) else record['sha256']
            add(folder/name, group, kind, relative=name, expected=hashed)
        return manifest

    core_data = manifest_data('.benchmarks/formulation_core_v69/data-03', '02_shared_training',
        'prepared_training_inputs_mixed_observed_and_teacher_labels', expected=model['data_sha256'])
    observed = manifest_data('.benchmarks/v76_system_repair/observed-data-02', '03_observed_formulations/prepared',
        'published_pair_annotations_and_experimental_liquids', expected=model['joint_observed_refit']['data_sha256'])
    manifest_data('.benchmarks/v76_system_repair/public-sources-01', '03_observed_formulations/sources',
        'published_source_data_and_licenses', expected=observed['source_manifest_sha256'])
    nonlinear = ROOT/'.benchmarks/v76_system_repair/nonlinear-teacher-02'
    add(nonlinear/'manifest.json', '04_physics_training/nonlinear_mixtures', 'provenance_manifest')
    add(nonlinear/'mixtures.npz', '04_physics_training/nonlinear_mixtures', 'synthetic_nonlinear_physical_labels',
        expected=model['joint_observed_refit']['physical_teacher_sha256'])
    manifest_data('.benchmarks/v74_autoregressive/bank-01', '04_physics_training/teacher_material_bank',
        'computed_features_and_material_data_not_measured_mixtures')
    inverse = ROOT/'.benchmarks/v76_system_repair/inverse-training-06'
    protocol = read(inverse/'protocol.json')
    add(inverse/'protocol.json', '05_inverse_formulation', 'training_protocol',
        expected=model['autoregressive']['training_protocol_sha256'])
    manifest_data('.benchmarks/v76_system_repair/runtime-bank-02', '05_inverse_formulation/material_bank',
        'computed_features_and_transport_basis', expected=protocol['bank_sha256'])
    episodes = read(inverse/'episodes.json')
    add(inverse/'episodes.json', '05_inverse_formulation', 'train_validation_test_split_index')
    for row in episodes:
        add(inverse/row['file'], '05_inverse_formulation/episodes',
            'synthetic_inverse_task_split_' + str(row['split']), expected=row['sha256'])
    for name in ('perfume-bank.npz', 'body_lotion-bank.npz', 'training-parent-losses.json'):
        add(inverse/name, '05_inverse_formulation', 'derived_training_features_or_teacher_targets')
    process = ROOT/'.benchmarks/v76_system_repair/process-training-01'
    process_protocol = read(process/'protocol.json')
    add(process/'protocol.json', '06_manufacturing_process', 'source_policy_training_protocol',
        expected=model['process_graph_training']['protocol_sha256'])
    add(process/'histories.npz', '06_manufacturing_process', 'source_derived_procedure_labels_not_observed_batches',
        expected=process_protocol['history_sha256'])
    joint = ROOT/'.benchmarks/v76_system_repair/joint-training-03'
    add(joint/'protocol.json', '02_shared_training/joint_refit', 'training_protocol')
    add(joint/'mixture-training.npz', '02_shared_training/joint_refit', 'physical_teacher_training_matrix')

    catalog_path = ROOT/profile['catalog']['path']
    assert digest(catalog_path) == profile['catalog']['sha256']
    catalog = read(catalog_path)['runtime_catalog']
    raw_catalog = catalog_path.parent/catalog['path']
    add(raw_catalog, '01_materials', 'current_material_inventory', expected=catalog['sha256'])
    material_data = json.loads(gzip.decompress(raw_catalog.read_bytes()))
    material_rows = len(material_data['ingredients'])
    add(ROOT/'benchmarks/industrial_ingredient_registry_v1.db', '01_materials',
        'industrial_material_registry', expected=catalog['registry_sha256'])
    support_names = ('safe_ingredient_catalog.json', 'natural_material_compositions.json',
        'odor_threshold_registry.json', 'scientific_properties.db', 'epa_comptox_extract.db',
        'nonhuman_data_hub.db', 'reference_fragrances.db', 'moga_ingredients.db',
        'odor_descriptor_projections.json', 'physical_evidence_v56.json', 'odor_expression_v61.json',
        'formulation_process_knowledge_v1.json')
    for name in support_names:
        add(ROOT/'fragrance_ai/data'/name, '07_supporting_reference_data',
            'supporting_material_or_reference_data_not_all_used_as_training_labels')
    for role in ('lotion_target_reference', 'component_reference_observations', 'odor_space'):
        add(ROOT/profile[role]['path'], '08_current_target_references/' + role,
            'runtime_reference_or_dictionary_not_new_measurement', expected=profile[role]['sha256'])
    physics_path = ROOT/profile['physical_evidence']['path']
    for path in sorted(physics_path.parent.glob('*.json')):
        add(path, '04_physics_training/physical_evidence', 'typed_measured_estimated_or_source_physical_evidence',
            expected=profile['physical_evidence']['sha256'] if path == physics_path else None)
    for path in sorted((ROOT/'.benchmarks/odor_expression_v61/sources').iterdir()):
        if path.is_file():
            add(path, '08_current_target_references/taxonomy_sources', 'odor_vocabulary_source_and_license')

    schemas = {key: model[key] for key in ('actions', 'fine_endpoints', 'quantitative_endpoints', 'blend_labels')}
    lineage = {'accepted_model_manifest_sha256': profile['formulation_core']['sha256'],
        'training_sources': model['training_sources'], 'teacher_bindings': model['teacher_bindings'],
        'joint_refit': model['joint_observed_refit'], 'current_catalog_rows': material_rows,
        'core_data_manifest_sha256': model['data_sha256'],
        'model_checkpoint_files_included': False, 'code_included': False,
        'scope': 'accepted_shared_model_training_data_and_current_material_reference_data_not_all_historical_experiments'}
    inventory = {'schema': 'perfumery-training-data-export/v1', 'created_at': datetime.now(timezone.utc).isoformat(),
        'data_file_count': len(files), 'uncompressed_data_bytes': sum(r['bytes'] for _, r in files),
        'current_catalog_rows': material_rows, 'files': [row for _, row in files],
        'exclusions': ['source code', 'model weights/checkpoints', 'API credentials', 'deployment bundles',
                       'benchmark result logs', 'unaccepted experiments', 'empty input templates',
                       'noncommercial/no-derivatives lotion paper not promoted to training'],
        'redistribution_rights_granted_by_packaging': False}
    readme = f"""# Perfumery AI 학습·원료 데이터 묶음

현재 채택된 공유 조향 모델의 학습 기록과 연결된 데이터만 모았습니다.
원료 목록은 {material_rows:,}행이며, 이것이 모두 학습에 쓰였거나 제조용으로 승인됐다는 뜻은 아닙니다.
학습 입력·검증·시험 분할과 출처 매니페스트를 원본 바이트 그대로 보존했습니다.

## 폴더

- 01_materials: 현재 원료 목록과 산업 원료 레지스트리
- 02_shared_training: 분자 특징·교사 예측 라벨, 전이 계산, 절차, 유화 학습 입력
- 03_observed_formulations: 공개 향료 쌍 주석·액상 제형 실험 원본과 정제 입력
- 04_physics_training: 물성 근거와 물리 계산으로 생성한 혼합 학습값
- 05_inverse_formulation: 역배합 학습용 원료 은행, 에피소드, 학습/검증/시험 분할
- 06_manufacturing_process: 출처 기반 제조 절차 이력과 라벨
- 07_supporting_reference_data: 원료·독성·역치·물성·기준 향 지원 자료
- 08_current_target_references: 기준 향 프로필과 향 표현 사전
- metadata: 파일 목록, 체크섬, 라벨 순서, 사용된 모델/데이터 계보

## 데이터 구분

분자 라벨에는 기존 모델의 교사 예측값이, 물리·역배합 데이터에는 시뮬레이션 값이 포함됩니다.
제조 절차 이력은 출처의 절차 규칙으로 만든 학습값이며 실제 생산 배치 기록이 아닙니다.
향료 쌍 주석은 정량 배합비가 알려진 실측 혼합 관능 결과와 다릅니다.
액상 제형 실험과 유화 입도 실험도 향수·로션의 최종 후각 정확도 자료로 동일시하지 않습니다.
각 파일의 구분은 metadata/DATA_INDEX.json의 kind와 원본 매니페스트를 확인하세요.

학습·검증·시험 자료가 모두 포함되어 있으므로 split을 합쳐 다시 학습하면 기존 평가 분리가 깨집니다.
원본 매니페스트에 적힌 개발 환경 경로는 이 ZIP의 metadata/DATA_INDEX.json에서 original_path와
archive_path를 대응하여 찾을 수 있습니다. 교사 모델 경로/해시는 출처 기록이며 가중치 파일은 없습니다.

## 제외 항목과 사용 범위

프로그램 코드, 신경망 가중치, 체크포인트, API 키, .env, 배포 파일은 넣지 않았습니다.
모든 과거 실험이나 아직 학습에 채택하지 않은 자료를 묶은 것은 아닙니다.
이 ZIP은 데이터 전달용이며 코드와 가중치가 없는 독립 실행 패키지는 아닙니다.
자료별 이용 조건은 그대로 적용됩니다. 이 묶음을 만든 것으로 외부 재배포·상업 이용 권한이
새로 생기는 것은 아닙니다. 포함된 원본 출처와 라이선스를 함께 확인하세요.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    generated = {'README_KO.md': readme.encode('utf8'), 'metadata/DATA_INDEX.json': json_bytes(inventory),
                 'metadata/TRAINING_LINEAGE.json': json_bytes(lineage), 'metadata/LABEL_SCHEMA.json': json_bytes(schemas)}
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for index, (path, row) in enumerate(files):
            archive.write(path, row['archive_path'])
            if digest(path) != row['sha256']:
                raise ValueError('Source changed during packaging')
            if index % 20 == 0:
                print(json.dumps({'archived': index + 1, 'total': len(files)}), flush=True)
        for name, content in generated.items():
            archive.writestr(name, content)
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        for _, row in files:
            with archive.open(row['archive_path']) as handle:
                assert hashlib.file_digest(handle, 'sha256').hexdigest() == row['sha256']
        assert len(archive.infolist()) == len(files) + len(generated)
    if output.exists():
        raise ValueError('Output appeared during packaging; refusing to overwrite')
    temporary.rename(output)
    result = {'zip': str(output), 'zip_bytes': output.stat().st_size, 'zip_sha256': digest(output),
        'data_files': len(files), 'total_zip_entries': len(files) + len(generated),
        'current_catalog_rows': material_rows, 'all_member_hashes_verified': True,
        'checkpoints_and_code_excluded': True, 'uncompressed_data_bytes': inventory['uncompressed_data_bytes']}
    output.with_suffix('.manifest.json').write_bytes(json_bytes(result))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
