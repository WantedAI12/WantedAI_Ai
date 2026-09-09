"""Package only the pinned research-runtime closure; never publish source data."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
RELEASE_ID='v62-20260910'
WHEEL_REL='dist/odor-calibration-v62/candidate-01/wheel/perfumery_ai_core-1.4.0-py3-none-any.whl'
WHEEL_SHA256='16a44da4a1b371b0e37c001103e4616efa930fe6c0d30b3f590c9342b21981ea'
PROFILE_SHA256='34c7cba01c49d4dcd1e37fefe13e11c0ea24d371900c33a2601f8c9fbd774238'
ROLES=('catalog','perfume','body_lotion','atlas','stock_mixture','lotion_release',
       'unified_product','odor_expression','odor_calibration','lotion_target_reference')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def collect(root=ROOT):
    root=Path(root).resolve();path=root/'perfumery.local.json'
    if sha(path)!=PROFILE_SHA256:
        raise ValueError('the selected V62 profile changed; prepare a new release')
    profile=json.loads(path.read_text(encoding='utf-8'))
    files={}
    def add(path,digest):
        path=Path(path).resolve(strict=True)
        if not path.is_relative_to(root) or sha(path)!=digest:
            raise ValueError('runtime dependency path or hash mismatch')
        name=path.relative_to(root).as_posix()
        if name in files and files[name]!=digest:raise ValueError('conflicting artifact bindings')
        files[name]=digest
        return path
    add(root/WHEEL_REL,WHEEL_SHA256)
    for role in ROLES:
        item=profile[role];path=add(root/item['path'],item['sha256'])
        m=json.loads(path.read_text(encoding='utf-8'))
        if role=='catalog':
            binding=m['runtime_catalog'];add(path.parent/binding['path'],binding['sha256'])
        if role in ('perfume','body_lotion'):
            for name in ('base_model','component_model','registry'):
                spec=m[name];add(path.parent/spec['path'],spec['sha256'])
        if role in ('lotion_release','unified_product','odor_expression'):
            spec=m['weights'];add(path.parent/spec['path'],spec['sha256'])
        if role=='odor_expression':
            add(path.parent/'split.json',m['evaluation_summary']['split_sha256'])
        if role=='odor_calibration':
            spec=m['training_bank'];add(path.parent/spec['path'],spec['sha256'])
    # The Linux service injects the private on-demand language worker. Do not
    # copy a Windows executable or silently choose a different language model.
    profile['language']=None
    return profile,files


def prepare(output):
    output=Path(output).resolve()
    if output.exists():raise ValueError('use a new private bundle directory')
    profile,files=collect()
    output.mkdir(parents=True)
    for name,digest in files.items():
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(ROOT/name,target)
        if sha(target)!=digest:raise ValueError('copied artifact differs')
    profile_path=output/'perfumery.local.json'
    profile_path.write_text(json.dumps(profile,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    files['perfumery.local.json']=sha(profile_path)
    manifest={'release_id':RELEASE_ID,'scope':'authenticated_noncommercial_research_service',
        'source_profile_sha256':PROFILE_SHA256,'profile_sha256':files['perfumery.local.json'],
        'wheel_sha256':WHEEL_SHA256,'files':files,'public_data_redistribution_authorized':False}
    (output/'bundle.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'release_id':RELEASE_ID,'files':len(files),
        'bytes':sum((output/name).stat().st_size for name in files),'bundle_sha256':sha(output/'bundle.json')}))


def verify(root,expected=None):
    root=Path(root).resolve();path=root/'bundle.json'
    if expected is not None and sha(path)!=expected:raise ValueError('runtime bundle hash mismatch')
    m=json.loads(path.read_text(encoding='utf-8'))
    if m['release_id']!=RELEASE_ID or m['wheel_sha256']!=WHEEL_SHA256:
        raise ValueError('wrong runtime release')
    for name,digest in m['files'].items():
        p=(root/name).resolve()
        if not p.is_relative_to(root) or sha(p)!=digest:raise ValueError('runtime bundle file changed')
    return m


def check_installed(root):
    import os
    root=Path(root).resolve();manifest=verify(root)
    if os.environ.get('PERFUMERY_AI_ENV')!='research':
        raise ValueError('this artifact is for explicitly selected research use')
    from fragrance_ai.recommender.runtime import load_configured_catalog
    from fragrance_ai.recommender.perception_runtime import configured_perception
    from fragrance_ai.recommender.fine_odor_model import configured_fine_odor
    from fragrance_ai.recommender.local_runtime import local_atlas_provider,local_profile
    from fragrance_ai.recommender.unified_product import configured_unified_product
    from fragrance_ai import StockMixturePredictor
    catalog,_=load_configured_catalog();provider=configured_perception()
    assert configured_perception('body_lotion') is not provider
    fine=configured_fine_odor();fine.predict(['CCO'])
    unified=configured_unified_product(catalog,component_provider=provider);unified.assert_current()
    path,digest=local_profile()['stock_mixture']
    stock=StockMixturePredictor(provider,path,sha256=digest,experimental=True,atlas_predictor=local_atlas_provider())
    stock.assert_current()
    import resource
    print(json.dumps({'release_id':manifest['release_id'],'installed_model_preflight':'passed',
        'catalog_rows':len(catalog.ingredients),'fine_outputs':len(fine.endpoints),
        'maximum_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--check-installed',type=Path)
    args=p.parse_args()
    if args.output:prepare(args.output)
    elif args.check_installed:check_installed(args.check_installed)
    else:p.error('choose --output or --check-installed')
