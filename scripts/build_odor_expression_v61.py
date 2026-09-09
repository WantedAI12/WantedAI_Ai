"""Build an attributed, extensible language registry; never invent intensities.

The expert taxonomy is used as vocabulary, not as new measured odor labels.
The molecular annotations remain in the separately pinned research checkpoint.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REPO = 'https://github.com/akshay-sajann/computer_ontology'

# Project-authored lexical normalization. Only spelling/true synonyms merge;
# different fruits, roasted/raw variants and source co-labels stay distinct.
RENAMES = {
    'pinapple': 'pineapple', 'jasmin': 'jasmine', 'camomile': 'chamomile',
    'caramelic': 'caramel', 'caramellic': 'caramel', 'cacao': 'cocoa',
    'cedarwood': 'cedar', 'musky': 'musk', 'musklike': 'musk',
    'muguet': 'lilyofthevalley', 'earth': 'earthy', 'flower': 'floral',
    'flowery': 'floral', 'floralaroma': 'floral', 'fruityaroma': 'fruity',
    'fruit': 'fruity', 'wood': 'woody', 'nut': 'nutty', 'nuttyaroma': 'nutty',
    'fat': 'fatty', 'fattyaroma': 'fatty', 'butter': 'buttery',
    'minty': 'mint', 'mintyaroma': 'mint', 'leaf': 'leafy', 'leafyaroma': 'leafy',
    'herb': 'herbal', 'herbaceous': 'herbal', 'herbaceousaroma': 'herbal',
    'camphoraceous': 'camphor', 'camphoraceousaroma': 'camphor', 'camphoreous': 'camphor',
    'cloves': 'clove', 'spice': 'spicy', 'spicyaroma': 'spicy', 'anisic': 'anise',
    'alcoholic': 'alcohol', 'animallike': 'animal', 'cheese': 'cheesy', 'cream': 'creamy',
    'fish': 'fishy', 'fishyaroma': 'fishy', 'grass': 'grassy', 'meat': 'meaty',
    'medicine': 'medicinal', 'medical': 'medicinal', 'metal': 'metallic',
    'moss': 'mossy', 'oil': 'oily', 'resin': 'resinous', 'rubber': 'rubbery',
    'smoke': 'smoky', 'soap': 'soapy', 'sulfur': 'sulfurous', 'sulfury': 'sulfurous',
    'sweat': 'sweaty', 'sweetaroma': 'sweet', 'wax': 'waxy', 'rummy': 'rum', 'winey': 'wine',
    'roast': 'roasted', 'roastedaroma': 'roasted', 'tropicalfruit': 'tropical',
    'fruittropicalfruit': 'tropical', 'ammoniacal': 'ammonia', 'sassafrass': 'sassafras',
    'potatobakedpotato': 'bakedpotato', 'breadbaked': 'bakedbread',
    'currantblackcurrant': 'blackcurrant', 'sugarburntsugar': 'burntsugar',
    'roselikearoma': 'rose', 'tobaccolike': 'tobacco', 'coumarinic': 'coumarin',
    'ozonic': 'ozone', 'terpenic': 'terpene', 'mentholic': 'menthol',
}
DISPLAY = {
    'applegreenapple':'green apple', 'applepeel':'apple peel', 'appleskin':'apple skin',
    'almondroastedalmond':'roasted almond', 'almondbitteralmond':'bitter almond',
    'bananaunripebanana':'unripe banana', 'bananapeel':'banana peel',
    'barleyroastedbarley':'roasted barley', 'beangreenbean':'green bean',
    'beefyroastedbeefy':'roasted beef', 'breadbaked':'baked bread',
    'cherrymaraschinocherry':'maraschino cherry', 'clothlaunderedcloth':'laundered cloth',
    'coffeeroastedcoffee':'roasted coffee', 'currantbudblackcurrantbud':'blackcurrant bud',
    'fruitdriedfruit':'dried fruit', 'fruitoverripefruit':'overripe fruit',
    'graintoastedgrain':'toasted grain', 'grapefruitpeel':'grapefruit peel',
    'grapeskin':'grape skin', 'haynewmownhay':'new mown hay',
    'hazelnutroastedhazelnut':'roasted hazelnut', 'licoriceblacklicorice':'black licorice',
    'meatyroastedmeaty':'roasted meat', 'melonrind':'melon rind', 'nutskin':'nut skin',
    'onioncookedonion':'cooked onion', 'oniongreenonion':'green onion',
    'peagreenpea':'green pea', 'peanutroastedpeanut':'roasted peanut',
    'pepperbellpepper':'bell pepper', 'pearskin':'pear skin',
    'potatolike':'potato like', 'potatorawpotato':'raw potato',
    'roseredrose':'red rose', 'rosetearose':'tea rose', 'sageclarysage':'clary sage',
    'sugarbrownsugar':'brown sugar', 'teagreentea':'green tea', 'woodyoldwood':'old wood',
    'sweetonionyaroma':'sweet onion aroma', 'fruityfloral':'fruity floral',
    'fruityrosy':'fruity rose', 'honeyfloral':'honey floral', 'freshoutdoors':'fresh outdoors',
    'citruspeel':'citrus peel', 'citrusrind':'citrus rind', 'cortex':'bark',
}
KO_TEXT = '''
acacia=아카시아;acorn=도토리;aldehydic=알데하이딕,알데하이드;algae=해조;allspice=올스파이스;almond=아몬드
amber=앰버;ambergris=앰버그리스,용연향;ambrette=앰브레트;ammonia=암모니아;anise=아니스;apple=사과
green apple=청사과,풋사과;apple peel=사과 껍질;apple skin=사과 껍질향;apricot=살구;asparagus=아스파라거스
bacon=베이컨;balsam=발삼;banana=바나나;unripe banana=덜 익은 바나나;banana peel=바나나 껍질;basil=바질
beeswax=밀랍;benzoin=벤조인,안식향;bergamot=베르가못;berry=베리;blackberry=블랙베리;blackcurrant=블랙커런트,카시스
blackcurrant bud=블랙커런트 새싹;blueberry=블루베리;bois de rose=로즈우드;brandy=브랜디;bread=빵;baked bread=갓 구운 빵
broccoli=브로콜리;bubble gum=풍선껌;burnt=탄 냄새;burnt sugar=태운 설탕;buttermilk=버터밀크;butterscotch=버터스카치
buttery=버터;cabbage=양배추;camphor=장뇌;cananga=카낭가;caramel=캐러멜,카라멜;caraway=캐러웨이;cardamom=카다멈
carnation=카네이션;carrot=당근;cashew=캐슈넛;cassia=카시아;castoreum=캐스토리움;cedar=시더우드,삼나무;celery=셀러리
cereal=곡물;chamomile=카모마일,캐모마일;cheesy=치즈;cherry=체리;maraschino cherry=마라스키노 체리
chicken=닭고기;chlorine=염소 냄새;chocolate=초콜릿;chrysanthemum=국화;cider=사이더;cilantro=고수 잎
cinnamon=시나몬,계피;citronella=시트로넬라;citrus=시트러스,감귤;citrus peel=감귤 껍질;clove=정향;clover=클로버
cocoa=코코아;coconut=코코넛;coffee=커피;roasted coffee=볶은 커피,로스팅 커피;cognac=코냑;cookie=쿠키
coriander=코리앤더;corn=옥수수;cotton candy=솜사탕;cranberry=크랜베리;creamy=크리미; cucumber=오이;cumin=커민
custard=커스터드;cyclamen=시클라멘;cypress=사이프러스,편백;dewy=이슬 맺힌;dill=딜;durian=두리안;dusty=먼지
earthy=흙내음;elderflower=엘더플라워;elemi=엘레미;eucalyptus=유칼립투스;fennel=펜넬,회향;fenugreek=페뉴그릭
fig=무화과;fir needle=전나무 잎;fishy=비린;forest=숲;frankincense=프랑킨센스,유향;freesia=프리지아
galbanum=갈바넘;gardenia=치자,가드니아;garlic=마늘;geranium=제라늄;ginger=생강;gooseberry=구스베리
grape=포도;grapefruit=자몽;grapefruit peel=자몽 껍질;grape skin=포도 껍질;grassy=풀;green bean=풋콩
green tea=녹차;guaiacwood=과이악우드;guava=구아바;hawthorn=산사나무;hay=건초;new mown hay=갓 벤 건초
hazelnut=헤이즐넛;roasted hazelnut=볶은 헤이즐넛;heliotrope=헬리오트로프;honey=꿀;honeydew=허니듀
honeysuckle=인동덩굴,허니서클;horseradish=서양고추냉이;hyacinth=히아신스;immortelle=이모르텔;incense=인센스,향불
iris=아이리스;jam=잼;jasmine=자스민,쟈스민;juicy=즙이 많은;juniper=주니퍼;kimchi=김치;kiwi=키위
labdanum=랩다넘,라브다넘;lavender=라벤더;leather=가죽;leathery=레더;leek=리크;lemon=레몬
lemon peel=레몬 껍질;lemongrass=레몬그라스;lettuce=상추;licorice=감초;lilac=라일락;lily=백합
lily of the valley=은방울꽃,뮤게;lime=라임;linden flower=린덴 블로섬,보리수꽃;lovage=러비지;lychee=리치
magnolia=목련,매그놀리아;malt=맥아;mandarin=만다린;mango=망고;maple=단풍 시럽;maple syrup=메이플 시럽
marigold=메리골드;marine=해양;marjoram=마조람;marshmallow=마시멜로;marzipan=마지팬;melon=멜론
melon rind=멜론 껍질;menthol=멘톨;metallic=금속;mimosa=미모사;mint=민트;molasses=당밀;mossy=이끼
mushroom=버섯;musk=머스크;mustard=겨자;musty=곰팡내;myrrh=미르,몰약;narcissus=수선화;neroli=네롤리
nutmeg=넛맥,육두구;nutty=견과류;oak=참나무;oakmoss=오크모스;ocean=바다;onion=양파;cooked onion=익힌 양파
opoponax=오포포낙스;orange=오렌지;orange flower=오렌지 블로섬;orange peel=오렌지 껍질;orchid=난초
orris=오리스,붓꽃 뿌리;osmanthus=오스만투스,금목서;ozone=오존;papaya=파파야;paper=종이;parsley=파슬리
passion fruit=패션프루트;patchouli=파촐리,패출리;peach=복숭아;peanut=땅콩;peanut butter=땅콩버터
roasted peanut=볶은 땅콩;pear=배 향,서양배;pear skin=배 껍질;peony=피오니,작약;pepper=후추;black pepper=흑후추
peppermint=페퍼민트;petal=꽃잎;petitgrain=페티그레인;pine=소나무;pineapple=파인애플;pistachio=피스타치오
plum=자두;pomegranate=석류;popcorn=팝콘;potato=감자;raw potato=생감자;praline=프랄린;prune=말린 자두
quince=모과;radish=무 냄새;raisin=건포도;raspberry=라즈베리;resinous=수지;reseda=레세다;rhubarb=루바브
roasted almond=볶은 아몬드;bitter almond=쓴 아몬드;root beer=루트비어;rose=장미;red rose=붉은 장미
tea rose=티로즈;rosemary=로즈마리;rubbery=고무;rum=럼주;saffron=사프란;sage=세이지;clary sage=클라리세이지
sandalwood=샌달우드,백단향;sassafras=사사프라스;sawdust=톱밥;seafood=해산물;seashore=해변;seaweed=해초
shellfish=조개;shrimp=새우;smoky=스모키;soapy=비누;soil=흙;soy=간장;spearmint=스피어민트
spruce=가문비나무;starfruit=스타프루트;storax=스토락스;straw=볏짚;strawberry=딸기;sugar=설탕;brown sugar=흑설탕
sulfurous=유황;sweaty=땀 냄새;sweet pea=스위트피;tagette=타제트;tangerine=탠저린;tarragon=타라곤
tea=차 향;thyme=타임;tobacco=담배;toffee=토피;tomato=토마토;tomato leaf=토마토 잎;tonka=통카
truffle=트러플;tuberose=튜베로즈;turpentine=테레빈;valerian root=발레리안 뿌리;vanilla=바닐라
verbena=버베나;vetiver=베티버;vinegar=식초;violet=바이올렛,제비꽃;violet leaf=바이올렛 리프,제비꽃 잎
walnut=호두;wasabi=와사비;watercress=물냉이;watermelon=수박;waxy=왁시;wet paper=젖은 종이;whiskey=위스키
white flowers=흰꽃;wintergreen=윈터그린;woody=우디;old wood=오래된 나무;wormwood=쑥;yeasty=효모
ylang=일랑일랑;yogurt=요구르트
'''


def norm(text):
    return ''.join(c for c in text.casefold() if c.isalnum())


def canonical(text):
    key = norm(text)
    if key in DISPLAY:
        key = norm(DISPLAY[key])
    return RENAMES.get(key, key)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def fetch(folder):
    folder.mkdir(parents=True, exist_ok=True)
    def get(url):
        with urlopen(Request(url, headers={'User-Agent': 'PerfumeryAI-source-audit'}), timeout=30) as response:
            return response.read()
    commit = json.loads(get('https://api.github.com/repos/akshay-sajann/computer_ontology/commits/main'))['sha']
    files = ['LICENSE', 'create_taxonomy_rdf/expert_taxonomy.ttl',
             'create_taxonomy_rdf/expert_taxonomy-quality.ttl']
    manifest = {'repository': REPO, 'commit': commit, 'files': {}}
    for name in files:
        url = f'https://raw.githubusercontent.com/akshay-sajann/computer_ontology/{commit}/{name}'
        raw = get(url)
        (folder/Path(name).name).write_bytes(raw)
        manifest['files'][Path(name).name] = {'url': url, 'sha256': sha(raw), 'bytes': len(raw)}
    write(folder/'sources.json', manifest)


def turtle_concepts(raw):
    """Strict reader for this pinned SKOS export (not a general RDF parser)."""
    records = re.findall(r'^<([^>]+)>\s+a\s+skos:Concept\s*;(.*?)(?=^<|\Z)', raw, re.M | re.S)
    output = []
    for uri, body in records:
        labels = {}
        for role in ('prefLabel', 'altLabel'):
            part = re.search(r'skos:'+role+r'\s+(.*?)(?:;|\s\.\s*$)', body, re.S)
            labels[role] = [json.loads('"'+s+'"') for s in re.findall(r'"((?:\\.|[^"\\])*)"@en', part.group(1))] if part else []
        if len(labels['prefLabel']) != 1:
            raise ValueError('unsupported SKOS preferred label syntax: '+uri)
        output.append({'uri': uri, 'label': labels['prefLabel'][0], 'aliases': labels['altLabel'],
                       'parents': re.findall(r'skos:broader\s+<([^>]+)>', body)})
    if len(records) != raw.count('a skos:Concept ;'):
        raise ValueError('SKOS concept parser omitted a record')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, default=ROOT/'.benchmarks/odor_expression_v61/sources')
    parser.add_argument('--fetch', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.fetch:
        fetch(args.sources)
    manifest = json.loads((args.sources/'sources.json').read_text(encoding='utf-8'))
    for name, ref in manifest['files'].items():
        if sha((args.sources/name).read_bytes()) != ref['sha256']:
            raise ValueError('source content drift: '+name)
    records = []
    for name in ('expert_taxonomy.ttl', 'expert_taxonomy-quality.ttl'):
        parsed = turtle_concepts((args.sources/name).read_text(encoding='utf-8'))
        for row in parsed:
            row['kind'] = 'quality' if 'quality' in name else 'odor'
        records.extend(parsed)
    write(args.sources/'parsed.json', records)
    print(json.dumps({'records': len(records), 'kinds': dict(Counter(r['kind'] for r in records)),
        'families': sorted({r['uri'].split('expert-taxonomy/')[1].split('/')[0] for r in records})}))
    if not args.audit_only:
        build(records, manifest, args.sources)


def build(records, manifest, folder):
    from fragrance_ai.recommender.brief_parser import KEYWORDS
    from fragrance_ai.recommender.odor_descriptors import load_builtin_odor_descriptor_lexicon
    atlas_path = ROOT/'.benchmarks/quantitative_profiles_v54/run-01/model.json'
    parent_raw = atlas_path.read_bytes()
    fine = json.loads(parent_raw)['fine_features']
    concepts = {}
    for r in records:
        key = canonical(r['label'])
        row = concepts.setdefault(key, {'id': key, 'label_en': r['label'].casefold(), 'aliases': set(),
            'kind': r['kind'], 'hierarchy_paths': [], 'source_terms': [], 'source_uris': []})
        row['aliases'].update([r['label'].casefold(), *[s.casefold() for s in r['aliases']]])
        row['source_uris'].append(r['uri'])
        row['hierarchy_paths'].append(r['uri'].split('expert-taxonomy/')[1].split('/')[:-1])
    # Do not adopt broad or misleading upstream alternative labels as identity
    # equivalences (amber != ambergris; beer != root beer). Exact source labels
    # win; remaining multi-concept aliases are disabled and reported below.
    index = defaultdict(set)
    for key, row in concepts.items():
        for alias in row['aliases']:
            index[norm(alias)].add(key)
    term_to_concept = {}
    for term in fine['vocabulary']:
        label = DISPLAY.get(term, term)
        key = canonical(label)
        if key not in concepts:
            # Only a one-to-one expert alias is used when no exact identity
            # exists. Source terms remain distinct when ambiguous.
            choices = index.get(norm(term), set())
            if len(choices) == 1 and term not in DISPLAY and term not in {'ambergris', 'rootbeer'}:
                key = next(iter(choices))
        row = concepts.setdefault(key, {'id': key, 'label_en': label, 'aliases': set(),
            'kind': 'odor', 'hierarchy_paths': [], 'source_terms': [], 'source_uris': []})
        row['aliases'].update([label, term])
        if not row['hierarchy_paths'] and term in DISPLAY and len(index.get(norm(term), set())) == 1:
            related = concepts[next(iter(index[norm(term)]))]
            row['hierarchy_paths'] = list(related['hierarchy_paths'])
        row['source_terms'].append(term)
        term_to_concept[term] = key
    # Quality/strength words are retained for interpretation, never advertised
    # as additional independent odor identities.
    generic = set('agreeable almostodorless aromatic characteristic deep delicate disagreeable distinctive faint fragrant heated light low mild natural neutral nuance offensive peculiar pleasant powerful raw reminiscent repulsive solution strong unpleasant volatile wild foodlike hairy brown'.split())
    for key in generic:
        if key in concepts:
            concepts[key]['kind'] = 'quality'
    korean_unmapped = []
    for entry in re.split(r'[;\n]', KO_TEXT):
        if '=' not in entry:
            continue
        english, values = (x.strip() for x in entry.split('=', 1))
        key = canonical(english)
        if key not in concepts:
            choices = index.get(norm(english), set())
            key = next(iter(choices)) if len(choices) == 1 else key
        if key not in concepts:
            korean_unmapped.append(english)
            continue
        concepts[key]['aliases'].update(v.strip() for v in values.split(','))
    # Coarse bridge is an engineering language projection only. Existing
    # material profiles and the V54 checkpoint are not edited/relabelled.
    coarse = defaultdict(dict)
    for axis, words in KEYWORDS.items():
        for word in words:
            coarse[norm(word)][axis] = 1.
    for row in load_builtin_odor_descriptor_lexicon().descriptors:
        if row.formula_supported:
            for word in row.aliases:
                coarse[norm(word)] = row.profile
    families = {'flower':'floral', 'fruity':'fruity', 'gourmand':'gourmand', 'green':'green',
        'herbal':'aromatic', 'woody':'woody', 'spices':'spicy', 'earthy':'earthy',
        'balsamic':'amber', 'smoky':'smoky', 'aquatic':'aquatic'}
    counts = Counter()
    for terms in fine['by_structure'].values():
        counts.update({term_to_concept[fine['vocabulary'][i]] for i in terms})
    for key, row in concepts.items():
        projected = {}
        for alias in sorted(row['aliases']):
            if norm(alias) in coarse:
                projected = coarse[norm(alias)]
                break
        if not projected and row['kind'] == 'odor':
            paths = row['hierarchy_paths']
            axes = {families[p[0]] for p in paths if p and p[0] in families}
            if any('citrus' in p for p in paths):
                axes = {'citrus'}
            projected = {a: 1/len(axes) for a in sorted(axes)}
        if not projected and row['kind'] == 'odor':
            # Compositional English labels keep their identity, while this
            # routing-only bridge inherits a named component's coarse family.
            matches = [coarse[norm(word)] for word in coarse if len(word) >= 4
                       and word in norm(row['label_en'])]
            if matches:
                projected = dict(Counter({}))
                for match in matches:
                    for axis, value in match.items():
                        projected[axis] = projected.get(axis, 0.)+value
        # Blood/fish/food/sulfur/chemical source labels are not automatically
        # reinterpreted as generic pleasant perfumery notes.
        total = sum(projected.values())
        row['coarse_projection'] = {a:v/total for a,v in projected.items()} if total else {}
        row['projection_basis'] = 'project_authored_language_bridge_not_measured_intensity'
        row['structure_annotation_support'] = counts[key]
        row['aliases'] = sorted(row['aliases'])
        row['hierarchy_paths'] = sorted({tuple(p) for p in row['hierarchy_paths']})
        row['source_uris'] = sorted(set(row['source_uris']))
        if not row['source_terms'] and row['source_uris'] and all(
                any(uri in other['parents'] for other in records) for uri in row['source_uris']):
            row['kind'] = 'family'
        if key in DISPLAY:
            row['label_en'] = DISPLAY[key]
    # Prefer a true canonical label over a broader group's alias; drop all
    # other ambiguity instead of changing meaning with dictionary order.
    alias_owners = defaultdict(set)
    for key, row in concepts.items():
        for alias in row['aliases']:
            alias_owners[alias].add(key)
    conflicts = {}
    for alias, owners in alias_owners.items():
        if len(owners) > 1:
            preferred = canonical(alias)
            keep = preferred if preferred in owners else None
            conflicts[alias] = {'candidates': sorted(owners), 'selected': keep}
            for key in owners:
                if key != keep:
                    concepts[key]['aliases'].remove(alias)
    payload = {'schema':'odor-expression-registry/v1', 'version':'odor-expression-v61',
        'source': manifest, 'upstream_license': (folder/'LICENSE').read_text(encoding='utf-8'),
        'attribution':'Expert taxonomy: Akshay Sajan et al.; Odeuropa; project-authored Korean aliases and normalization.',
        'source_annotation_parent_sha256': sha(parent_raw),
        'source_term_to_concept': term_to_concept, 'concepts': sorted(concepts.values(), key=lambda r:r['id']),
        'alias_conflicts':conflicts, 'korean_unmapped':korean_unmapped,
        'claim_boundary':'Vocabulary and public annotation support, not new measured odor-intensity axes or human similarity.',
        'quantitative_axes_changed':False}
    target = ROOT/'fragrance_ai/data/odor_expression_v61.json'
    write(target, payload)
    audit = {'canonical_concepts':len(concepts), 'kinds':dict(Counter(r['kind'] for r in concepts.values())),
        'aliases':sum(len(r['aliases']) for r in concepts.values()),
        'korean_aliases':sum(any('가'<=c<='힣' for c in a) for r in concepts.values() for a in r['aliases']),
        'molecularly_annotated_concepts':sum(counts[k]>0 for k in concepts),
        'fine_source_structures':len(fine['by_structure']), 'korean_unmapped':korean_unmapped,
        'registry_sha256':sha(target.read_bytes())}
    write(folder.parent/'registry-build.json', audit)
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__':
    main()
