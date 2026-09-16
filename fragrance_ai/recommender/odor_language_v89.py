"""An open, compositional scent lexicon with explicit literary interpretations.

Scenes are authored interpretations, not additional measured odor dimensions.
Every atom resolves through the existing source registry; unknown detail and
explicit exclusions remain visible. No recipe or benchmark outcome is used.
"""

import hashlib
import json

from .catalog import find_text_spans

VERSION = 'compositional-odor-language/v89'

# The order is descriptive, not a recipe ratio or measured intensity claim.
SCENES = (
    ('rain_forest','비가 그친 숲|비에 젖은 숲|비 온 뒤 숲|rain soaked forest|forest after rain','earthy green woody'),
    ('moss_stone','이끼 낀 돌|젖은 돌담|moss covered stone|wet stone wall','earthy green woody'),
    ('morning_garden','새벽 정원|이슬 맺힌 정원|dawn garden|dewy garden','floral green fresh'),
    ('moon_garden','달빛 아래 정원|달빛 정원|moonlit garden|garden in moonlight','jasmine green musky'),
    ('pine_cabin','소나무 숲 오두막|숲속 오두막|pine cabin|forest cabin','woody pine resinous'),
    ('fallen_leaves','낙엽 쌓인 숲길|가을 숲길|fallen leaf path|autumn forest','earthy woody green'),
    ('mountain_air','산속의 맑은 공기|고산의 바람|mountain air|alpine breeze','fresh green pine'),
    ('spring_meadow','봄날의 초원|봄 초원|spring meadow|spring field','green floral fresh'),
    ('summer_grove','한여름의 과수원|여름 과수원|summer orchard','fruity green citrus'),
    ('winter_woods','겨울 숲|눈 덮인 숲|winter woods|snow covered forest','woody pine fresh'),
    ('bamboo_grove','대나무 숲|바람 부는 대숲|bamboo grove','green woody fresh'),
    ('tea_garden','차밭의 아침|아침 차밭|morning tea garden','tea green fresh'),
    ('orchard_blossom','과수원의 꽃|꽃 핀 과수원|orchard blossom','floral fruity green'),
    ('rose_greenhouse','장미 온실|온실 속 장미|rose greenhouse','rose green fresh'),
    ('night_blossom','밤에 피는 꽃|밤의 꽃 정원|night blooming garden','jasmine floral musky'),
    ('coastal_wind','해안의 바람|바닷가의 바람|coastal breeze|coastal wind','aquatic fresh green'),
    ('sea_foam','파도가 남긴 거품|하얀 파도|sea foam|white surf','aquatic fresh musky'),
    ('driftwood','해변의 유목|바닷가 나무|beach driftwood','aquatic woody'),
    ('harbour_morning','항구의 새벽|새벽 항구|harbour at dawn','aquatic woody fresh'),
    ('lakeside','호숫가의 아침|아침 호숫가|morning lakeside','aquatic green fresh'),
    ('riverbank','강가의 풀밭|강변 산책|riverbank meadow','aquatic green earthy'),
    ('rain_window','비 내리는 창가|빗물이 흐르는 창문|rainy window','aquatic fresh green'),
    ('mist_valley','안개 낀 계곡|안개 속 계곡|misty valley','aquatic woody green'),
    ('water_lily','연못의 꽃|꽃 핀 연못|flowering pond','aquatic floral green'),
    ('sun_linen','햇살에 말린 린넨|햇볕에 말린 이불|sun dried linen|sun dried sheets','clean musky powdery'),
    ('fresh_laundry','갓 세탁한 셔츠|깨끗한 셔츠|freshly washed shirt','clean musky fresh'),
    ('cotton_cloud','솜구름 같은 향|포근한 솜구름|cotton cloud','musky powdery clean'),
    ('white_towel','보송한 흰 수건|새하얀 수건|fluffy white towel','clean powdery musky'),
    ('soap_bubbles','목욕 후 비눗방울|하얀 비눗방울|white soap bubbles','clean floral musky'),
    ('warm_skin','따뜻한 살결|포근한 살내음|warm skin|skin warmth','musky powdery'),
    ('cashmere','캐시미어 스웨터|포근한 스웨터|cashmere sweater','musky woody powdery'),
    ('quiet_room','고요한 다다미방|나무로 된 고요한 방|quiet wooden room','woody green musky'),
    ('old_library','오래된 도서관|낡은 도서관|old library','woody leathery powdery'),
    ('old_book','오래된 책장|낡은 책장|old bookshelf','woody powdery leathery'),
    ('leather_study','가죽 의자가 있는 서재|가죽 서재|leather study','leathery woody smoky'),
    ('ink_paper','잉크와 종이|종이 위의 잉크|ink and paper','woody powdery smoky'),
    ('wood_workshop','나무 공방|목공 작업실|wood workshop','woody resinous'),
    ('temple_courtyard','절의 마당|고요한 사찰|temple courtyard','incense woody earthy'),
    ('fireplace','벽난로 앞|겨울의 벽난로|winter fireplace','smoky woody resinous'),
    ('campfire','숲속 모닥불|모닥불이 꺼진 자리|forest campfire','smoky woody earthy'),
    ('tea_steam','따뜻한 찻김|찻잔에서 피어나는 김|tea steam','tea aromatic fresh'),
    ('citrus_tea','귤껍질을 띄운 차|찻잔의 감귤 껍질|citrus peel tea','tea citrus aromatic'),
    ('tea_ceremony','다실의 향|고요한 다실|tea ceremony room','tea woody green'),
    ('coffee_shop','이른 아침 카페|문을 연 카페|early morning cafe','coffee gourmand woody'),
    ('bakery','갓 문을 연 빵집|아침의 빵집|morning bakery','gourmand vanilla buttery'),
    ('spice_market','향신료 시장|골목의 향신료 가게|spice market','spicy woody aromatic'),
    ('citrus_peel','손끝에 남은 귤껍질|막 벗긴 감귤 껍질|freshly peeled citrus','citrus green fresh'),
    ('fruit_basket','잘 익은 과일 바구니|여름 과일 바구니|ripe fruit basket','fruity citrus'),
    ('berry_jam','끓이고 있는 베리 잼|따뜻한 베리 잼|warm berry jam','fruity sweet gourmand'),
    ('vanilla_wood','바닐라가 스민 나무|바닐라 나무 상자|vanilla wood box','vanilla woody musky'),
    ('honey_tea','꿀을 넣은 따뜻한 차|꿀 찻잔|honey tea','honey tea floral'),
    ('cocoa_evening','따뜻한 코코아 한 잔|코코아가 있는 저녁|warm cocoa evening','cocoa vanilla gourmand'),
    ('mediterranean','지중해의 정원|지중해 산책|mediterranean garden','citrus aromatic woody'),
    ('lavender_field','라벤더 밭의 바람|바람 부는 라벤더 밭|lavender field breeze','lavender aromatic fresh'),
    ('orange_grove','오렌지 나무 그늘|오렌지 숲의 그늘|orange grove shade','orange green woody'),
    ('herb_kitchen','허브가 놓인 주방|싱싱한 허브 주방|fresh herb kitchen','aromatic green citrus'),
    ('sunset_garden','노을 진 정원|해 질 녘 정원|sunset garden','floral woody musky'),
    ('rain_city','비 내린 도시|비가 그친 골목|city after rain','aquatic woody earthy'),
    ('rooftop_garden','옥상 정원의 바람|도시의 작은 정원|rooftop garden','green floral fresh'),
    ('botanical_house','식물로 가득한 집|초록 식물이 있는 방|botanical house','green woody floral'),
)

FACETS = {
    'airiness':('공기처럼 가벼운','공기 같은','airy','air-like'),
    'transparency':('투명한','맑게 비치는','transparent','crystalline'),
    'softness':('부드러운','실크처럼 매끄러운','soft','silky'),
    'dryness':('메마른','건조한','dry','parched'),
    'dewiness':('이슬 맺힌','촉촉한','dewy','moist'),
    'warmth':('온기 있는','따뜻하게 감싸는','warm','warming'),
    'coolness':('서늘한','차갑게 스치는','cool','chilly'),
    'brightness':('빛나는','햇살 같은','bright','sunlit'),
    'darkness':('어두운','그늘진','dark','shadowed'),
    'velvet':('벨벳 같은','벨벳처럼 포근한','velvety','velvet-like'),
    'powder_texture':('분가루처럼 보송한','보송보송한','powder-soft','downy'),
    'cream_texture':('크림처럼 둥근','우유처럼 부드러운','creamy','milky-soft'),
    'roughness':('거친','결이 살아있는','rough','textured'),
    'sparkle':('반짝이는','톡 쏘듯 반짝이는','sparkling','effervescent'),
    'depth':('깊이 있는','깊숙이 남는','deep','resonant'),
    'distance':('멀리서 스치는','아득하게 퍼지는','distant','faraway'),
    'intimacy':('가까이 머무는','살결에 가까운','intimate','close-to-skin'),
    'diffusion':('넓게 퍼지는','공간을 채우는','diffusive','room-filling'),
    'restraint':('절제된','조용히 머무는','restrained','understated'),
    'density':('농밀한','짙게 감싸는','dense','rich'),
    'freshness':('싱그러운','막 피어난','fresh','newly-bloomed'),
    'maturity':('잘 익은','시간이 묻어나는','ripe','aged'),
    'stillness':('고요한','잔잔하게 가라앉은','quiet','still'),
    'movement':('바람에 실린','물결처럼 흐르는','breezy','flowing'),
}


def resolve_atoms(words, registry):
    result=[]
    for word in words.split():
        key=registry['aliases'].get(word,word)
        if key not in registry['rows']:
            return None
        result.append(key)
    return tuple(dict.fromkeys(result))


def interpret_scenes(text, registry, negated):
    matches=[]
    occupied=[]
    proposals=[]
    for key,aliases,words in SCENES:
        atoms=resolve_atoms(words,registry)
        if atoms is None:
            continue
        for alias in aliases.split('|'):
            for start,end in find_text_spans(text.casefold(),alias.casefold()):
                proposals.append((start,end,key,atoms))
    for start,end,key,atoms in sorted(proposals,key=lambda p:(-(p[1]-p[0]),p[0],p[2])):
        if any(a<end and start<b for a,b in occupied):
            continue
        occupied.append((start,end))
        matches.append({'scene_id':key,'text':text[start:end],'start':start,'end':end,
                        'atoms':atoms,'polarity':'avoid' if negated(start,end) else 'want',
                        'basis':'project_authored_compositional_interpretation_not_measurement'})
    return sorted(matches,key=lambda m:m['start'])


def facets(text,negated=None):
    return [{'facet':key,'text':text[a:b],'start':a,'end':b,
             'polarity':'avoid' if negated is not None and negated(a,b) else 'want',
             'basis':'descriptive_style_not_measured_physical_coefficient'}
            for key,aliases in FACETS.items() for alias in aliases
            for a,b in find_text_spans(text.casefold(),alias.casefold())]


def contract(registry):
    valid=[row for row in SCENES if resolve_atoms(row[2],registry)]
    raw=json.dumps([SCENES,FACETS],ensure_ascii=False,sort_keys=True).encode()
    return {'version':VERSION,'definition_sha256':hashlib.sha256(raw).hexdigest(),
            'source_concepts':len(registry['rows']),'source_aliases':len(registry['aliases']),
            'authored_scenes':len(valid),'scene_phrasings':sum(len(row[1].split('|')) for row in valid),
            'style_facets':len(FACETS),'productive_style_phrasings':sum(map(len,FACETS.values())),
            'composition_grammar':'optional_style_facets + one_or_more_source_odors_or_scenes + phase_and_exclusion_clauses',
            'independent_measured_axes_added':0,'poetic_interpretations_are_explicit':True,
            'whole_dictionary_quantitatively_verified':False}


def dictionary(registry, query='', offset=0, limit=50):
    term=query.strip().casefold()
    items=[]
    for key,row in registry['rows'].items():
        aliases=row.get('aliases',())
        if term and not any(term in str(v).casefold() for v in (key,row.get('label_en',''),*aliases)):
            continue
        items.append({'concept_id':key,'label':row.get('label_en',key),'aliases':list(aliases),
            'families':row.get('hierarchy_paths',[]),'coarse_facets':row.get('coarse_projection',{}),
            'quantitative_status':row.get('quantitative_status','not_declared'),
            'reference_key':row.get('reference_key'),'evidence_kind':row.get('resolution',{}).get('reference_evidence_kind'),
            'source_uris':row.get('source_uris',[])})
    return {'version':VERSION,'total':len(items),'offset':offset,'items':items[offset:offset+limit],
            'next_offset':offset+limit if offset+limit<len(items) else None}
