from scripts.build_odor_integrity_catalog import resolve_cross_source_identity
from fragrance_ai.recommender.odor_integrity import assess_odor_assertions


def test_explicit_valid_cas_unique_cid_fallback():
    assert resolve_cross_source_identity('goodscents','101-86-0',{'101-86-0':1550884},{},{'1550884':{'id'}}) == ('id','1550884',True)


def test_ambiguous_or_unmapped_identity_is_not_guessed():
    for links in ({}, {'1550884':{'id1','id2'}}):
        assert resolve_cross_source_identity('goodscents','101-86-0',{'101-86-0':1550884},{},links)[0] is None
    assert resolve_cross_source_identity('goodscents','1550884',{}, {}, {'1550884':{'id'}})[0] is None


def test_invalid_cas_and_non_numeric_cid_cannot_fallback():
    assert resolve_cross_source_identity('goodscents','101-86-1',{'101-86-1':1550884},{},{'1550884':{'id'}})[0] is None
    assert resolve_cross_source_identity('goodscents','101-86-0',{'101-86-0':True},{},{'True':{'id'}})[0] is None


def test_direct_identity_is_preserved():
    assert resolve_cross_source_identity('goodscents','101-86-0',{'101-86-0':1550884}, {('goodscents','1550884'):'direct'}, {'1550884':{'id'}}) == ('direct','1550884',False)


def test_negative_and_unsupported_evidence_are_not_discarded_in_join():
    assert assess_odor_assertions(('goodscents:rose','goodscents:odorless'))[0] == 'conflicting_odor_reports'
    assert assess_odor_assertions(('goodscents:rose','goodscents:sulfurous'))[0] == 'unmodeled_odor_descriptors'
