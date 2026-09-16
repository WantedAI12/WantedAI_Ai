import pytest
from fragrance_ai.recommender.compact_language import AssistantRequest, assistant_reply
from fragrance_ai.recommender.brief_parser import NaturalLanguageBriefParser
from fragrance_ai.recommender.catalog import IngredientCatalog


@pytest.mark.parametrize('text', ['머스크 없이 장미 향 바디로션으로 만들어줘', '우디 향수', 'rose perfume without musk'])
def test_complete_literals_skip_llm_and_preserve_parser_intent(text):
    parser = NaturalLanguageBriefParser(IngredientCatalog.load_builtin())
    expected = parser.parse(text)
    result = assistant_reply(AssistantRequest(message=text), parser,
        lambda _: pytest.fail('known literal intent must not start a language worker'), prefer_literal=True)
    assert result['source'] == 'deterministic_complete_literal_intent'
    assert result['intent_proposal']['desired'] == expected.desired_dimensions
    assert result['intent_proposal']['avoided'] == expected.avoided_dimensions
    assert not result['formula_generated'] and result['requires_confirmation']


@pytest.mark.parametrize('text', ['우디 향수 20%', '우디 향수지만 만화책 같은 질감', '로션 말고 우디 향수'])
def test_unexplained_content_does_not_take_literal_fast_path(text):
    calls = []
    def backend(message):
        calls.append(message)
        return {'desired': ['woody'], 'avoided': [], 'product': 'perfume', 'clarification': 'none'}
    assistant_reply(AssistantRequest(message=text), NaturalLanguageBriefParser(IngredientCatalog.load_builtin()), backend, prefer_literal=True)
    assert calls == [text]
