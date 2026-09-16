import pytest
from scripts.verify_junit_floor import verify


def test_header_count_cannot_fake_executed_tests(tmp_path):
    path = tmp_path/'tests.xml'
    path.write_text('<testsuite tests="1000"><testcase classname="a" name="one"/></testsuite>')
    with pytest.raises(ValueError, match='incomplete'):
        verify([path], 220)
    assert verify([path], 1)['passed'] == 1


def test_duplicate_or_failed_or_skipped_test_cannot_count_as_passed(tmp_path):
    path = tmp_path/'tests.xml'
    path.write_text('<testsuite><testcase classname="a" name="one"/></testsuite>')
    with pytest.raises(ValueError, match='duplicate'):
        verify([path, path], 1)
    for state in ('failure', 'error', 'skipped'):
        path.write_text(f'<testsuite><testcase classname="a" name="one"><{state}/></testcase></testsuite>')
        with pytest.raises(ValueError, match='incomplete'):
            verify([path], 1)
