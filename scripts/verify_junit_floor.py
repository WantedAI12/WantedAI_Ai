"""Reject incomplete, failed or duplicate regression evidence before publishing."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def verify(paths, minimum_tests):
    if minimum_tests < 1:
        raise ValueError('minimum tests must be positive')
    seen = set()
    passed = failed = skipped = 0
    for path in paths:
        path = Path(path)
        if path.stat().st_size > 32*1024*1024:
            raise ValueError('JUnit evidence too large')
        for case in ET.parse(path).iter('testcase'):
            identity = (case.get('classname', ''), case.get('name', ''))
            if not identity[1] or identity in seen:
                raise ValueError('missing or duplicate test identity')
            seen.add(identity)
            if case.find('failure') is not None or case.find('error') is not None:
                failed += 1
            elif case.find('skipped') is not None:
                skipped += 1
            else:
                passed += 1
    if failed or passed < minimum_tests:
        raise ValueError(f'regression evidence incomplete: passed={passed}, failed={failed}, required={minimum_tests}')
    return {'passed': passed, 'failed': failed, 'skipped': skipped, 'minimum_tests': minimum_tests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--junit-xml', type=Path, required=True, action='append')
    parser.add_argument('--minimum-tests', type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.junit_xml, args.minimum_tests)))


if __name__ == '__main__':
    main()
