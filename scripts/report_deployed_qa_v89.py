"""Summarize deployment QA without relabelling model misses as passes."""
import argparse
from collections import Counter
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basic', type=Path, required=True)
    parser.add_argument('--qa', type=Path, required=True)
    parser.add_argument('--junit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    basic = json.loads(args.basic.read_text(encoding='utf8'))
    qa = json.loads(args.qa.read_text(encoding='utf8'))
    if not qa['complete'] or basic['wheel_sha256'] != qa['expected_wheel_sha256']:
        raise ValueError('complete QA from the same deployed wheel is required')
    tests = ET.parse(args.junit).getroot().find('testsuite')
    cases = [row for row in qa['jobs'] if row['suite'] == 'retired']
    statuses = Counter(str(row['http_status']) for row in basic['calls'])
    for job in qa['jobs']:
        detail = json.loads((args.qa.parent/job['result_file']).read_text(encoding='utf8'))
        statuses.update(str(row['http_status']) for row in detail.get('calls', []))
    unit_total, unit_failed, unit_errors, unit_skipped = [int(tests.attrib[k]) for k in ('tests','failures','errors','skipped')]
    failed_names = [row.attrib['name'] for row in tests.findall('testcase') if row.find('failure') is not None]
    lines = [f"# AI 배포 및 QA 결과 — Modal {basic['modal_version']}", '',
        f"배포 시각: {basic['time_deployed']}",
        f"API: {basic['url']}", '',
        '지정한 향수 3건·로션 7건의 배합만 재선택하지 않는 수정본이다. 향 종류, 원료, 가중치와 평가식은 유지했다.',
        '주소·인증·실행 자원은 변경하지 않았다. GitHub push는 이번 작업에 포함하지 않았다.', '',
        '## 기능 검사', '',
        f"- 기본 배포 검사: {'통과' if basic['passed'] else '실패'}.",
        f"- 추가 QA: {sum(row['passed'] for row in qa['jobs'])}/{len(qa['jobs'])} 검사 그룹 통과.",
        f"- HTTP 기록: {sum(statuses.values())}건. 상태별 기록 {dict(statuses)}.",
        '- 의도된 422 입력/근거 거부, 409 이전 확인값 거부, 429 요청 제한도 검사에 포함한다.',
        '- confidence는 숫자 또는 null, 로션 생략 풀은 conditional_research로 확인했다.',
        '- 조향 SSE, 감사·버전 이벤트 SSE, 보고서 SSE와 JSON 일치를 확인했다.',
        '- diagnostic_only evaluate/reassess, 저장 후보 비교·자연어 수정, 보완 질문 왕복을 검사했다.',
        '- 바디워시 예측은 명시적인 합성 계수로 계약과 질량 보존을 검사했다. 실측 계수 검증이 아니다.', '',
        '## 제외 대상 10개 요청의 새 계산', '',
        '아래는 배포 이미지에서 다시 계산한 결과다. 기존 배합이 제외되는 것과 새 배합이 90점에 도달하는 것은 구분한다.', '',
        '| 제품 | 요청 ID | 다른 배합 반환 | 모델 점수 | 목표 통과 | 계산 시간 |',
        '| --- | --- | --- | ---: | --- | ---: |']
    for row in cases:
        score = row.get('score')
        duration = row.get('seconds')
        lines.append('| '+ ' | '.join((row.get('product','-'),row.get('case_id',str(row['case_index'])),
            '예' if row.get('replacement_composition_returned') else '확인 실패',
            f'{score:.4f}' if score is not None else '-', '예' if row.get('target_met') else '아니오',
            f'{duration:.2f}초' if duration is not None else '-')) + ' |')
    lines += ['', f"다른 배합 반환 {sum(bool(r.get('replacement_composition_returned')) for r in cases)}/{len(cases)}건, "
        f"90점 목표 통과 {sum(bool(r.get('target_met')) for r in cases)}/{len(cases)}건.",
        '이 결과를 전체 요청 100%나 실제 인간 후각 정확도라고 표시하지 않는다.', '',
        '## QA에서 확인한 문제', '',
        *[f"- {row['name']}: {row.get('error_type', 'failure')} — {row.get('error', 'see raw result')}"
          for row in qa['jobs'] if not row['passed']],
        '- 상세 관측값은 QA 폴더의 findings.json과 해당 요청·응답 원문에 보존한다.', '',
        '## 회귀 검사와 검증 범위', '',
        f'- 설치된 동일 wheel 기준 {unit_total-unit_failed-unit_errors-unit_skipped}/{unit_total}개 통과, '
        f'실패 {unit_failed}개, 오류 {unit_errors}개, 건너뜀 {unit_skipped}개.',
        f'- 실패 이름: {failed_names}.',
        '- V32 전용 테스트는 과거 봉인 배포 파일이 현재 설치 패키지에 없어 실패했다. 직전 v22 패키지에서도 같은 실패를 재현했다.',
        '- 현재 v23의 로션 생성과 물리 계산은 별도의 실제 배포 검사로 확인했다. 과거 테스트 실패 기록도 보존했다.',
        '- 원격 검사는 기존 Modal 관리자 SDK → 실제 배포 이미지 내부 ASGI 경로다.',
        '- 표의 시간은 이미지 내부 API 호출 시간이다. 공개 네트워크 왕복과 컨테이너 시작 시간을 포함한 종단간 지연은 아니다.',
        '- 무인증 공개 URL은 401이었다. 실제 백엔드 Proxy Token으로 공개 HTTPS를 호출하는 검사는 백엔드팀 담당으로 남아 있다.',
        '- 빈 근거 change-impact는 운영 자료를 삭제하지 않고 같은 이미지의 격리된 빈 저장소로 검사했다.',
        '- 감사 이력은 저장하지 않는 합성 시험 입력이다. 새 공급·규제 증거를 등록하거나 승인을 변경하지 않았다.', '',
        '## 원문과 식별 정보', '', f"- Wheel SHA256: `{basic['wheel_sha256']}`",
        f"- 묶음 SHA256: `{basic['bundle_sha256']}`", f'- 기본 검사: `{args.basic}`',
        f'- 추가 QA·요청/응답: `{args.qa.parent}`', f'- 회귀 원문: `{args.junit}`', '']
    if args.output.exists():
        raise ValueError('preserve prior report; select a new output')
    args.output.write_text('\n'.join(lines), encoding='utf8')
    print(str(args.output.resolve()))


if __name__ == '__main__':
    main()
