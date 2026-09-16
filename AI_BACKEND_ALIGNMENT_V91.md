# V91 백엔드 연동 정리 — 2026-09-17

## 기준

대상 백엔드는 `WantedAI12/WantedAI_BE`의 `main`, 커밋 `51c60a4cbe0e8eff8674482a2db770ea54a3967f`입니다. 전체 소스 목록에서 AI 호출 경로를 추적하고 실제 HTTP 클라이언트, 28개 계약 DTO, 후보 저장·재평가·수정·예측·로션 상세 변환 경로를 대조했습니다. 백엔드 소스는 변경하지 않았습니다. 정적 리뷰와 실제 DTO 실행을 함께 사용했으며 전체 Spring 서버를 구동한 검사는 아닙니다.

## AI에서 고친 내용

| 항목 | 반영 내용 |
| --- | --- |
| confidence | 기존 숫자 또는 null 계약을 유지합니다. 근거 설명은 confidence_kind입니다. |
| simulation_confidence | 설명 문자열 때문에 Java 숫자 변환이 실패하던 문제를 수정했습니다. 숫자 또는 null이며 설명은 simulation_confidence_kind에 보존합니다. |
| 목표 미달 향수 | 이미 계산된 유효 배합을 recipe에 반환합니다. 전달 상태는 recipe_generated_target_not_met, 원래 엔진 상태는 assessment_status에 보존합니다. 점수와 full_profile_target_met는 변경하지 않습니다. |
| 큰 로션 응답 | 실제 약 28.95MB 응답을 7.64MB로 줄였습니다. 레시피·점수·목표 판정·제조 준비·제품 모델·지각 모델은 평문 JSON 그대로입니다. 큰 시뮬레이션 및 규제 상세 자료는 응답 내부 diagnostic_archive에 무손실 보관합니다. |
| 빈 진단 정책 | diagnostic_only=true와 evidence_policy={}로 준비→재평가가 가능합니다. 1kg 기준 배치 및 비구속 진단 상한을 명시하고 사용자 구매 조건이나 운영 승인으로 취급하지 않습니다. 일반 모드는 누락된 조건을 질문합니다. |
| 고정 배합 정밀도 | 서버가 생성한 소수점 5자리 이상·미량 원료의 비율을 다시 보낼 때 임의 반올림이나 삭제 없이 평가합니다. 음수·NaN·합계 오류는 거부합니다. |
| 고정 배합 목표 | 기존 재평가가 요청과 무관하게 95점을 강제하던 부분을 수정했습니다. 배포 설정 90점과 사용자가 명시한 더 높은 목표를 동일하게 따릅니다. |
| 진단 후 수정 | 재평가→자연어 수정 시 진단 상태와 기본 정책 출처가 유지됩니다. 진단용 가정이 확정된 구매 조건으로 승격되지 않습니다. |

현재 백엔드는 조향·감사·보고서 SSE를 AI에서 직접 받아 중계하는 구조가 아니라 자체 작업/이벤트·보고서 흐름을 갖고 있습니다. AI의 기존 SSE 엔드포인트는 별도로 유지합니다.

## 백엔드가 그대로 사용할 항목

- 기존 Modal URL, 인증 방식, 경로를 유지합니다. 토큰을 새로 만들거나 기존 비밀값을 바꾸지 않았습니다.
- 향수 생성 결과의 confidence와 simulation_confidence는 nullable 숫자입니다. null을 0점으로 해석하지 않습니다.
- recipe는 배합 데이터 존재 여부이고 target_match_met 및 full_profile_target_met는 목표 달성 여부입니다. 서로 다른 의미입니다.
- target_match_score는 계산 일치도이며 similarity_score의 실제 계산값과 함께 전달됩니다. 점수 하한 보정은 없습니다.
- 로션 registry_pool 생략 시 conditional_research가 적용됩니다. 사용자가 명시한 위험·가격 조건은 유지합니다.
- 진단 준비와 실행은 같은 입력을 보내야 review_id가 일치합니다. 빈 정책으로 준비했다면 재평가에도 동일하게 빈 정책을 보냅니다.
- 운영 증거가 없을 때 일반 evaluate/reassess는 계속 차단됩니다. 실제 등록된 공개자료를 이용하는 명시적 진단 모드와 구분합니다.
- change-impact는 등록된 이전·현재 자료로 비교할 수 있을 때만 성공합니다. 자료 자체가 없으면 기존 422/EVIDENCE_SNAPSHOTS_MISSING입니다.
- 8MiB 백엔드 읽기 한도를 유지합니다. 압축 후에도 필수 결과가 8,000,000바이트를 초과하면 데이터 삭제 대신 413/AI_RESPONSE_TOO_LARGE로 명시합니다.
- diagnostic_archive는 현재 백엔드 DTO에서 무시하는 대용량 세부 진단만 포함합니다. 원문 저장 시 함께 보존됩니다. 상세 원본 복구에는 AI 패키지 response_transport.unpack_lotion_response를 사용할 수 있습니다. 해제 시 크기·SHA256을 검증합니다.

## 남아 있는 백엔드 정책 경계

로션 `LotionDesignResponse.isUsableCandidate()`는 **profile_target_met=true와 recipe 존재를 모두 요구**합니다. 따라서 실제 85.56점·목표 미달 로션은 JSON 파싱과 레시피 전달이 성공해도 백엔드에 후보로 저장되지는 않습니다. 이를 우회하기 위해 false를 true로 바꾸지 않았습니다. 목표 달성 로션의 저장 조건은 실제 Java 메서드에서 통과했습니다.

로션의 전용 설계·시뮬레이션·최적화와 향수용 v2 후보 재평가는 같은 계산이 아닙니다. v2 향수 흐름을 로션 제조 검증으로 해석하지 않습니다. 바디워시 실제 물리 계수 확보도 이번 연동 수정의 완료 항목이 아닙니다.

## 로컬 검증

- 허용한 파일 9개만 V90 동결 패키지 위에 반영했습니다. 나머지 Python 소스 205개를 보존했습니다.
- 원료 29,259행, 공유 모델 가중치, 목표 참조 데이터가 변경되지 않았습니다. 미채택 모델 실험은 포함하지 않았습니다.
- 설치된 패키지 관련 테스트 **278개 통과**(주요 계약 216개 + 배합·확장 API·감사/보고서 62개), 실패·건너뜀 0개입니다. 추가 검사에서 V32 배포 파일을 직접 불러오던 과거 테스트를 현재 동결 패키지 팩토리와 실제 프로필 검사로 갱신했습니다.
- 기존 저장 결과 12개와 API 6회 재생: 원 점수 보존, JSON/SSE 결과 일치, 운영 증거 차단 유지입니다. 재생은 새 모델 품질 평가가 아닙니다.
- 새 로컬 추론과 API 호출 **23개**: 준비·진단 평가·고정 재평가·비교·수정·정책 보완 연결 확인입니다. 의도된 빈 증거 422를 포함합니다.
- 실제 백엔드 DTO와 Jackson 3.1.5로 새 응답 **15개 모두 파싱 성공**했습니다. Java 17에서 DTO를 컴파일했으며 Java 21 Spring 애플리케이션 전체 테스트는 아닙니다.
- 목표 미달 향수 및 1,096원료 로션의 추가 Java 검사에서 파싱·8MiB 한도·숫자 변환 정상, 향수 저장 조건 통과, 로션 목표 미달 저장 조건 거부를 각각 확인했습니다.
- 큰 로션 응답의 압축 후 복구 JSON은 원본과 완전히 같습니다.

검증 기록:

- `.benchmarks/v91_backend_alignment/installed-tests-01.xml`
- `.benchmarks/v91_backend_alignment/installed-extra-tests-04.xml`
- `.benchmarks/v91_backend_alignment/java-fresh-local.json`
- `.benchmarks/v91_backend_alignment/java-after-replay.json`
- `output/backend-v91-fresh-local-20260917-02/verification.json`
- `output/backend-alignment-v91-local-20260917/verification.json`

## 패키지 식별

- 동결 패키지: `.benchmarks/v91_backend_alignment/package-01`
- wheel SHA256: `344f384f254bfd5cbfc9d7a76091167577b7e46e3a56cc055935e0783c5a87cd`
- 모델 SHA256: `158b9f82ff9fb784eeec114ef1f7bbfc4d613badeb98f46e0f60435d5af38d2f`
- 배포 묶음 SHA256: `5b00fa7445adff767f2b04b20863c5f782d07dbff0bf05f77c623fc8af7ed54d`
- 릴리스 ID: `v91-backend-compatibility-20260917`

배포 후 검증은 `output/backend-v91-deployed-20260917/verification.json`에서 별도로 확인합니다. 이 검사는 비공개 Modal SDK로 실제 배포 이미지의 ASGI 앱을 호출하는 방식입니다. 백엔드 팀의 Proxy Token으로 공개 URL을 통과하는 종단 간 검증과는 다릅니다.

## 실제 배포 확인

- 배포 완료: 2026-09-17 00:16:37 KST.
- URL: https://junseong2im--perfumery-ai-core-web.modal.run
- 현재 앱 ID: `ap-CQvtEZSVOwIU5zCcfpjPyV`, 앱 이력 `v1`.
- 이전 앱 `ap-Uc8rTdrF9SDei5yiFV14Ke`는 이번 배포 전인 2026-09-16 23:31:18 KST에 이미 중지된 상태였습니다. 따라서 새 앱의 이력을 이전 앱 v24의 연속 번호로 표기하지 않습니다. 기존 URL과 requires_proxy_auth 설정은 유지합니다.
- 배포 이미지 API **26건**: HTTP 200 25건, 의도된 빈 증거 HTTP 422 1건. 모두 기대 결과와 일치합니다.
- 배포 응답을 실제 백엔드 DTO/Jackson으로 다시 읽은 **15건 모두 성공**했습니다. 향수·목표 달성 로션의 후보 저장 조건도 통과했습니다.
- 준비→진단 평가→고정 재평가→저장 결과 비교→자연어 수정과 일반 모드의 정책 질문→답변 흐름을 확인했습니다.
- 예시 새 추론: 향수 91.990706점/12원료, 로션 90.131675점/73원료. 두 예시는 전체 요청 정확도나 통과율 평가가 아닙니다.
- 공개 URL의 무인증 접근은 HTTP 401입니다. 백엔드 Proxy Token으로의 인증 종단 검사는 하지 않았습니다.
- 배포 검증 당시에는 GitHub push를 하지 않았습니다. 이후 누적 소스 공개 및 PR 범위는 [V91 공개 기록](release/V91/README.md)을 참고하세요.

배포 응답 Java 검사: `.benchmarks/v91_backend_alignment/java-deployed.json`.

## 큰 로션 요청의 배포 후 재계산

이전에 1,096개 원료로 28.95MB가 나왔던 동일한 요청을 새로 계산했습니다. 이번에는 시간 제한 탐색의 결과로 853원료 배합이 반환됐습니다. 원료 수를 강제로 잘라낸 결과가 아닙니다.

- 실제 배포 이미지 ASGI 처리: **281.6316초**, 300초 이내. 콜드 스타트·공개 URL·백엔드 네트워크를 포함한 총 응답시간 보장은 아닙니다.
- 실제 점수: **85.788375점**, profile_target_met=false, recipe 853개 반환.
- 원래 JSON **23,322,844바이트 → 전송 JSON 6,186,523바이트**.
- 무손실 복원 해시 및 실제 Java DTO 파싱 통과. 기존 8MiB 한도 이내입니다.
- 기존에 제외한 배합을 다시 선택하지 않았습니다.
- 이 결과는 목표 미달이므로 백엔드의 로션 저장 조건에서는 여전히 거부됩니다.
- 원문 및 시간 기록: `output/backend-v91-large-lotion-deployed-20260917/retired_04/result.json`.
- Java 기록: `.benchmarks/v91_backend_alignment/java-deployed-large.json`.
