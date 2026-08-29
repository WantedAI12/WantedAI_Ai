# 비인간 데이터 허브

`fragrance_ai/data/nonhuman_data_hub.db`는 출처가 다른 비인간·공개 참조
자료를 덮어쓰지 않고 보존하는 lineage DB다. 사람 패널, 전문가 평가, 사용자
피드백과 관능 검증 자료는 이 허브에서 제외한다.

## 현재 저장 범위

현재 DB 스냅샷에는 75개 데이터 소스, 7,612개 물질 관찰, 98개 참조 처방과
605개 노트 행이 있다. EPA 원본 5개는 허브에 출처 연결로 기록되어 있고,
합성 학습 자산 24개는 격리된다. 이 숫자는 저장된 레코드 수이며 전 세계
향료·규제·관능 데이터를 포괄한다는 의미가 아니다.

```powershell
python -c "from fragrance_ai.recommender.data_hub import NonHumanDataHub; print(NonHumanDataHub().stats())"
```

각 source에는 원본 URI 또는 로컬 경로, SHA-256, 바이트 수, 갱신일,
라이선스 메모, 허용 용도 및 금지 용도를 남긴다. 서로 상충하는 관찰은 하나의
‘정답’으로 합치지 않는다.

## 출처 등급과 사용 경계

`curated_engineering`, `unverified_reference`, `published_reference` 등 일부
등급의 `reference_only` 처방 노트만 역사적 동시출현 prior에 제한적으로
참여할 수 있다. 이는 질량비가 없는 노트 참조이며 목표 향의 실제 조성이나
정답 처방이 아니다.

다음 자료는 검색·감사 목적이며 출시 증거가 될 수 없다.

- EPA 제품용도·독성 참조
- 미검증 규제 참조 행
- 합성 학습 자산
- 미검증 물성, 가격 또는 동시출현 기록

허브는 사람 관능 증거, 공급사 적격성, IFRA/법적 적합성, 실제 공급 로트의
물성, 시장 출시 승인을 제공하지 않는다.

## 재구축

다음 스크립트는 작업공간에서 비인간 파일을 찾아 출처·해시·용도 경계를
기록하며, 사람 관련 경로는 제외한다.

```powershell
python scripts\build_nonhuman_data_hub.py
python scripts\audit_data.py
```

입력 파일의 존재·권리·최신성은 운영자가 별도로 확인해야 한다. 허브를
재구축했다고 해서 외부 데이터를 내려받거나 원본의 상용 사용 권한을 얻는 것은
아니다.

EPA 추출의 개별 범위는 [EPA CompTox 데이터](EPA_COMPTOX_DATA.md)를, 번들
파일의 해시와 claim boundary는
[`data_manifest.json`](fragrance_ai/data/data_manifest.json)을 참조한다.

사람의 공개 관능값이 포함된 농도·혼합 연구 자료는 이 비인간 허브에 섞지 않는다.
별도 작업공간 DB와 적용 결과는
[HEADSPACE_SENSORY_DATA_V1.md](HEADSPACE_SENSORY_DATA_V1.md)에 기록한다.
