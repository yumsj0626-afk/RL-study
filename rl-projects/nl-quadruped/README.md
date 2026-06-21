# NL-Quadruped: 자연어 명령을 수행하는 사족보행 로봇 (캡스톤)

## 프로젝트 정체성

이 프로젝트는 `nl-conditioned-grid` 정찰 작업의 후속 캡스톤이다. 8x8 그리드 + tabular Q-learning에서 검증한
**"자연어는 정책이 아니라 명령 명세(spec)로 해석한다"** 는 아이디어를, **Isaac Sim 위의 사족보행 로봇(Unitree Go2)**
으로 끌어올린다.

- **LLM은 여전히 정책을 만들지 않는다.** 자연어를 JSON spec으로 변환하는 *해석기* 역할만 한다.
- **실제 제어(보행)는 강화학습(PPO) 정책이 맡는다.** 즉 RL은 locomotion 계층에 위치한다.
- **목표 산출물은 단일 인상적 시연 1개** — 자연어 한 문장이 실제 사족보행 로봇의 동작으로 이어지는 것을
  사진·영상으로 증명하는 보고서.

> 그리드 프로젝트는 "어디서 작동하고 어디서 깨지는지" 정찰이었다. 이 캡스톤은 "이렇게도 가능하더라"를
> 증거로 보여주는 PoC다.

---

## 핵심 통찰: 2층 계층 구조

자연어 명령으로 다리 관절을 직접 학습하는 것은 불가능에 가깝다. 그래서 **해석 계층**과 **제어 계층**을 분리한다.

```
자연어 1문장  (예: "오른쪽 위 구석으로 가되 중앙은 피해서 천천히")
   │
   ▼  LLM 파서 (gpt-4o, temperature=0)            ← nl-conditioned-grid 자산 재사용
JSON spec v2  (goal pose · 금지구역 · clearance · 속도선호 · preference · goal=null 실패)
   │
   ▼  spec → 2D occupancy grid → A* 경로계획        ← 그리드 프로젝트의 계획 로직 계승
waypoint 리스트
   │
   ▼  go-to-goal 컨트롤러 (스크립트)
velocity command  (vx, vy, yaw_rate)
   │
   ▼  Go2 locomotion 정책 (Isaac Lab에서 PPO로 학습 → freeze)   ← 진짜 RL, 보고서 해설 대상
관절 목표값
   │
   ▼  Isaac Sim 물리 시뮬레이션 → 카메라 녹화
mp4 + 스냅샷  →  보고서
```

원래 그리드 프로젝트의 **A* + 좌표 목표가 하이레벨 플래너로 승격**된다. 자연스러운 캡스톤 서사.

---

## 주요 결정사항

| 차원 | 결정 | 이유 |
|---|---|---|
| RL의 위치 | **locomotion (PPO velocity-tracking)** | LLM은 해석만, 제어가 RL이라는 원 프로젝트 정체성 유지 |
| locomotion 정책 | 사전학습 가져오기보다 **Isaac Lab에서 직접 짧게 PPO 학습 → freeze** | 학습곡선·reward 그래프 등 진짜 RL 산출물 확보. 비용 거의 동일 |
| 하이레벨 내비 | **스크립트(go-to-goal + A*)** | 연결·시연에 집중. 학습은 locomotion에 한정해 스코프 관리 |
| 시뮬레이터 | **Isaac Sim + Isaac Lab** | 사족보행 표준 RL 환경, GPU 병렬, 영상 녹화 지원 |
| 로봇 | **Unitree Go2** | Isaac Lab 기본 제공 에셋, velocity-tracking 태스크 존재 |
| 컴퓨팅 | **AWS EC2 GPU (g5.2xlarge)** | Lambda는 GPU·시간·렌더 제약으로 불가. 하이브리드(로컬+EC2)로 비용 최소화 |
| 인프라 | **NVIDIA 공식 Isaac Automator** | 직접 Terraform 작성 대신 공식 도구로 프로비저닝+Isaac Sim/Lab 설치 자동화(내부적으로 Terraform+Ansible). 트러블슈팅 최소화, `./stop`/`./start`로 비용 통제. 런북: [docs/01_phase1_isaac_automator.md](docs/01_phase1_isaac_automator.md) |
| 시연 범위 | **단일 인상적 시연 1개** | 보고서 증거 확보가 목적, 학습 일반화 검증이 아님 |

---

## 컴퓨팅 아키텍처 (하이브리드 — 비용 최적화)

GPU가 필요 없는 작업은 전부 로컬에서 공짜로, GPU는 학습+렌더링에만 잠깐.

```
[ 로컬 PC — 비용 0 ]
  LLM 파싱 · A* 플래너 · 컨트롤러 · 2D 목업 · 보고서 생성
        │  (spec.json, 코드, 시나리오 전달)
        ▼
[ EC2 g5.2xlarge GPU — 작업할 때만 ]
  Isaac Sim 학습 + 렌더링.  끝나면 S3로 산출물 → destroy
        │
        ▼
[ S3 — 月 몇 센트 ]  체크포인트 · mp4 · 스냅샷
```

**비용 수칙 (예상 총액 $15~40):** g5.2xlarge(~$1.2/hr) 온디맨드로 시작 → 익숙해지면 Spot(~$0.45/hr).
Terraform apply/destroy로 작업할 때만 과금. user-data에 **유휴 자동 stop** 안전장치. Deep Learning AMI로
드라이버 선설치. 리전 `us-east-1`. 자세한 내용은 [docs/00_setup_and_aws.md](docs/00_setup_and_aws.md).

---

## 단계 계획

| Phase | 내용 | 환경 | 비용 |
|---|---|---|---|
| **0** | 스키마 v2 + 파서 + A* 플래너 + go-to-goal 컨트롤러 + matplotlib 2D 목업 | 로컬 | 0 |
| **1** | AWS GPU 쿼터 증액(완료) + Isaac Automator로 g5.2xlarge 배포 → Isaac Sim/Lab 설치 + 스톡 태스크 검증 | 로컬→AWS | $ |
| **2** | Isaac Lab 설치 + Go2 평지 PPO 짧게 학습 → 체크포인트 + 녹화 파이프라인 검증 | EC2 GPU | $ |
| **3** | spec → Isaac 씬 생성(장애물 배치) + 컨트롤러 루프 + freeze 정책, NL 1문장 end-to-end | EC2 GPU | $ |
| **4** | 단일 인상 시연 렌더 + 스냅샷 + 자동 interpretation 보고서 | EC2 GPU | $ |

Phase 0·1은 병렬. 쿼터 승인을 기다리는 동안 두뇌 로직(Phase 0)을 완성한다.

---

## 디렉터리 구조 (예정)

```
nl-quadruped/
├── README.md                 # 이 문서
├── .env.example              # OpenAI 키 등 (복사 → .env)
├── docs/
│   └── 00_setup_and_aws.md   # 자격증명 체크리스트 + AWS 쿼터 + 비용 수칙
├── schemas/
│   └── command_schema_v2.json   # 3D 월드 spec (goal pose / 금지구역 / clearance / speed)
├── prompts/
│   └── parser_prompt_v2.txt
├── nl_parser.py              # 자연어 → spec (gpt-4o, temp=0)
├── planner.py                # spec → occupancy grid → A* → waypoint
├── controller.py             # waypoint → velocity command (go-to-goal)
├── preview_2d.py             # Isaac 없이 NL→경로→속도명령 시각 검증 (단일)
├── run_parser_tests.py       # 배치 파서 실험 (commands.json, OpenAI 호출)
├── run_experiments.py        # 오프라인 회귀 검증 (specs/*, API 불필요, 비용 0)
├── test_cases/
│   ├── commands.json         # 자연어 명령 세트 (파서 실험용)
│   └── specs/                # 오프라인 spec 예제 (회귀·목업용)
├── results/                  # 생성물: preview_*.png, spec_*, interpretation_*, 분석표
└── isaac/                    # Isaac Lab 태스크/씬/실행 스크립트 (Phase 2~3)
                              # (인프라는 외부 Isaac Automator 사용 — infra/ 디렉터리 불필요)
```

---

## nl-conditioned-grid에서 이어지는 자산

- LLM 파서 프롬프트 + JSON 스키마 검증 패턴
- 자연어 spec을 환경 생성자로 넘기는 builder 구조
- 단계별 실패 기록(parse / build / control)
- 실험별 `interpretation.md` 자동 생성 흐름 → 보고서 자동화

## 실패 분류 (Failure Taxonomy)

자연어/명세가 불완전할 때 시스템은 목표를 지어내지 않고 **계획 단계에서 통제된 실패(controlled failure)** 로
멈춘다. 각 케이스는 `results/preview_*.png` 실패 증거 이미지 + `failure_taxonomy.md` 로 기록된다.
(캡스톤에서는 이 지점이 사용자 재질문 루프 / 재계획 / 안전 모니터로 연결된다.)

| 케이스 | 트리거 | 메시지 |
|---|---|---|
| `underspecified` | 목표 미명시 (`goal=null`) | Goal is underspecified |
| `fail_goal_in_obstacle` | 목표가 금지영역 안 | Goal ... inside an inflated forbidden region |
| `fail_start_in_obstacle` | 시작이 금지영역 안 | Start ... inside an inflated forbidden region |
| `fail_no_path` | 시작·목표가 장애물로 분리 | No collision-free path from start to goal |
| `fail_start_equals_goal` | 시작 == 목표 | Start equals goal |

## 실험 결과 요약 (2D, GPU 0)

- **파서 (T01–T10)**: 한국어 명령 10종 **전부 스키마 검증 통과**. 공간표현 좌표화, hard/soft 구분,
  속도·선호 부사 해석, 불완전 명령 시 `goal=null`(목표 미생성)까지 의도대로 처리.
  → `results/parser_test_analysis.md`
- **기하 회귀 (7 spec)**: 성공 2종(하드 회피·소프트 우회, **충돌 0**) + 통제된 실패 5종.
  `preference=safe`일 때 위험원을 실제로 우회(waypoint 2→6, 경로 차이로 입증).
  → `results/all_experiments.json`, `results/preview_*.png`
- **전체 수치·분석·증거 그림**은 [REPORT.md](REPORT.md) 의 4.4 / 6.1 절 참고.

## 파서 모델 / 게이트웨이

`nl_parser` 는 OpenAI 호환 클라이언트를 쓴다. `.env` 로 모델/엔드포인트를 바꿀 수 있다:
- `NLQ_PARSER_MODEL` — 모델명 (기본 `gpt-4o`, 현재 `claude-sonnet-4-6` 게이트웨이 사용)
- `OPENAI_BASE_URL` — OpenAI 호환 게이트웨이 주소 (선택)
- `NLQ_PARSER_JSON_MODE` — `auto`(기본) | `on` | `off`. `auto` 는 첫 호출에서 `response_format=json_object`
  지원 여부를 1회 탐지해 캐시한다. **Anthropic 계열 게이트웨이는 미지원이므로 `off` 권장**(불필요한 탐지 호출 제거).

## 실행 (Phase 0)

```powershell
pip install -r ..\requirements.txt

# (1) 오프라인 회귀 검증 — API 불필요, 비용 0. 기하 스택이 안 깨졌는지 잠금.
python run_experiments.py

# (2) 파서 실험 — .env 에 OPENAI_API_KEY 필요. 여러 번 돌려 interpretation_notes 점검.
Copy-Item .env.example .env   # OPENAI_API_KEY 채우기
python run_parser_tests.py --preview      # 파싱 + plan/sim/render 까지 end-to-end
python preview_2d.py "오른쪽 위 구석으로 가되 중앙은 절대 피해서 천천히"   # 단일 명령

# 결과는 results/ : preview_*.png, parser_test_analysis.md, interpretation_*.md
```

> 상태: **Phase 0~3 완료, end-to-end 동작 확인.** 자연어 → spec → A*/go-to-goal → 속도명령 →
> Isaac Sim의 Go2 PPO 보행 정책으로 목표 좌표까지 보행("goal reached" 로그·영상 확인). 결과 정리는
> **[REPORT.md](REPORT.md)** 참고. 장애물 prim 스포닝(Isaac 3D)은 시간상 생략, 고수준 회피는 2D 목업으로 입증.
