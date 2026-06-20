# -*- coding: utf-8 -*-
"""Build an Obsidian vault from the RL-study archive.
- copies notes with clean filenames
- prepends YAML frontmatter (title/course/module/type/tags)
- rewrites image links to Obsidian ![[...]] embeds with unique names
- replaces broken Notion attachment: links with callouts
- appends a "관련 노트" wikilink section
- generates per-course MOCs + home MOC
"""
import os, re, shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
VAULT = os.path.join(ROOT, "RL-study-vault")

C1 = "코스1 - 강화학습의 기초"
C2 = "코스2 - 샘플기반 학습 방법"
C3 = "코스3 - 함수근사 예측·제어"
C4 = "코스4 - 캡스톤 시스템"
PROJ = "프로젝트 - 개인 RL 실험"

COURSE_DIRS = [C1, C2, C3, C4, PROJ]

IMG_RE = re.compile(r'!\[[^\]]*\]\(([^)]+)\)')


def clean_tag(s):
    s = s.strip()
    s = re.sub(r'[()\[\]{}:/\\,.\s]+', '-', s)
    s = re.sub(r'-+', '-', s).strip('-')
    return s


# Each note: src (relative to ROOT), course dir, dst stem, module, type,
# tags, related (dst stems), summary, prefix, extra_images (relative to src dir)
NOTES = [
    # ---------------- Course 1 ----------------
    dict(src="1. 강화학습의 기초/Module1, 2.md", course=C1,
         dst="M1·2 - 밴딧과 순차적 의사결정", module="Module 1-2", type="강의노트",
         tags=["밴딧", "탐험-활용", "행동가치", "UCB", "낙관적초기값", "epsilon-greedy"],
         related=["M3 - 마르코프 의사결정 과정(MDP)"],
         summary="K-armed 밴딧과 탐험-활용 트레이드오프, 행동가치 추정의 기초.",
         prefix="c1m12"),
    dict(src="1. 강화학습의 기초/Module3/Module3.md", course=C1,
         dst="M3 - 마르코프 의사결정 과정(MDP)", module="Module 3", type="강의노트",
         tags=["MDP", "마르코프성", "보상", "할인율", "전이함수", "수익"],
         related=["M1·2 - 밴딧과 순차적 의사결정", "M4 - 정책·가치함수·벨만방정식",
                  "M3 과제 - MDP 설계 3가지", "논문리뷰 - 위성통신 에너지효율 최적화 RL"],
         summary="상태·행동·보상·전이로 정의되는 MDP와 수익/할인율 개념.",
         prefix="c1m3"),
    dict(src="1. 강화학습의 기초/Module3/모듈3 과제/모듈 3 task.md", course=C1,
         dst="M3 과제 - MDP 설계 3가지", module="Module 3", type="과제",
         tags=["MDP설계", "보상설계", "상태설계", "행동설계"],
         related=["M3 - 마르코프 의사결정 과정(MDP)"],
         summary="병원 응급실·음악추천·농업관개를 MDP로 직접 설계한 과제.",
         prefix="c1m3task"),
    dict(src="1. 강화학습의 기초/Module4/Module4.md", course=C1,
         dst="M4 - 정책·가치함수·벨만방정식", module="Module 4", type="강의노트",
         tags=["정책", "가치함수", "벨만방정식", "최적정책"],
         related=["M3 - 마르코프 의사결정 과정(MDP)", "M5 - 동적 프로그래밍(DP)",
                  "M4 - 최종 요약"],
         summary="정책·상태/행동 가치함수와 벨만방정식, 최적정책 도출.",
         prefix="c1m4"),
    dict(src="1. 강화학습의 기초/Module4/모듈4 최종 요약.md", course=C1,
         dst="M4 - 최종 요약", module="Module 4", type="요약",
         tags=["정책", "가치함수", "벨만방정식", "최적정책"],
         related=["M4 - 정책·가치함수·벨만방정식"],
         summary="정책·가치함수·벨만방정식·최적정책 관계를 압축 정리.",
         prefix="c1m4sum"),
    dict(src="1. 강화학습의 기초/Module 5/Module5.md", course=C1,
         dst="M5 - 동적 프로그래밍(DP)", module="Module 5", type="강의노트",
         tags=["동적프로그래밍", "정책반복", "가치반복", "GPI", "부트스트래핑"],
         related=["M4 - 정책·가치함수·벨만방정식", "실습 - ParkingWorld 동적계획법",
                  "M3 - TD 학습(예측)"],
         summary="정책평가/개선/반복, 가치반복, GPI 프레임워크와 근사 DP.",
         prefix="c1m5"),
    dict(src="1. 강화학습의 기초/모듈1 최종 실습 과제 파일/Readme.md", course=C1,
         dst="실습 - ParkingWorld 동적계획법", module="Module 1 실습", type="실습",
         tags=["정책반복", "가치반복", "GPI", "동적프로그래밍"],
         related=["M5 - 동적 프로그래밍(DP)"],
         summary="ParkingWorld에서 DP로 주차 가격 정책을 최적화한 실습.",
         prefix="c1hw1"),
    dict(src="1. 강화학습의 기초/MDP논문 리뷰/저궤도 인공위성 통신에서 도플러 효과를 고려한 에너지 효율 최적화 강화학습 알고리즘.md",
         course=C1, dst="논문리뷰 - 위성통신 에너지효율 최적화 RL", module="논문 리뷰",
         type="논문리뷰",
         tags=["MDP설계", "PPO", "위성통신", "응용", "도플러효과"],
         related=["M3 - 마르코프 의사결정 과정(MDP)", "M5 - 동적 프로그래밍(DP)"],
         summary="LEO 위성통신 에너지효율 최적화를 MDP/PPO로 푼 논문 분석.",
         prefix="c1paper"),

    # ---------------- Course 2 ----------------
    dict(src="2. 샘플기반 학습 방법/Module2/Module2.md", course=C2,
         dst="M2 - 몬테카를로 방법", module="Module 2", type="강의노트",
         tags=["몬테카를로", "exploring-starts", "epsilon-soft", "off-policy", "중요도샘플링"],
         related=["M2 퀴즈 - MC·오프폴리시 복습", "M3 - TD 학습(예측)",
                  "M5 - 동적 프로그래밍(DP)"],
         summary="에피소드 경험으로 Q를 추정하는 MC와 off-policy 중요도샘플링.",
         prefix="c2m2"),
    dict(src="2. 샘플기반 학습 방법/Module2/퀴즈 정리/퀴즈를 통한 이전 복습 정리.md", course=C2,
         dst="M2 퀴즈 - MC·오프폴리시 복습", module="Module 2", type="퀴즈",
         tags=["몬테카를로", "off-policy", "coverage", "중요도샘플링"],
         related=["M2 - 몬테카를로 방법"],
         summary="MC 적용 조건과 off-policy·coverage 개념 오답노트.",
         prefix="c2m2quiz"),
    dict(src="2. 샘플기반 학습 방법/Module3/Module3.md", course=C2,
         dst="M3 - TD 학습(예측)", module="Module 3", type="강의노트",
         tags=["TD학습", "부트스트래핑", "온라인학습", "TD오차"],
         related=["M2 - 몬테카를로 방법", "M3 - TD 예측 통합 요약",
                  "M4 - TD 제어(SARSA·Q러닝)", "M5 - 동적 프로그래밍(DP)"],
         summary="DP의 부트스트래핑과 MC의 모델프리를 결합한 TD(0) 예측.",
         prefix="c2m3"),
    dict(src="2. 샘플기반 학습 방법/Module3/Module3_통합요약.md", course=C2,
         dst="M3 - TD 예측 통합 요약", module="Module 3", type="요약",
         tags=["TD학습", "DP-MC-TD비교", "강화학습역사"],
         related=["M3 - TD 학습(예측)", "M3 특강 - Barto·Sutton 역사 대담"],
         summary="DP/MC/TD 비교와 랜덤워크 실험, 특강 핵심을 한눈에.",
         prefix="c2m3sum"),
    dict(src="2. 샘플기반 학습 방법/Module3/모듈3특강.md", course=C2,
         dst="M3 특강 - Barto·Sutton 역사 대담", module="Module 3", type="특강",
         tags=["강화학습역사", "행동주의", "내재적동기", "신경과학"],
         related=["M3 - TD 예측 통합 요약"],
         summary="TD 학습의 기원과 강화학습의 역사에 대한 대담.",
         prefix="c2m3talk"),
    dict(src="2. 샘플기반 학습 방법/Module3/모듈3 퀴즈 정리/모듈3 퀴즈정리.md", course=C2,
         dst="M3 퀴즈 - TD(0)·MC 개념 정리", module="Module 3", type="퀴즈",
         tags=["TD학습", "부트스트래핑", "bias-variance"],
         related=["M3 - TD 학습(예측)"],
         summary="TD(0) 정의·bootstrapping·bias-variance 퀴즈 정리.",
         prefix="c2m3quiz"),
    dict(src="2. 샘플기반 학습 방법/Module3/모듈3실습/Readme.md", course=C2,
         dst="M3 실습 - Cliff Walking TD(0) 정책평가", module="Module 3 실습", type="실습",
         tags=["TD-0", "정책평가", "환경설계", "절벽걷기"],
         related=["M3 - TD 학습(예측)", "절벽걷기 - SARSA vs Q러닝 비교"],
         summary="Cliff Walking을 코드로 옮기고 TD(0)로 세 정책을 평가.",
         prefix="c2m3lab"),
    dict(src="2. 샘플기반 학습 방법/Module4/Module4.md", course=C2,
         dst="M4 - TD 제어(SARSA·Q러닝)", module="Module 4", type="강의노트",
         tags=["SARSA", "Q러닝", "온폴리시", "오프폴리시", "expected-sarsa", "절벽걷기"],
         related=["M3 - TD 학습(예측)", "M4 - TD 제어 종합 요약",
                  "M4 실습 - Q러닝·Expected SARSA Cliff World",
                  "절벽걷기 - SARSA vs Q러닝 비교"],
         summary="SARSA(on-policy)·Q러닝(off-policy)·Expected SARSA 제어.",
         prefix="c2m4"),
    dict(src="2. 샘플기반 학습 방법/Module4/모듈4종합요약.md", course=C2,
         dst="M4 - TD 제어 종합 요약", module="Module 4", type="요약",
         tags=["SARSA", "Q러닝", "expected-sarsa"],
         related=["M4 - TD 제어(SARSA·Q러닝)"],
         summary="세 TD 제어 알고리즘의 타겟·장단점 비교 요약.",
         prefix="c2m4sum"),
    dict(src="2. 샘플기반 학습 방법/Module4/모듈4 실습/README.md", course=C2,
         dst="M4 실습 - Q러닝·Expected SARSA Cliff World", module="Module 4 실습", type="실습",
         tags=["Q러닝", "expected-sarsa", "절벽걷기", "step-size"],
         related=["M4 - TD 제어(SARSA·Q러닝)", "절벽걷기 - SARSA vs Q러닝 비교"],
         summary="Cliff World에서 Q러닝 vs Expected SARSA 경로·민감도 실험.",
         prefix="c2m4lab"),
    dict(src="2. 샘플기반 학습 방법/Module5/Module5..md", course=C2,
         dst="M5 - 계획·학습·행동(Dyna)", module="Module 5", type="강의노트",
         tags=["모델기반RL", "Dyna", "샘플모델", "분포모델", "Q계획"],
         related=["M5 특강 - 모델기반 RL", "M5 - 동적 프로그래밍(DP)"],
         summary="모델·계획·학습을 통합한 Dyna 아키텍처와 Dyna-Q(+).",
         prefix="c2m5"),
    dict(src="2. 샘플기반 학습 방법/Module5/모듈5특강 요약.md", course=C2,
         dst="M5 특강 - 모델기반 RL", module="Module 5", type="특강",
         tags=["모델기반RL", "로봇제어", "시뮬레이션"],
         related=["M5 - 계획·학습·행동(Dyna)"],
         summary="로봇에서 모델기반 RL이 중요한 이유와 근사 계획.",
         prefix="c2m5talk"),

    # ---------------- Course 3 ----------------
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module2/Module2.md", course=C3,
         dst="M2 - 함수근사 예측(선형근사·경사하강)", module="Module 2", type="강의노트",
         tags=["함수근사", "선형근사", "경사하강법", "semi-gradient-TD", "목적함수VE"],
         related=["M3 - 특징 설계(Coarse·Tile·신경망)", "M4 - 함수근사 제어·평균보상",
                  "M3 - TD 학습(예측)"],
         summary="선형 함수근사와 경사하강으로 가치함수를 학습(MC/TD).",
         prefix="c3m2"),
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module2/Module2_특강.md", course=C3,
         dst="M2 특강 - 옵션과 시간추상화", module="Module 2", type="특강",
         tags=["옵션", "시간추상화", "계층강화학습", "semi-MDP"],
         related=["M2 - 함수근사 예측(선형근사·경사하강)"],
         summary="Doina Precup의 옵션 프레임워크와 시간적 추상화.",
         prefix="c3m2talk"),
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module2/모듈2실습/README.md", course=C3,
         dst="M2 실습 - 상태집계 랜덤워크 분석", module="Module 2 실습", type="실습",
         tags=["상태집계", "일반화", "RMSVE", "표현력"],
         related=["M2 - 함수근사 예측(선형근사·경사하강)", "M3 실습 - 신경망 vs 타일코딩"],
         summary="500-상태 랜덤워크에서 상태집계 semi-gradient TD 분석.",
         prefix="c3m2lab"),
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module3/Module3.md", course=C3,
         dst="M3 - 특징 설계(Coarse·Tile·신경망)", module="Module 3", type="강의노트",
         tags=["특징설계", "coarse-coding", "tile-coding", "신경망", "역전파"],
         related=["M2 - 함수근사 예측(선형근사·경사하강)", "M3 실습 - 신경망 vs 타일코딩",
                  "M4 - 함수근사 제어·평균보상"],
         summary="고정 특징(Coarse/Tile)부터 학습된 특징(신경망)까지.",
         prefix="c3m3"),
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module3/모듈3실습/README_assignment2.md", course=C3,
         dst="M3 실습 - 신경망 vs 타일코딩", module="Module 3 실습", type="실습",
         tags=["신경망", "tile-coding", "Adam", "sample-efficiency"],
         related=["M3 - 특징 설계(Coarse·Tile·신경망)", "M2 실습 - 상태집계 랜덤워크 분석"],
         summary="랜덤워크에서 신경망과 타일코딩의 학습 효율 비교.",
         prefix="c3m3lab"),
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module4/Module4.md", course=C3,
         dst="M4 - 함수근사 제어·평균보상", module="Module 4", type="강의노트",
         tags=["함수근사제어", "expected-sarsa", "Q러닝", "평균보상", "differential-sarsa"],
         related=["M3 - 특징 설계(Coarse·Tile·신경망)", "M5 - 정책 경사·Actor-Critic",
                  "M4 - TD 제어(SARSA·Q러닝)"],
         summary="행동가치 근사 제어와 연속 문제용 평균보상 정식화.",
         prefix="c3m4"),
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module4/모듈4실습/README_assignment3.md", course=C3,
         dst="M4 실습 - Mountain Car Sarsa", module="Module 4 실습", type="실습",
         tags=["MountainCar", "tile-coding", "sarsa", "연속상태"],
         related=["M4 - 함수근사 제어·평균보상"],
         summary="연속상태 Mountain Car에서 타일코딩 semi-gradient SARSA.",
         prefix="c3m4lab"),
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module5/Module5.md", course=C3,
         dst="M5 - 정책 경사·Actor-Critic", module="Module 5", type="강의노트",
         tags=["정책경사", "actor-critic", "softmax정책", "가우시안정책", "평균보상"],
         related=["M4 - 함수근사 제어·평균보상", "M5 실습 - 진자 세우기 Actor-Critic"],
         summary="가치함수를 거치지 않고 정책을 직접 최적화(Actor-Critic).",
         prefix="c3m5"),
    dict(src="3. 함수 근사치를 사용한 예측 및 제어/Module5/모듈5 실습/README_assignment4.md", course=C3,
         dst="M5 실습 - 진자 세우기 Actor-Critic", module="Module 5 실습", type="실습",
         tags=["actor-critic", "평균보상", "wrap-tile-coding", "진자"],
         related=["M5 - 정책 경사·Actor-Critic"],
         summary="Pendulum Swing-Up에 평균보상 Softmax Actor-Critic 적용.",
         prefix="c3m5lab"),

    # ---------------- Projects ----------------
    dict(src="rl-projects/cliff-walking/README.md", course=PROJ,
         dst="절벽걷기 - SARSA vs Q러닝 비교", module="cliff-walking", type="프로젝트",
         tags=["SARSA", "Q러닝", "온폴리시", "오프폴리시", "절벽걷기"],
         related=["M4 - TD 제어(SARSA·Q러닝)", "M4 실습 - Q러닝·Expected SARSA Cliff World"],
         summary="4×12 절벽에서 SARSA(안전)와 Q러닝(위험·최단) 경로 비교.",
         prefix="projcliff",
         extra_images=["results/learning_curves.png",
                       "results/policy_sarsa_eps01.png",
                       "results/policy_qlearning_eps01.png"]),
    dict(src="rl-projects/nl-conditioned-grid/README.md", course=PROJ,
         dst="NL-Grid - 정찰 정리", module="nl-conditioned-grid", type="프로젝트",
         tags=["자연어조건부", "MDP명세", "grid-world", "Q러닝", "LLM파서"],
         related=["NL-Grid - 시스템 아키텍처", "NL-Grid - 명령 스키마 v1 설계",
                  "NL-Grid - 구현 명세서", "실험 - 실패 분류·위험도 분석"],
         summary="한국어 명령을 LLM이 MDP 명세로 파싱해 Q러닝으로 푸는 시스템.",
         prefix="projnl"),
    dict(src="rl-projects/nl-conditioned-grid/architecture.md", course=PROJ,
         dst="NL-Grid - 시스템 아키텍처", module="nl-conditioned-grid", type="프로젝트",
         tags=["아키텍처", "파이프라인", "파서", "환경빌더"],
         related=["NL-Grid - 정찰 정리", "NL-Grid - 명령 스키마 v1 설계"],
         summary="명령→파서(gpt-4o)→스키마검증→환경빌더→Q러닝 5단계 파이프라인.",
         prefix="projarch"),
    dict(src="rl-projects/nl-conditioned-grid/daily_log.md", course=PROJ,
         dst="NL-Grid - 프로젝트 일지", module="nl-conditioned-grid", type="일지",
         tags=["일지", "디버깅"],
         related=["NL-Grid - 정찰 정리"],
         summary="작업 일지(2026-05-04): Phase 1-2 진행과 API 키 이슈.",
         prefix="projlog"),
    dict(src="rl-projects/nl-conditioned-grid/results/exp_A_clear/interpretation.md", course=PROJ,
         dst="실험 A - 명확한 목표 (성공)", module="nl-conditioned-grid 실험", type="실험",
         tags=["실험", "명확한목표", "shortest-path", "성공사례"],
         related=["NL-Grid - 정찰 정리", "실험 B - Hard 제약 (성공)",
                  "실험 C - 모호한 선호도 (부분성공)", "실험 D - 목표 미명시 (제어된 실패)",
                  "실험 - 실패 분류·위험도 분석"],
         summary="명확한 좌표+선호도 → 최적 14스텝 경로 학습(성공).",
         prefix="projexpA",
         extra_images=["learning_curve.png", "policy.png"]),
    dict(src="rl-projects/nl-conditioned-grid/results/exp_B_constrained/interpretation.md", course=PROJ,
         dst="실험 B - Hard 제약 (성공)", module="nl-conditioned-grid 실험", type="실험",
         tags=["실험", "hard제약", "obstacles", "성공사례"],
         related=["실험 A - 명확한 목표 (성공)", "실험 C - 모호한 선호도 (부분성공)",
                  "실험 D - 목표 미명시 (제어된 실패)", "실험 - 실패 분류·위험도 분석"],
         summary="'절대 피해'를 hard obstacles로 매핑 → 우회 학습(성공).",
         prefix="projexpB",
         extra_images=["learning_curve.png", "policy.png"]),
    dict(src="rl-projects/nl-conditioned-grid/results/exp_C_ambiguous/interpretation.md", course=PROJ,
         dst="실험 C - 모호한 선호도 (부분성공)", module="nl-conditioned-grid 실험", type="실험",
         tags=["실험", "모호한선호도", "intention-weakening", "soft제약"],
         related=["실험 A - 명확한 목표 (성공)", "실험 B - Hard 제약 (성공)",
                  "실험 D - 목표 미명시 (제어된 실패)", "실험 - 실패 분류·위험도 분석"],
         summary="'안전하게'만으로는 위험영역이 없어 의도가 약화됨(부분성공).",
         prefix="projexpC",
         extra_images=["learning_curve.png", "policy.png"]),
    dict(src="rl-projects/nl-conditioned-grid/results/exp_D_failure/interpretation.md", course=PROJ,
         dst="실험 D - 목표 미명시 (제어된 실패)", module="nl-conditioned-grid 실험", type="실험",
         tags=["실험", "목표미명시", "controlled-failure", "clarification"],
         related=["실험 A - 명확한 목표 (성공)", "실험 B - Hard 제약 (성공)",
                  "실험 C - 모호한 선호도 (부분성공)", "실험 - 실패 분류·위험도 분석"],
         summary="목표 미명시 → goal=null로 안전하게 빌드 거부(제어된 실패).",
         prefix="projexpD"),
    dict(src="rl-projects/nl-conditioned-grid/results/failure_taxonomy.md", course=PROJ,
         dst="실험 - 실패 분류·위험도 분석", module="nl-conditioned-grid 실험", type="실험",
         tags=["실패분류", "위험도분석", "safety", "schema-v2"],
         related=["실험 A - 명확한 목표 (성공)", "실험 B - Hard 제약 (성공)",
                  "실험 C - 모호한 선호도 (부분성공)", "실험 D - 목표 미명시 (제어된 실패)",
                  "NL-Grid - 명령 스키마 v1 설계"],
         summary="5가지 실패 모드 분류와 의도약화의 위험성, v2 스키마 제안.",
         prefix="projfail"),
    dict(src="rl-projects/nl-conditioned-grid/results/parser_test_analysis.md", course=PROJ,
         dst="NL-Grid - 파서 단위 테스트 결과", module="nl-conditioned-grid", type="실험",
         tags=["파서테스트", "테스트케이스", "언어패턴"],
         related=["NL-Grid - 정찰 정리", "NL-Grid - 명령 스키마 v1 설계"],
         summary="10개 파서 테스트 전부 통과; 좌표·경계표현·goal=null 안정.",
         prefix="projparser"),
    dict(src="rl-projects/nl-conditioned-grid/schemas/design_rationale.md", course=PROJ,
         dst="NL-Grid - 명령 스키마 v1 설계", module="nl-conditioned-grid", type="프로젝트",
         tags=["스키마설계", "MDP매핑", "제약타입", "언어grounding"],
         related=["NL-Grid - 시스템 아키텍처", "NL-Grid - 정찰 정리",
                  "실험 - 실패 분류·위험도 분석"],
         summary="자연어 신호를 goal/obstacles/soft_avoid/preference로 매핑.",
         prefix="projschema"),
    dict(src="rl-projects/nl_conditioned_qlearning_spec (1).md", course=PROJ,
         dst="NL-Grid - 구현 명세서", module="nl-conditioned-grid", type="명세",
         tags=["명세", "구현가이드", "phase1", "phase2", "캡스톤"],
         related=["NL-Grid - 시스템 아키텍처", "NL-Grid - 정찰 정리",
                  "절벽걷기 - SARSA vs Q러닝 비교"],
         summary="Phase 1(절벽걷기)+Phase 2(NL-Grid) 전체 구현 마스터 명세.",
         prefix="projspec"),
]


def process_images(content, src_dir, course_dir, prefix):
    att_dir = os.path.join(VAULT, course_dir, "attachments")
    os.makedirs(att_dir, exist_ok=True)

    def repl(m):
        path = m.group(1).strip()
        if path.startswith("attachment:"):
            return "\n> [!warning] 원본 첨부 이미지 누락 (Notion attachment, 파일 없음)\n"
        rel = path
        if rel.startswith("./"):
            rel = rel[2:]
        srcimg = os.path.join(src_dir, rel.replace("/", os.sep))
        if os.path.isfile(srcimg):
            flat = rel.replace("/", "-").replace("\\", "-")
            newname = f"{prefix}-{flat}"
            shutil.copy2(srcimg, os.path.join(att_dir, newname))
            return f"![[{newname}]]"
        return f"\n> [!missing] 이미지 없음: `{path}`\n"

    return IMG_RE.sub(repl, content)


def copy_extra_images(extra, src_dir, course_dir, prefix):
    att_dir = os.path.join(VAULT, course_dir, "attachments")
    os.makedirs(att_dir, exist_ok=True)
    embeds = []
    for rel in extra:
        srcimg = os.path.join(src_dir, rel.replace("/", os.sep))
        if os.path.isfile(srcimg):
            flat = rel.replace("/", "-")
            newname = f"{prefix}-{flat}"
            shutil.copy2(srcimg, os.path.join(att_dir, newname))
            embeds.append(f"![[{newname}]]")
    return embeds


def build_frontmatter(n):
    lines = ["---"]
    lines.append(f'title: "{n["dst"]}"')
    lines.append(f'course: "{n["course"]}"')
    lines.append(f'module: "{n["module"]}"')
    lines.append(f"type: {n['type']}")
    lines.append("tags:")
    course_tag = {C1: "코스1-기초", C2: "코스2-샘플기반", C3: "코스3-함수근사",
                  C4: "코스4-캡스톤", PROJ: "프로젝트"}[n["course"]]
    lines.append(f"  - rl/{course_tag}")
    lines.append(f"  - 유형/{clean_tag(n['type'])}")
    for t in n["tags"]:
        lines.append(f"  - 개념/{clean_tag(t)}")
    lines.append("---")
    return "\n".join(lines)


def build_related(n):
    out = ["\n\n---\n\n## 🔗 관련 노트"]
    for r in n["related"]:
        out.append(f"- [[{r}]]")
    return "\n".join(out)


COURSE_TAG = {C1: "코스1-기초", C2: "코스2-샘플기반", C3: "코스3-함수근사",
              C4: "코스4-캡스톤", PROJ: "프로젝트"}

COURSE_INTRO = {
    C1: "MDP·밴딧에서 동적 프로그래밍까지, 강화학습의 **이론적 토대**를 다지는 코스.",
    C2: "환경 모델 없이 **경험만으로** 학습한다 — 몬테카를로·TD·TD제어·Dyna.",
    C3: "테이블을 벗어나 **함수로 가치·정책을 근사** — 선형근사·신경망·정책경사.",
    C4: "배운 모든 것을 종합하는 **캡스톤**. 현재 아래 개인 프로젝트로 이어지는 중.",
    PROJ: "직접 구현·실험한 RL 프로젝트 — 절벽걷기, 자연어 조건부 Q러닝.",
}

MOC_NAME = {C1: "🗂️ 코스1 지도", C2: "🗂️ 코스2 지도", C3: "🗂️ 코스3 지도",
            C4: "🗂️ 코스4 지도", PROJ: "🗂️ 프로젝트 지도"}

TYPE_BADGE = {"강의노트": "📘", "요약": "📝", "실습": "🧪", "퀴즈": "❓",
              "특강": "🎙️", "논문리뷰": "📄", "과제": "✍️", "프로젝트": "🚀",
              "실험": "🔬", "명세": "📐", "일지": "🗒️"}


def write_moc(course):
    course_notes = [n for n in NOTES if n["course"] == course]
    lines = ["---", f'title: "{MOC_NAME[course]}"', "type: MOC",
             "tags:", f"  - rl/{COURSE_TAG[course]}", "  - 유형/MOC", "---", ""]
    lines.append(f"# {course}")
    lines.append("")
    lines.append("> [!info] 코스 개요")
    lines.append("> " + COURSE_INTRO[course])
    lines.append("")
    lines.append("[[🏠 RL 학습 지도|← 홈으로]]")
    lines.append("")

    if course == C4:
        lines.append("## 현황")
        lines.append("코스 4의 강의 노트는 아직 정리 전입니다. "
                     "캡스톤의 실질적 작업은 아래 개인 프로젝트로 진행 중입니다.")
        lines.append("")
        lines.append("- [[NL-Grid - 구현 명세서]] — 캡스톤으로 이어지는 마스터 명세")
        lines.append("- [[NL-Grid - 정찰 정리]] — 자연어 조건부 Q러닝 프로젝트")
        lines.append("- [[절벽걷기 - SARSA vs Q러닝 비교]] — 알고리즘 검증 단계")
        lines.append("- [[🗂️ 프로젝트 지도]]")
        lines.append("")
        path = os.path.join(VAULT, course, MOC_NAME[course] + ".md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines).rstrip() + "\n")
        return

    # group by module preserving order
    order = []
    groups = {}
    for n in course_notes:
        if n["module"] not in groups:
            groups[n["module"]] = []
            order.append(n["module"])
        groups[n["module"]].append(n)

    lines.append("## 노트 목록")
    lines.append("")
    for mod in order:
        lines.append(f"### {mod}")
        for n in groups[mod]:
            badge = TYPE_BADGE.get(n["type"], "•")
            lines.append(f"- {badge} [[{n['dst']}]] — {n['summary']}")
        lines.append("")

    path = os.path.join(VAULT, course, MOC_NAME[course] + ".md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def write_home():
    lines = ["---", 'title: "🏠 RL 학습 지도"', "type: MOC",
             "tags:", "  - rl/홈", "  - 유형/MOC", "---", ""]
    lines += [
        "# 🏠 RL 학습 지도",
        "",
        "> [!abstract] 이 보관소(Vault)는?",
        "> Coursera **Reinforcement Learning Specialization**(4개 코스)을 수강하며 정리한 "
        "강의 노트·실습 회고·퀴즈·특강·논문 리뷰와, 직접 구현한 개인 RL 프로젝트를 모은 옵시디언 보관소입니다.",
        "",
        "## 🗺️ 학습 흐름",
        "",
        "```mermaid",
        "flowchart LR",
        "    C1[\"코스1<br/>강화학습의 기초<br/>MDP·DP\"] --> C2[\"코스2<br/>샘플기반 학습<br/>MC·TD·Dyna\"]",
        "    C2 --> C3[\"코스3<br/>함수근사<br/>근사·정책경사\"]",
        "    C3 --> C4[\"코스4<br/>캡스톤 시스템\"]",
        "    C4 --> P[\"개인 프로젝트<br/>절벽걷기·NL-Grid\"]",
        "```",
        "",
        "## 📚 코스 지도",
        "",
        f"- [[{MOC_NAME[C1]}]] — {COURSE_INTRO[C1]}",
        f"- [[{MOC_NAME[C2]}]] — {COURSE_INTRO[C2]}",
        f"- [[{MOC_NAME[C3]}]] — {COURSE_INTRO[C3]}",
        f"- [[{MOC_NAME[C4]}]] — {COURSE_INTRO[C4]}",
        f"- [[{MOC_NAME[PROJ]}]] — {COURSE_INTRO[PROJ]}",
        "",
        "## 🔖 태그로 찾기",
        "",
        "- **코스별**: `#rl/코스1-기초` `#rl/코스2-샘플기반` `#rl/코스3-함수근사` "
        "`#rl/코스4-캡스톤` `#rl/프로젝트`",
        "- **유형별**: `#유형/강의노트` `#유형/요약` `#유형/실습` `#유형/퀴즈` "
        "`#유형/특강` `#유형/논문리뷰` `#유형/실험`",
        "- **개념별**: `#개념/MDP` `#개념/TD학습` `#개념/Q러닝` `#개념/함수근사` "
        "`#개념/정책경사` `#개념/actor-critic` 등",
        "",
        "> [!tip] 그래프 뷰로 보기",
        "> 좌측 그래프 뷰(Graph view)를 열면 개념 간 연결이 한눈에 보입니다. "
        "각 노트 하단의 **🔗 관련 노트**가 연결을 만들어 줍니다.",
        "",
        "## 💻 실습 코드",
        "",
        "노트북(`.ipynb`)·파이썬 소스는 원본 레포지토리 폴더에 그대로 있습니다 "
        "(`../1. 강화학습의 기초/`, `../rl-projects/` 등). 이 보관소는 학습 노트 중심입니다.",
    ]
    path = os.path.join(VAULT, "🏠 RL 학습 지도.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main():
    if os.path.exists(VAULT):
        shutil.rmtree(VAULT)
    for d in COURSE_DIRS:
        os.makedirs(os.path.join(VAULT, d), exist_ok=True)

    for n in NOTES:
        src = os.path.join(ROOT, n["src"].replace("/", os.sep))
        src_dir = os.path.dirname(src)
        with open(src, "r", encoding="utf-8") as f:
            content = f.read()
        content = process_images(content, src_dir, n["course"], n["prefix"])
        parts = [build_frontmatter(n), "", content]
        if n.get("extra_images"):
            embeds = copy_extra_images(n["extra_images"], src_dir, n["course"], n["prefix"])
            if embeds:
                parts.append("\n\n## 📊 결과 이미지\n\n" + "\n\n".join(embeds))
        parts.append(build_related(n))
        out = "\n".join(parts).rstrip() + "\n"
        dst = os.path.join(VAULT, n["course"], n["dst"] + ".md")
        with open(dst, "w", encoding="utf-8") as f:
            f.write(out)
    for course in COURSE_DIRS:
        write_moc(course)
    write_home()
    print("wrote 5 course MOCs + home MOC")

    print("TOTAL notes:", len(NOTES))


if __name__ == "__main__":
    main()
