# Phase 2 — Go2 locomotion 검증 (Isaac Lab 스톡 태스크, 인스턴스 위)

목표: 통합 코드 붙이기 전에 (1) Isaac Lab이 이 인스턴스에서 돌고, (2) 스톡 Unitree Go2 속도추종 정책을
짧게 학습해 체크포인트를 만들고, (3) 재생+영상 녹화가 되는지(Phase 4 디리스킹) 확인한다.
**리워드·config는 손대지 않는다.** 스톡 그대로 — 튜닝 = 디버깅 시간.

배포 IP: `100.55.171.220` · 접속: `./ssh nlq` (학습용) / `./novnc nlq` (화면으로 보기·녹화)

---

## Step 0 — 접속 + Isaac Lab 위치 확인
```bash
./ssh nlq
# 인스턴스 안에서:
ls ~
ls ~/IsaacLab 2>/dev/null || find ~ -maxdepth 2 -name isaaclab.sh
cd ~/IsaacLab          # (경로 다르면 맞춰서)
```

## Step 1 — 스모크 게이트 (공식 예제, known-good)
Isaac Lab 스택이 도는지 가장 빠르게 확인. 짧게.
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py \
  --task=Isaac-Cartpole-v0 --headless --max_iterations 20
```
완료되면 → 설치/GPU/렌더 스택 정상. 다음 단계로.

## Step 2 — Go2 태스크 이름 확인 (추측 금지, 실제 등록명 확인)
```bash
./isaaclab.sh -p scripts/environments/list_envs.py | grep -i "go2\|unitree"
```
출력에서 평지 속도추종 태스크명을 찾는다. (후보: `Isaac-Velocity-Flat-Unitree-Go2-v0`)

## Step 3 — Go2 평지 속도추종 짧게 학습 (rsl_rl, 스톡)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<Step2에서_확인한_Go2_평지_태스크> --headless --max_iterations 500
```
- A10G에서 ~10~20분. 체크포인트는 `logs/rsl_rl/<task>/<timestamp>/` 에 저장.
- 보고서용 학습 산출물(reward/학습곡선)도 여기서 나옴.

## Step 4 — 재생 + 영상 녹화 (Phase 4 디리스킹)
```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task=<같은 태스크> --num_envs 16 --headless --video --video_length 300
```
- freeze된 정책으로 걷는 영상 mp4 생성(`logs/.../videos/`). 헤드리스에서 RecordVideo로 오프스크린 렌더.
- 화면으로 직접 보고 싶으면 `--headless --video` 대신 noVNC 접속 후 `play.py`(GUI)로.

산출물(체크포인트·mp4)은 이후 S3로 회수.

---

## 폴백 / 주의
- 플래그가 거부되면 추측 말고: `./isaaclab.sh -p <script> --help` 로 그 버전의 실제 플래그 확인.
- 첫 Isaac Sim 로딩 10~15분(셰이더 캐시). 이후 빠름.
- 막히면 깊게 파지 말고: 태스크명/플래그/라이브러리(rsl_rl↔rl_games)만 바꿔보고, 안 되면 보고.
- 작업 끝 `./stop nlq`, 프로젝트 끝 `./destroy nlq`.
```
```
