# Setup & AWS — 시크릿 정리 + 비용 최적화 가이드

이 문서는 캡스톤(`nl-quadruped`)을 돌리기 위해 **네가 직접 준비해야 하는 정보**와 **AWS 비용 안전 수칙**을 모은 체크리스트다.
자세한 배경/결정은 메모리 `capstone-nl-quadruped` 참고.

---

## A. 준비해야 할 자격증명 (한 곳에 정리)

> ⚠️ 키는 **절대 git에 커밋하지 말 것.** `.gitignore`가 `.env`, `*.tfvars`, `*.pem`, `secrets/` 등을 막아둠.
> ⚠️ 키를 **채팅창에 붙여넣지 말 것.** 아래 파일/명령으로 로컬에만 둔다.

| # | 무엇 | 어디서 발급 | 어디에 저장 |
|---|---|---|---|
| 1 | **OpenAI API Key** | platform.openai.com → API keys | `nl-quadruped/.env` 의 `OPENAI_API_KEY` |
| 2 | **AWS IAM 액세스 키** | AWS 콘솔 → IAM → 전용 사용자 생성 후 발급 | `aws configure --profile rl-capstone` (→ `~/.aws/credentials`) |
| 3 | **EC2 SSH 키페어** | `ssh-keygen` 또는 AWS 콘솔에서 생성 | 로컬 `~/.ssh/rl-capstone.pem` (권한 400) |
| 4 | **NGC API Key** | ngc.nvidia.com → Setup → Generate API Key | Phase 2에서 EC2 안에서 사용 (Isaac Sim 컨테이너 pull) |

### 1. OpenAI 키
```powershell
Copy-Item .env.example .env   # 그리고 .env 안 OPENAI_API_KEY 채우기
```

### 2. AWS 자격증명 (전용 IAM 사용자 권장 — 최소권한)
루트 계정 키를 쓰지 말고, 이 프로젝트용 IAM 사용자를 만든다:
- 콘솔 → IAM → Users → Create user (예: `rl-capstone`)
- 권한: 우선 `AmazonEC2FullAccess` + `AmazonS3FullAccess` (프로젝트 끝나면 사용자 삭제)
- Security credentials → Create access key (Use case: CLI)
- 로컬에서:
```powershell
aws configure --profile rl-capstone
# Access Key ID / Secret / region=us-east-1 / output=json
```
Terraform은 이 프로필을 자동으로 읽는다. (키 파일을 코드에 안 넣음)

### 3. SSH 키페어
```bash
ssh-keygen -t ed25519 -f ~/.ssh/rl-capstone -C "rl-capstone"
```
공개키(`~/.ssh/rl-capstone.pub`)를 Terraform이 EC2에 등록하게 한다.

---

## B. AWS GPU 쿼터 증액 — ⏰ 지금 걸어둬야 하는 long-pole

신규/소규모 계정은 GPU 인스턴스 쿼터가 **0**이라 그냥 띄우면 실패한다. 승인까지 **수 시간~1일** 걸리므로 **제일 먼저** 요청한다.

1. AWS 콘솔 → **Service Quotas** → AWS services → **Amazon EC2**
2. 검색: **"Running On-Demand G and VT instances"**
3. Request increase → 값에 **8** 이상 입력 (g5.2xlarge = vCPU 8개)
4. 리전이 **us-east-1** 인지 확인 후 제출
5. (Spot도 쓸 거면) **"All G and VT Spot Instance Requests"** 도 같은 방식으로 8 이상 요청

> 사유 칸: "Running NVIDIA Isaac Sim for a reinforcement learning capstone project, single GPU instance." 정도면 충분.

---

## C. 비용 최적화 수칙 (예상 총액 $15~40)

| 항목 | 선택 | 비고 |
|---|---|---|
| 인스턴스 | `g5.2xlarge` (A10G 24GB) ~$1.2/hr 온디맨드 | 학습+렌더링 적정. g4dn.xlarge(~$0.53/hr)는 예비 |
| 가격 모델 | **처음 온디맨드 → 익숙해지면 Spot(~$0.45/hr)** | Spot은 중단 가능 → S3 체크포인트 갖춘 뒤 전환 |
| 켜고 끄기 | Terraform `apply`/`destroy` | 작업할 때만 과금 |
| 🔑 자동 종료 | user-data에 **유휴 자동 stop** 스크립트 | 깜빡 켜둠 = 하룻밤 $30 방지 |
| AMI | Deep Learning AMI (드라이버 선설치, 무료) | 유료 GPU 시간을 드라이버 설치에 안 씀 |
| 스토리지 | EBS gp3 ~150GB ($0.08/GB·月) | 프로젝트 끝나면 destroy |
| 산출물 | S3 표준 ($0.023/GB·月) | mp4/체크포인트 보관, 月 몇 센트 |
| 리전 | `us-east-1` | 최저가 + GPU 재고 |

**황금률**: GPU 없이 되는 일(파싱·플래닝·컨트롤러·2D 목업·보고서)은 전부 로컬에서. GPU는 Isaac 학습+렌더링에만 잠깐.
