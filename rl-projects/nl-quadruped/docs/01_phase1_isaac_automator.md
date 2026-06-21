# Phase 1 — Isaac Automator로 AWS에 Isaac Sim+Lab 배포 (런북)

NVIDIA 공식 도구 [Isaac Automator](https://github.com/isaac-sim/IsaacAutomator)로 GPU 인스턴스 프로비저닝 +
Isaac Sim + Isaac Lab 설치를 한 번에. 직접 Terraform 작성 대신 공식 도구 사용(시간 절약·트러블슈팅 최소화).
Automator가 내부적으로 Terraform+Ansible을 돌린다.

> ⚠️ 모든 명령은 **WSL(Ubuntu) 안에서** 실행한다. (`./build` 등은 bash 스크립트라 Windows cmd/PowerShell이 아닌
> WSL에서 돌려야 함. Docker Desktop의 WSL2 통합을 켜두면 WSL 안에서 docker가 그대로 동작.)

---

## 0. 사전 준비물 체크리스트

| # | 항목 | 상태/방법 |
|---|---|---|
| 1 | **Docker Desktop** (WSL2 백엔드) | 설치 필요. https://docs.docker.com/desktop/install/windows-install/ → 설치 후 Settings → Resources → WSL integration 켜기 |
| 2 | **WSL(Ubuntu)** | 이미 있음(`wsl` 확인됨). 없으면 `wsl --install -d Ubuntu` |
| 3 | **NGC API Key** | https://ngc.nvidia.com → 가입 → Setup → Generate API Key. 배포 중 프롬프트에서 입력 |
| 4 | **AWS 자격증명** | `aws configure --profile rl-capstone` 완료됨. ※ Automator가 IAM Identity Center(SSO)를 요구하면 `aws configure sso` 필요할 수 있음(아래 주의) |
| 5 | **GPU 쿼터** | 증액 요청 완료. 배포 전 승인됐는지 콘솔에서 확인 |
| 6 | **IAM 권한** | `AmazonEC2FullAccess` 필요 |

---

## 1. 배포 (WSL 안에서)

```bash
# WSL Ubuntu 터미널
git clone https://github.com/isaac-sim/IsaacAutomator.git
cd IsaacAutomator

# 1) automator 컨테이너 빌드
./build
#   (Windows에서 ./build 가 안 되면:  docker build --platform linux/x86_64 -t isaac_automator . )

# 2) AWS 배포 — 비용 핀: g5.2xlarge(A10G 24GB, ~$1.2/hr), us-east-1
#    버전은 Automator 기본값 사용(Sim/Lab 매칭-테스트됨). 깨지면 그때 핀.
./deploy-aws --instance-type g5.2xlarge --region us-east-1
#   이후 프롬프트: deployment 이름, NGC API key, AWS 자격증명 등 입력
```

**왜 버전 기본값인가:** Automator는 자기 기본 Sim/Lab 버전 조합으로 테스트되어 있어, 임의로 옛 버전을 핀하면
오히려 호환성 디버깅이 생긴다. 기본값으로 먼저 가고, 문제가 생기면 `--isaaclab <ver>`로 조정.

**왜 g6e.2xlarge(기본) 대신 g5.2xlarge:** g6e는 L40S로 ~$2.5/hr. 우리 PoC(단일 로봇 보행+렌더)엔 A10G 24GB로 충분.
메모리 부족 등 문제가 생기면 g6e.2xlarge로 승급.

---

## 2. 접속

```bash
./novnc <deployment-name>     # 브라우저 VNC (Isaac Sim GUI를 화면으로 봄 — 시연·녹화에 유용)
./ssh   <deployment-name>     # 터미널
# 접속 정보: state/<deployment-name>/info.txt
```

---

## 3. 비용 통제 (중요)

```bash
./stop    <deployment-name>   # 작업 끝나면 매번. 컴퓨팅 과금 중단(IP는 보존, EBS는 소액 과금 지속)
./start   <deployment-name>   # 다시 작업할 때
./destroy <deployment-name>   # 프로젝트 종료 시. 과금 0
```

황금률: **세션 끝 = `./stop`**, **프로젝트 끝 = `./destroy`**. 켜둔 채 자면 하룻밤 ~$30.

---

## 4. Phase 2 진입 전 검증 게이트 (스톡 태스크로 먼저 확인)

통합 코드를 붙이기 전에, 깡통 Isaac Lab이 이 인스턴스에서 학습되는지부터 확인한다(막히면 여기서 알게 됨).

```bash
# 인스턴스 접속 후, Isaac Lab 디렉터리에서
./isaaclab.sh -p scripts/reinforcement_learning/rl_games/train.py --task=Isaac-Cartpole-v0 --headless
```

통과하면 사족보행 스톡 태스크로 진행 (정확한 태스크명은 Phase 2에서 설치된 버전 기준 확인):
- 후보: `Isaac-Velocity-Flat-Unitree-Go2-v0` (평지 속도추종) 및 `--video` 녹화 플래그
- 짧게 1회 학습 → 체크포인트 저장 → freeze → 시연에 사용

---

## 주의 / 폴백

- **AWS 인증**: Automator README는 SSO(`aws sso login`)를 언급한다. access-key 프로필(`rl-capstone`)로 안 되면
  `aws configure sso`로 전환. (Terraform AWS provider는 보통 `~/.aws`를 읽으므로 access-key도 동작할 가능성 높음)
- **첫 Isaac Sim 로딩 10~15분**(셰이더 캐시). 캐시 마운트되면 이후 1분 미만.
- **헤드리스 영상**: 클라우드 문서엔 영상 녹화가 명시 안 됨. noVNC 화면 녹화 또는 Isaac Lab `--video`(gymnasium
  RecordVideo)로 처리 — Phase 4에서 확정.
- 막히면 깊게 파지 말고 폴백(인스턴스 승급 / 버전 핀 / 접속 방식 변경)으로 후퇴. 시간 우선.
