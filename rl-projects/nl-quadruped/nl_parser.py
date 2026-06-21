"""자연어 명령 -> command_schema_v2 JSON spec.

nl-conditioned-grid/nl_parser.py 의 패턴을 그대로 계승하되,
- v2 프롬프트/스키마(연속 월드, 미터)를 사용하고
- python-dotenv 로 .env 의 OPENAI_API_KEY 를 자동 로드한다(설치 안 돼 있으면 조용히 건너뜀).
"""
import json
import os
from pathlib import Path

from jsonschema import validate

BASE_DIR = Path(__file__).resolve().parent

# .env 자동 로드 (선택 의존성 — 없으면 환경변수에 의존)
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


def _extract_json(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]


# response_format=json_object 지원 여부 캐시 (None=미탐지). 게이트웨이 1회 탐지 후 재사용.
_JSON_MODE_SUPPORTED = None


def _chat(client, model, messages):
    """채팅 호출. NLQ_PARSER_JSON_MODE 로 json_object 모드 제어.

    - on:   항상 response_format 사용
    - off:  사용 안 함 (Anthropic 계열 게이트웨이 권장 — 불필요한 탐지 호출 제거)
    - auto: 첫 호출에서 한 번만 탐지하고 결과를 프로세스 내 캐시 (기본값)
    """
    global _JSON_MODE_SUPPORTED
    mode = os.getenv("NLQ_PARSER_JSON_MODE", "auto").lower()
    if mode == "off":
        use_json = False
    elif mode == "on":
        use_json = True
    else:
        use_json = True if _JSON_MODE_SUPPORTED is None else _JSON_MODE_SUPPORTED

    kwargs = {"model": model, "temperature": 0, "messages": messages}
    if use_json:
        kwargs["response_format"] = {"type": "json_object"}
    try:
        resp = client.chat.completions.create(**kwargs)
        if mode == "auto" and use_json:
            _JSON_MODE_SUPPORTED = True
        return resp
    except Exception:
        if not use_json:
            raise  # json 모드도 아닌데 실패 -> 인증/네트워크 등 진짜 오류
        _JSON_MODE_SUPPORTED = False  # response_format 미지원으로 추정, 캐시 후 제거 재시도
        kwargs.pop("response_format", None)
        return client.chat.completions.create(**kwargs)


def load_schema() -> dict:
    return json.loads((BASE_DIR / "schemas" / "command_schema_v2.json").read_text(encoding="utf-8"))


def validate_spec(spec: dict) -> dict:
    """스키마 검증만 수행(LLM 호출 없음). 오프라인 spec 테스트에 사용."""
    validate(instance=spec, schema=load_schema())
    return spec


def parse_command(command: str, model: str | None = None) -> dict:
    """
    자연어 명령을 schema_v2 호환 spec(dict)으로 변환.

    Raises:
        ValueError: JSON 파싱 실패 또는 스키마 위반.
    """
    from openai import OpenAI  # 지연 import: 오프라인 경로에서는 불필요

    model = model or os.getenv("NLQ_PARSER_MODEL", "gpt-4o")
    client = OpenAI()
    prompt_template = (BASE_DIR / "prompts" / "parser_prompt_v2.txt").read_text(encoding="utf-8")
    full_prompt = prompt_template.replace("{COMMAND}", command)

    response = _chat(client, model, [{"role": "user", "content": full_prompt}])
    raw_text = response.choices[0].message.content.strip()

    try:
        spec = json.loads(_extract_json(raw_text))
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON.\nRaw: {raw_text}\nError: {e}") from e

    return validate_spec(spec)


if __name__ == "__main__":
    import sys

    cmd = " ".join(sys.argv[1:]) or "오른쪽 위 구석으로 가되 중앙은 절대 피해서 천천히"
    spec = parse_command(cmd)
    print(json.dumps(spec, ensure_ascii=False, indent=2))
