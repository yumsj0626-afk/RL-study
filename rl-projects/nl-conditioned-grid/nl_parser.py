import json
from pathlib import Path

from jsonschema import validate
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent


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


def parse_command(command: str, model: str = "gpt-4o") -> dict:
    """
    Convert a natural language navigation command into a structured JSON spec.

    Returns:
        dict: schema_v1-compatible spec.
    Raises:
        ValueError: JSON parsing failure or schema violation.
    """
    client = OpenAI()
    prompt_template = (BASE_DIR / "prompts" / "parser_prompt_v1.txt").read_text(encoding="utf-8")
    full_prompt = prompt_template.replace("{COMMAND}", command)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": full_prompt}],
    )
    raw_text = response.choices[0].message.content.strip()

    try:
        spec = json.loads(_extract_json(raw_text))
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON.\nRaw: {raw_text}\nError: {e}") from e

    schema = json.loads((BASE_DIR / "schemas" / "command_schema_v1.json").read_text(encoding="utf-8"))
    validate(instance=spec, schema=schema)
    return spec
