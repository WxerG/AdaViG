"""Extract explicit final answers without a model or external service."""

import re


def extract_answer(text: str) -> str | None:
    tags = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if tags and tags[-1].strip():
        return tags[-1].strip()
    start = text.rfind("\\boxed{")
    if start >= 0:
        start += len("\\boxed{")
        depth = 1
        for index in range(start, len(text)):
            depth += (text[index] == "{") - (text[index] == "}")
            if depth == 0:
                return text[start:index].strip() or None
    answers = re.findall(
        r"(?:Final Answer|Answer|最终答案|答案)\s*[:：]\s*([^\n]+)", text, re.IGNORECASE
    )
    return answers[-1].strip() if answers else None
