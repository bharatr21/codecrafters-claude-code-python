import subprocess
import argparse
import os
from pathlib import Path
import sys
import json
import re

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

def read_file(file_path: str) -> str:
    with open(file_path, encoding="utf-8") as f:
        return f.read()


def write_file(file_path: str, content: str):
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, encoding="utf-8", mode="w") as f:
        return str(f.write(content))

def run_bash(command: str):
    try:
        proc = subprocess.run(
            command,
            shell=True,           # interpret the string with /bin/sh
            capture_output=True,  # collect stdout/stderr instead of printing them
            text=True,            # decode to str instead of bytes
            timeout=120,          # don't let a hung command stall the agent
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after 120s"

    output = proc.stdout
    if proc.stderr:
        output += f"\n[stderr]\n{proc.stderr}"
    output += f"\n[exit code: {proc.returncode}]"
    return output


def parse_frontmatter(text: str) -> dict:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    fm = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fm
        key, sep, value = line.partition(":")
        if sep and not line.startswith((" ", "\t")):
            fm[key.strip()] = value.strip().strip("\"'")
    return {}


class SkillError(Exception):
    pass


class UnknownSkillError(SkillError):
    """The user asked for a skill that doesn't exist."""


class InvalidSkillError(SkillError):
    """The skill exists but its SKILL.md is broken."""


def load_skill(path: Path) -> tuple[str, str]:
    try:
        fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise InvalidSkillError(f"{path}: {e}") from e
    name, description = fm.get("name"), fm.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        raise InvalidSkillError(f"{path}: missing name or description")
    name, description = name.strip(), description.strip()
    if not name or not description:
        raise InvalidSkillError(f"{path}: empty name or description")
    if name != path.parent.name:
        raise InvalidSkillError(f"{path}: name '{name}' does not match directory '{path.parent.name}'")
    return name, description


def resolve_skill(name: str, skills: dict[str, Path]) -> Path:
    if name not in skills:
        raise UnknownSkillError(f"Unknown skill: /{name}")
    return skills[name]


# \$ is a literal $, $ARGUMENTS is the whole argument string,
# $ARGUMENTS[N] and $N are single arguments (zero-based, \d+ so $10 is argument 10)
ARGUMENTS_PATTERN = re.compile(r"\\\$|\$ARGUMENTS(?:\[(?P<index>\d+)\])?|\$(?P<short>\d+)")


def expand_arguments(body: str, raw_args: str, skill_args: list[str]) -> str:
    def replace(match: re.Match) -> str:
        if match.group(0) == "\\$":
            return "$"
        index = match.group("index") or match.group("short")
        if index is None:
            return raw_args
        i = int(index)
        return skill_args[i] if i < len(skill_args) else ""

    return ARGUMENTS_PATTERN.sub(replace, body)


def uses_arguments(body: str) -> bool:
    return any(m.group(0) != "\\$" for m in ARGUMENTS_PATTERN.finditer(body))


def skill_body(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1:]).strip()
    return "\n".join(lines).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    skills_dir = Path(".claude/skills")

    skill_files = sorted(skills_dir.glob("*/SKILL.md"))
    skills: dict[str, Path] = {}
    loaded_skills = ""
    for f in skill_files:
        try:
            name, description = load_skill(f)
        except InvalidSkillError as e:
            print(f"Skipping {e}", file=sys.stderr)
            print(f"Could not load the skill at {f}", file=sys.stderr)
            continue
        skills[name] = f
        loaded_skills += f"- {name}: {description}\n"

    system_message =  (
        f"You have access to the following skills:\n\n{loaded_skills}\n"
        "When a skill is relevant, call the Skill tool with its name to load its full instructions."
    )

    messages = [{"role": "system", "content": system_message}]
    prompt = args.p.strip()
    if prompt.startswith("/"):
        parts = prompt.split(maxsplit=1)
        command = parts[0][1:]
        raw_args = parts[1] if len(parts) > 1 else ""
        skill_args: list[str] = raw_args.split()
        if not command:
            raise NotImplementedError("'/' on its own is not a command")
        body = skill_body(resolve_skill(command, skills))
        messages.append({"role": "user", "content": expand_arguments(body, raw_args, skill_args)})
        if raw_args and not uses_arguments(body):
            messages.append({"role": "user", "content": raw_args})
    else:
        messages.append({"role": "user", "content": prompt})

    def run_skill(name: str) -> str:
        return skill_body(resolve_skill(name, skills))

    tool_map = {"Read": read_file, "Write": write_file, "Bash": run_bash, "Skill": run_skill}
    tools= [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "properties": {
                        "file_path": {
                        "type": "string",
                        "description": "The path to the file to read"
                        }
                    },
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                        "properties": {
                            "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to"
                            },
                            "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                            }
                        }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                    "type": "string",
                    "description": "The command to execute"
                    }
                }
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": "Load the full instructions of one of the available skills",
                "parameters": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the skill to load"
                        }
                    }
                }
            }
        },
    ]

    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        # You can use print statements as follows for debugging, they'll be visible when running tests.
        # print("Logs from your program will appear here!", file=sys.stderr)

        rmsg = chat.choices[0].message
        
        if rmsg.tool_calls:
            messages.append(rmsg)
            for call in rmsg.tool_calls:
                function = call.function
                tool_args = json.loads(function.arguments or "{}")
                tool_name = function.name
                try:
                    result = tool_map[tool_name](**tool_args)
                except Exception as e:
                    result=f"Error: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })

        else:
            print(rmsg.content)
            break


if __name__ == "__main__":
    main()
