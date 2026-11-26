import subprocess
import argparse
import os
from pathlib import Path
from typing import Callable
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


def placeholder_value(match: re.Match, raw_args: str, skill_args: list[str]) -> str | None:
    """The value a placeholder stands for, or None if it has none and stays literal."""
    index = match.group("index") or match.group("short")
    if index is None:
        return raw_args
    if not skill_args:
        return ""  # with zero arguments every placeholder is empty
    i = int(index)
    return skill_args[i] if i < len(skill_args) else None


def expand_arguments(body: str, raw_args: str, skill_args: list[str]) -> str:
    def replace(match: re.Match) -> str:
        if match.group(0) == "\\$":
            return "$"
        value = placeholder_value(match, raw_args, skill_args)
        return match.group(0) if value is None else value

    return ARGUMENTS_PATTERN.sub(replace, body)


def uses_arguments(body: str, raw_args: str, skill_args: list[str]) -> bool:
    return any(
        m.group(0) != "\\$" and placeholder_value(m, raw_args, skill_args) is not None
        for m in ARGUMENTS_PATTERN.finditer(body)
    )


def skill_body(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip() == "---":
        for i, line in enumerate(lines[1:], start=1):
            if line.strip() == "---":
                return "\n".join(lines[i + 1:]).strip()
    return "\n".join(lines).strip()


def with_skill_dir(path: Path, text: str) -> str:
    """Prefix a skill's text with its base directory so relative paths in it resolve."""
    text = f"Skill: {path.parent.name} (Base directory for this skill is located at {path.parent})\nPaths in the instructions below are relative to that folder.\n{text}"
    return text

def skill_context(path: Path) -> str:
    """The skill's `context` frontmatter value ("fork" runs it in a subagent), read at invocation time."""
    return parse_frontmatter(path.read_text(encoding="utf-8")).get("context", "").strip()


def render_skill(path: Path, raw_args: str = "") -> str:
    """A skill's full text as the model sees it: arguments expanded, under its directory header."""
    body = skill_body(path)
    skill_args = raw_args.split()
    content = expand_arguments(body, raw_args, skill_args)
    if not uses_arguments(body, raw_args, skill_args) and raw_args:
        content += f"\n\nARGUMENTS: {raw_args}"
    # header goes on after expansion so the path itself is never scanned for $N
    return with_skill_dir(path, content)


def expand_prompt(
    prompt: str,
    skills: dict[str, Path],
    render: Callable[[Path, str], str] = render_skill,
) -> list[dict]:
    """Turn a prompt into user messages, expanding any leading run of /skill tokens."""
    prompt = prompt.strip()
    if not prompt.startswith("/"):
        return [{"role": "user", "content": prompt}]

    names: list[str] = []
    raw_args = ""
    for token in re.finditer(r"\S+", prompt):
        name = token.group().removeprefix("/")
        if not token.group().startswith("/") or name not in skills:
            raw_args = prompt[token.start():]
            break
        if name not in names:  # a repeated skill continues the run but is expanded once
            names.append(name)
    if not names:
        # The first token starts with / but names no skill
        first = prompt.split()[0]
        if first == "/":
            raise NotImplementedError("'/' on its own is not a command")
        resolve_skill(first[1:], skills)  # raises UnknownSkillError
    
    return [{"role": "user", "content": render(skills[name], raw_args)} for name in names]


def run_agent(client: OpenAI, messages: list, tools: list[dict], tool_map: dict) -> str:
    """Run the tool-calling loop on `messages` until the model answers, and return that answer."""
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-5.5",
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
            return rmsg.content or ""


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
        skills[name] = f.resolve()  # absolute, so the header is right whatever Bash's cwd
        loaded_skills += f"- {name}: {description}\n"

    system_message =  (
        f"You have access to the following skills:\n\n{loaded_skills}\n"
        "When a skill is relevant, call the Skill tool with its name to load its full instructions."
    )

    tool_map = {"Read": read_file, "Write": write_file, "Bash": run_bash}
    tools: list[dict] = [
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
                        },
                        "arguments": {
                            "type": "string",
                            "description": "Optional arguments for the skill, as the user would type them after /name"
                        }
                    }
                }
            }
        },
    ]

    def invoke_skill(path: Path, raw_args: str = "") -> str:
        """Render a skill, or run it in a subagent and return the subagent's answer if it forks."""
        content = render_skill(path, raw_args)
        if skill_context(path) != "fork":
            return content
        sub_tools = [t for t in tools if "function" in t and t["function"]["name"] != "Skill"]
        sub_tool_map = {t: f for t, f in tool_map.items() if t != "Skill"}
        sub_messages = [
            {
                "role": "system",
                "content": "You are a subagent for Claude Code, Anthropic's official CLI for Claude. Given the user's message, use the tools available to complete the task. Do what has been asked; nothing more, nothing less. When you complete the task, respond with a detailed writeup, since this is the only output that the caller will see.\n\nNotes: in your final response, share relevant file names and code snippets. Any file paths you return must be absolute."
            },
            {
                "role": "user",
                "content": content
            }
        ]
        sub_content = run_agent(client, sub_messages, sub_tools, sub_tool_map)
        return sub_content

    def run_skill(name: str, arguments: str = "") -> str:
        return invoke_skill(resolve_skill(name, skills), arguments)

    tool_map["Skill"] = run_skill
    messages = [{"role": "system", "content": system_message}]
    messages += expand_prompt(args.p, skills, invoke_skill)

    print(run_agent(client, messages, tools, tool_map))

if __name__ == "__main__":
    main()
