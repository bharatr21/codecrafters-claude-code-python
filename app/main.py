import subprocess
import argparse
import os
from pathlib import Path
import sys
import json

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


def load_skill(path: Path) -> tuple[str, str] | None:
    try:
        fm = parse_frontmatter(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Skipping {path}: {e}", file=sys.stderr)
        return None
    name, description = fm.get("name"), fm.get("description")
    if not isinstance(name, str) or not isinstance(description, str):
        print(f"Skipping {path}: missing name or description", file=sys.stderr)
        return None
    name, description = name.strip(), description.strip()
    if not name or not description:
        print(f"Skipping {path}: empty name or description", file=sys.stderr)
        return None
    return name, description


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    skills_dir = Path(".claude/skills")

    skill_files = sorted(skills_dir.rglob("SKILL.md"))
    loaded_skills = ""
    for f in skill_files:
        fm = load_skill(f)
        if fm is None:
            print(f"Could not load the skill at {f}", file=sys.stderr)
        else:
            name, description = fm
            loaded_skills += f"- {name}: {description}\n"

    system_message =  f"You have access to the following skills:\n\n{loaded_skills}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": args.p}
        ]
    tool_map = {"Read": read_file, "Write": write_file, "Bash": run_bash}
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
