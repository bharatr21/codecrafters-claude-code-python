import argparse
import os
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
        return f.write(content)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]
    tool_map = {"Read": read_file, "Write": write_file}
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
