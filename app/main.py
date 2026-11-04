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

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    messages = [{"role": "user", "content": args.p}]
    tool_map = {"Read": read_file}
    tools= [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                        "type": "string",
                        "description": "The path to the file to read"
                        }
                    },
                    "required": ["file_path"]
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
        print(f"Messages array: {messages}", file=sys.stderr)
        print(f"Tool calls: {chat.choices[0].message.tool_calls}", file=sys.stderr)
        
        if chat.choices[0].message.tool_calls:
            for call in chat.choices[0].message.tool_calls:
                function = call.function
                args = json.loads(function.arguments)
                tool_name = function.name
                result = tool_map[tool_name](**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })
            print("Agent loop continues!", file=sys.stderr)

        else:
            print(chat.choices[0].message.content)
            print("Agent loop terminated!", file=sys.stderr)
            break


if __name__ == "__main__":
    main()
