import ollama
import json

stream = ollama.chat(
    model='gemma3:4b',
    messages=[{'role': 'user', 'content': '인공지능의 미래에 대해 에세이를 써줘.'}],
    stream=True,
)

for chunk in stream:
    # print(chunk['message']['content'], end='', flush=True)
    print(json.dumps(chunk.model_dump(), indent=4, ensure_ascii=False))

    last_chunk = chunk
else:
    print("\n[Stream Ended]")
    print(last_chunk.model_dump())

