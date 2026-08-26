import ollama

response = ollama.generate(
    model='gemma3:4b',
    prompt='왜 하늘은 파란색이야? 20자 이내로 답변해줘.',
)
# print(response['response'])

# print(type(response))
print(response.response)
print(response.model_dump())