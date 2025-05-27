from openai import OpenAI

client = OpenAI(api_key="sk-melrrwtxmdlxvuezhqcijmsapmmhbejtvemnbxepafaxheal", 
                base_url="https://api.siliconflow.cn/v1")
response = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-R1',
    # model="Qwen/Qwen2.5-72B-Instruct",
    messages=[
        {'role': 'user', 
        'content': "出一套强化学习的试卷"}
    ],
    stream=True
)

for chunk in response:
    if not chunk.choices:
        continue
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)