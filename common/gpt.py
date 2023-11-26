import openai, g4f

openai.api_key = "sk-47lztbvTmb43q1lVanVebwcFtG9R3nGYb1eeZUFXaci4XEe0"
openai.api_base = "https://api.chatanywhere.com.cn/v1"
proxy = "http://127.0.0.1:7890"


def anychat_gpt_35(messages: list):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return completion.choices[0].message.content


def anychat_gpt_35_stream(messages: list):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    completion = {"role": "", "content": ""}
    for event in response:
        if event["choices"][0]["finish_reason"] == "stop":
            break
        for delta_k, delta_v in event["choices"][0]["delta"].items():
            if delta_k == "content":
                print(delta_v, flush=True, end="")
            completion[delta_k] += delta_v
    print()
    return completion["content"]


def g4f_gpt_4(messages: list, stream=True):
    response = g4f.ChatCompletion.create(
        model=g4f.models.gpt_4, provider=g4f.Provider.Bing, messages=messages, stream=stream, proxy=proxy
    )
    if stream:
        for message in response:
            print(message, flush=True, end="")
        print()
    else:
        return response


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "鲁迅和周树人的关系"},
    ]
    g4f_gpt_4(messages, stream=True)
