import openai

openai.api_key = "sk-47lztbvTmb43q1lVanVebwcFtG9R3nGYb1eeZUFXaci4XEe0"
proxy = "http://127.0.0.1:7890"


def gpt_infer(messages: list, model="gpt-3.5-turbo"):
    completion = openai.ChatCompletion.create(model=model, messages=messages, proxy=proxy)
    return completion.choices[0].message.content  # type: ignore


if __name__ == "__main__":
    messages = [
        {"role": "user", "content": "鲁迅和周树人的关系"},
    ]
    print(gpt_infer(messages))
