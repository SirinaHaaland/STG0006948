from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "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"

openai = OpenAI()

prompt = "An astronaut lounging in a tropical resort in space, pixel art"
model = "dall-e-2"


def main() -> None:
    response = openai.images.generate(prompt=prompt, model=model)

    print(response)


if __name__ == "__main__":
    main()