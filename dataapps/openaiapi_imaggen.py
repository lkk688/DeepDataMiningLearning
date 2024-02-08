#https://cookbook.openai.com/articles/what_is_new_with_dalle_3

from openai import OpenAI
client = OpenAI()

response = client.images.generate(
  model="dall-e-3",
  prompt="a computer system diagram for professional research papers",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
revised_prompt = response.data[0].revised_prompt
print(image_url)
#revised_prompt:
print(revised_prompt)