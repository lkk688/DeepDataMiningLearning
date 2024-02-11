#https://cookbook.openai.com/articles/what_is_new_with_dalle_3

from openai import OpenAI
client = OpenAI()
#"a computer system diagram for professional research papers"
#"1024x1024"
response = client.images.generate(
  model="dall-e-3",
  prompt="Create an image that captures the spirit of Chinese New Year. The background should be a vibrant Chinese red, leaving some open spaces. A majestic Chinese dragon should be prominently featured, dancing and swirling amidst the scenery with joy and vitality. The words 'Happy Chinese New Year!', should be accurate and placed strategically within the image. The overall atmosphere should be filled with festive cheer, welcoming the new year with hope, peace, and prosperity.",
  size="1024x1024",
  quality="standard",
  n=1,
)

image_url = response.data[0].url
revised_prompt = response.data[0].revised_prompt
print(image_url)
#revised_prompt:
print(revised_prompt)