#https://platform.openai.com/docs/quickstart?context=python
#pip install --upgrade openai
#Mac setup API key, source ~/.zshrc export OPENAI_API_KEY='your-api-key-here'
#Windows: setx OPENAI_API_KEY "your-api-key-here"

from openai import OpenAI
client = OpenAI()

# response = client.chat.completions.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
#     {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
#   ]
# )

# response = client.chat.completions.create(
#   model="gpt-4",
#   messages=[
#     {
#       "role": "system",
#       "content": "You will be provided with several sentences in English, and your task is to translate it into Chinese and polish the writing."
#     },
#     {
#       "role": "user",
#       "content": "Google makes some of our favorite Android phones—its Pixels have some of the best cameras, along with slick software and tons of useful smart features."
#     }
#   ],
#   temperature=0.7,
#   max_tokens=256,
#   top_p=1
# )

# response = client.chat.completions.create(
#   model="gpt-3.5-turbo-1106",
#   response_format={ "type": "json_object" },
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
#     {"role": "user", "content": "Who won the world series in 2020?"}
#   ]
# )

#Grammar correction
response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {
      "role": "system",
      "content": "You will be provided with statements, and your task is to convert them to standard English."
    },
    {
      "role": "user",
      "content": "She no went to the market."
    }
  ],
  temperature=0.7,
  max_tokens=64,
  top_p=1
)

print(response)
print(response.choices[0].message.content)


print(response.choices[0].message)
#ChatCompletionMessage(content="In the realm of code's endless rhyme,\nWhere algorithms weave space and time,\nExists a technique, both simple and grand,\nA poetic dance called recursion, hand in hand.\n\nLike a mirror reflecting its own reflection,\nA function calls itself, a cyclical connection,\nAn elegant loop, with depth and precision,\nUnfolding mysteries in code's pure vision.\n\nWith eyes unclouded, let me explain,\nRecursion's power, in rhythm and refrain.\nImagine a puzzle, disassembled, undone,\nWith recursive steps, it swiftly becomes one.\n\nJust like a Russian doll, nested so fine,\nA function calls itself, time after time.\nA small problem breaks, into fragments untold,\nRevealing a pattern, as it gracefully unfolds.\n\nFor every iteration, a recursive embrace,\nBreaking down the problem, with grace and with grace.\nEach layer peeled back, till a base case is found,\nWhere solutions lie waiting, on solid ground.\n\nLike a fractal's intricate, infinite design,\nRecursion unwraps, the code's mystical shrine.\nA web of self-repetition, reaching to the core,\nUnlocking the secrets, that lie at the code's core.\n\nBut heed this warning, with caution and care,\nRecursion can spiral, wild and unaware.\nWithout a base case, it endlessly twirls,\nA loop without end, in an infinite whirl.\n\nSo tread with awareness, embrace the recursive,\nWith clarity and logic, let your code be cohesive.\nFor recursion, my friend, is a tool to be wield,\nIn the hands of a poet, whose stories shall be revealed.", role='assistant', function_call=None, tool_calls=None)