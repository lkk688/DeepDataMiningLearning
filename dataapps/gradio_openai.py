from openai import OpenAI
import gradio as gr

client = OpenAI()
#openai.api_key = "sk-..."  # Replace with your key

def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = client.chat.completions.create(
        model='gpt-3.5-turbo-1106',
        #response_format={ "type": "json_object" },
        messages= history_openai_format,
        temperature=1.0,
        #stream=True
    )
    #only works when stream is not set to True
    #print(response.choices[0].message.content)
    yield response.choices[0].message.content

    # partial_message = ""
    # for chunk in response:
    #     #print(chunk.choices[0].message.content)
        
    #     if len(chunk.choices[0].message) != 0:
    #         partial_message = partial_message + chunk.choices[0].message.content #chunk['choices'][0]['delta']['content']
            
    #         yield partial_message
# gr.ChatInterface.get_component('textbox')
# show_label
# show_copy_button
gr.ChatInterface(predict).queue().launch()
#demo = gr.ChatInterface(fn=echo, examples=["hello", "hola", "merhaba"], title="Echo Bot")