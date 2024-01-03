from openai import OpenAI
#import gradio as gr

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

#https://www.gradio.app/guides/creating-a-custom-chatbot-with-blocks
from gradio import Blocks
import gradio as gr

class ChatInterface(Blocks):
    def __init__(self, fn):
        
        super().__init__(mode="chat_interface")        
        self.fn = fn
        self.history = []                    
        
        with self:
            self.chatbot = gr.Chatbot(label="Input")
            self.textbox = gr.Textbox()
            self.stored_history = gr.State()
            self.stored_input = gr.State()
            
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                delete_btn = gr.Button("Delete Previous")
                retry_btn = gr.Button("Retry")
                clear_btn = gr.Button("Clear")
                
            # Invisible elements only used to set up the API
            api_btn = gr.Button(visible=False)
            api_output_textbox = gr.Textbox(visible=False, label="output")
            
            self.buttons = [submit_btn, retry_btn, clear_btn]

            self.textbox.submit(
                self.clear_and_save_textbox, [self.textbox], [self.textbox, self.stored_input], api_name=False, queue=False,
            ).then(
                self.submit_fn, [self.chatbot, self.stored_input], [self.chatbot], api_name=False
            )
            
            submit_btn.click(self.submit_fn, [self.chatbot, self.textbox], [self.chatbot, self.textbox], api_name=False)
            delete_btn.click(self.delete_prev_fn, [self.chatbot], [self.chatbot, self.stored_input], queue=False, api_name=False)
            retry_btn.click(self.delete_prev_fn, [self.chatbot], [self.chatbot, self.stored_input], queue=False, api_name=False).success(self.retry_fn, [self.chatbot, self.stored_input], [self.chatbot], api_name=False)
            api_btn.click(self.submit_fn, [self.stored_history, self.textbox], [self.stored_history, api_output_textbox], api_name="chat")
            clear_btn.click(lambda :[], None, self.chatbot, api_name="clear")          
    
    def clear_and_save_textbox(self, inp):
        return "", inp
        
    def disable_button(self):
        # Need to implement in the event handlers above
        return gr.Button.update(interactive=False)
        
    def enable_button(self):
        # Need to implement in the event handlers above
        return gr.Button.update(interactive=True)
                
    def submit_fn(self, history, inp):
        # Need to handle streaming case
        out = self.fn(history, inp)
        history.append((inp, out))
        return history
    
    def delete_prev_fn(self, history):
        try:
            inp, _ = history.pop()
        except IndexError:
            inp = None
        return history, inp

    def retry_fn(self, history, inp):
        if inp is not None:
            out = self.fn(history, inp)
            history.append((inp, out))
        return history
    
#ChatInterface(lambda x,y:y).launch()
ChatInterface(predict).launch()