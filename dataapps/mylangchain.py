import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate #https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html
#pip install -qU langchain-openai
from langchain_openai import ChatOpenAI

from typing import List

from fastapi import FastAPI
from langserve import add_routes

os.environ["OPENAI_API_KEY"] = getpass.getpass()

def llm_test():
    # Use the OpenAI LLM wrapper and text-davinci-003 model
    from langchain.llms import OpenAI
    # Track OpenAI token usage for a single API call
    from langchain.callbacks import get_openai_callback

    llm = OpenAI(model_name="text-davinci-003")#, openai_api_key=openai_api_key)
    # Generate a simple text response
    llm("Where is San Jose, CA?")
    
    # Show the generation output instead
    llm_result = llm.generate(["Where is San Jose, CA?", "Where is Santa Clara?"])
    llm_result.llm_output
    
    #track OpenAI token usage
    with get_openai_callback() as cb:
        result = llm("Where is San Jose, CA?")

        print(f"Total Tokens: {cb.total_tokens}")
        print(f"\tPrompt Tokens: {cb.prompt_tokens}")
        print(f"\tCompletion Tokens: {cb.completion_tokens}")
        print(f"Total Cost (USD): ${cb.total_cost}")

def get_template():
    #PromptTemplates are a concept in LangChain designed to assist with this transformation. They take in raw user input and return data (a prompt) that is ready to pass into a language model.
    system_template = "Translate the following into {language}:"
    #create the PromptTemplate. This will be a combination of the system_template as well as a simpler template for where to put the text to be translated
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", system_template), ("user", "{text}")]
    )
    #The input to this prompt template is a dictionary.
    result = prompt_template.invoke({"language": "Chinese", "text": "Good morning"})
    #it returns a ChatPromptValue that consists of two messages. If we want to access the messages directly we do
    print(result.to_messages())
    return prompt_template

def get_parser():
    #the response from the model is an AIMessage. This contains a string response along with other metadata about the response.
    #We can parse out just this response by using a simple output parser.
    parser = StrOutputParser()
    return parser

def test_chat(model):
    messages = [
        SystemMessage(content="Translate the following from English into Chinese"),
        HumanMessage(content="Hello!"),
    ]
    
    model.invoke([HumanMessage(content="Hi! I'm Bob")])

    #call the model, we can pass in a list of messages to the .invoke method
    result = model.invoke(messages)

    parser = get_parser()
    output = parser.invoke(result)
    print(output)

    #We can easily create the chain using the | operator. The | operator is used in LangChain to combine two elements together.
    chain = model | parser
    output = chain.invoke(messages)
    print(output)

    prompt_template = get_template()

    #We can now combine this with the model and the output parser from above using the pipe (|) operator:
    chain = prompt_template | model | parser
    output = chain.invoke({"language": "italian", "text": "hi"})
    print(output)

def run_langchainserver(model):
    prompt_template = get_template()
    
    parser = get_parser()
    
    # 4. Create chain
    chain = prompt_template | model | parser
    
    # 4. App definition
    app = FastAPI(
        title="LangChain Server",
        version="1.0",
        description="A simple API server using LangChain's Runnable interfaces",
    )
    
    # 5. Adding chain route
    add_routes(
        app,
        chain,
        path="/chain",
    )
    return app

#https://python.langchain.com/v0.2/docs/tutorials/llm_chain/
def langserve_client():
    from langserve import RemoteRunnable

    remote_chain = RemoteRunnable("http://localhost:8000/chain/")
    remote_chain.invoke({"language": "italian", "text": "hi"})

if __name__ == "__main__":
    import uvicorn
    #ChatModels are instances of LangChain "Runnables"
    model = ChatOpenAI(model="gpt-4") #"gpt-3.5-turbo"
    
    #test_chat(model)
    app=run_langchainserver(model)
    
    uvicorn.run(app, host="localhost", port=8000)
    #python mylangchain.py
    
    
    