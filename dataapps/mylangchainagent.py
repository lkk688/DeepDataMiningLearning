
import getpass
import os
import sys
from abc import abstractmethod
#from pydantic.v1 import root_validator, validator, Field, BaseModel
from pydantic import model_validator, Field #root_validator, 
from langchain.memory import ConversationBufferMemory
# from langgraph.prebuilt import create_react_agent
# agent_executor = create_react_agent(llm, tools)
from langchain.agents import AgentExecutor
from langchain.agents import initialize_agent, create_tool_calling_agent
from langchain.agents import tool #, load_tools
from langchain.agents import BaseSingleActionAgent
#from langchain.agents import Tool, AgentExecutor, BaseSingleActionAgent
#from langchain.agents.tools import Tool
#from langchain_core.tools import Tool
from langchain.agents import Tool

from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

#from langchain.chains import LLMChain #Deprecated

from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from io import StringIO
from typing import Dict, Optional, List, Tuple, Any, Union, ClassVar


#from langchain.chains import TransformChain, SequentialChain, LLMChain
from langchain.schema import AgentAction, AgentFinish
#from langchain.llms import BaseLLM
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.llms import LLM
from langchain.schema.runnable import RunnableSequence
#from langchain.schema.runnable import Runnable
from langchain_huggingface import HuggingFacePipeline
#from langchain.llms import HuggingFacePipeline


def simplellmchain(llm):
    # prompt_template = "Tell me a {adjective} joke"
    # prompt = PromptTemplate(
    #     input_variables=["adjective"], template=prompt_template
    # )
    prompt = PromptTemplate(
        input_variables=["query"],
        template="{query}"
    )
    llm_chain = prompt | llm | StrOutputParser() #LLMChain(llm=llm, prompt=prompt)
    llm_chain.invoke("What is AI agent?")
    return llm_chain

def get_APIkey(KEYNAME='OPENAI_API_KEY'): #LANGCHAIN_API_KEY
    if not os.environ.get("KEYNAME", ""):
        api_key = getpass.getpass(f"Enter your {KEYNAME} API key: ")
        if api_key:
            os.environ[f"KEYNAME"] = api_key
        
def get_llm(llmtype='nvidia', test=False):
    if llmtype == 'nvidia':
        if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
            nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
            assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
            os.environ["NVIDIA_API_KEY"] = nvidia_api_key

        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        llm = ChatNVIDIA(
            model="meta/llama-3.1-8b-instruct",
            api_key=os.environ["NVIDIA_API_KEY"],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
        )
    elif llmtype == 'openai':
        get_APIkey(KEYNAME='OPENAI_API_KEY')
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o-mini")#"gpt-4")
    elif llmtype == 'huggingface':
        from transformers import pipeline
        from langchain.llms import HuggingFacePipeline
        model_kwargs = {"do_sample": True, "temperature": 0.4, "max_length": 4096}
        # model_name = "TheBloke/Llama-2-70B-chat-GPTQ"  ## Feel free to use for faster inference
        model_name = "TheBloke/Llama-2-13B-chat-GPTQ"
        llama_pipe = pipeline("text-generation", model=model_name, device_map="auto", model_kwargs=model_kwargs)
        llm = HuggingFacePipeline(pipeline=llama_pipe)

    if test:
        response = llm.invoke([HumanMessage(content="hi!")])
        print(response.content)
        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Tell me more about Olympic 2028!"),
        ]
        response =llm.invoke(messages)
        print(response.content)
        #response = llm.predict("<s>[INST]<<SYS>>Hello World!<</SYS>>respond![/INST]", max_length=128)
        #predict` was deprecated
        #response = llm.invoke("<s>[INST]<<SYS>>Hello World!<</SYS>>respond![/INST]")#, max_length=128)
        #print(response)
    return llm
    
def testagent():
    # Uncomment the below code to list the availabe models
    # ChatNVIDIA.get_available_models()

    llm = get_llm(llmtype='nvidia', test=True)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant"),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history")

    @tool
    def magic_function(input: int) -> int:
        """Applies a magic function to an input."""
        return input + 2

    # tools = load_tools(
    #     ['llm-math'],
    #     llm=llm
    # )
    
    # initialize the LLM tool
    # llm_chain = simplellmchain()
    # llm_tool = Tool(
    #     name='Language Model',
    #     func=llm_chain.invoke,
    #     description='use this tool for general purpose queries and logic'
    # )
    # tools = [magic_function, llm_tool]
    tools = [magic_function]

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True)
    result = agent_executor.invoke({"input": "what is the value of magic_function(3)?"})

    # Using with chat history
    agent_executor.invoke(
        {
            "input": "what's my name?",
            "chat_history": [
                HumanMessage(content="hi! my name is bob"),
                AIMessage(content="Hello Bob! How can I assist you today?"),
            ],
        }
    )
    
    zero_shot_agent = initialize_agent(
        agent="zero-shot-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=3
    )
    print(zero_shot_agent.agent.llm_chain.prompt.template)
    
    conversational_agent = initialize_agent(
        agent='conversational-react-description', 
        tools=tools, 
        llm=llm,
        verbose=True,
        max_iterations=3,
        memory=memory,
    )
    print(conversational_agent.agent.llm_chain.prompt.template)
    '''
    Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

    TOOLS:
    ------

    Assistant has access to the following tools:

    > magic_function: Applies a magic function to an input.

    To use a tool, please use the following format:

    ```
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [magic_function]
    Action Input: the input to the action
    Observation: the result of the action
    ```

    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

    ```
    Thought: Do I need to use a tool? No
    AI: [your response here]
    ```

    Begin!

    Previous conversation history:
    {chat_history}

    New input: {input}
    {agent_scratchpad}
    '''
    
    result = conversational_agent(
        "What's my name?"
    )
    print("Done")


def get_llamaprompt():
    llama_full_prompt = PromptTemplate.from_template(
        template="<s>[INST]<<SYS>>{sys_msg}<</SYS>>\n\nContext:\n{history}\n\nHuman: {input}\n[/INST] {primer}",
    )

    llama_prompt = llama_full_prompt.partial(
        sys_msg = (
            "You are a helpful, respectful and honest AI assistant."
            "\nAlways answer as helpfully as possible, while being safe."
            "\nPlease be brief and efficient unless asked to elaborate, and follow the conversation flow."
            "\nYour answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content."
            "\nEnsure that your responses are socially unbiased and positive in nature."
            "\nIf a question does not make sense or is not factually coherent, explain why instead of answering something incorrect."
            "\nIf you don't know the answer to a question, please don't share false information."
            "\nIf the user asks for a format to output, please follow it as closely as possible."
        ),
        primer = "",
        history = "",
    )
    llama_hist_prompt = llama_prompt.copy()
    llama_hist_prompt.input_variables = ['input', 'history']
    return llama_prompt, llama_hist_prompt

########################################################################
## General recipe for making new tools.
## You can also subclass tool directly, but this is easier to work with
class AutoTool:

    """Keep-Reasoning Tool

    This is an example tool. The input will be returned as the output
    """

    def get_tool(self, **kwargs):
        ## Shows also how some open-source libraries like to support auto-variables
        doc_lines = self.__class__.__doc__.split('\n')
        class_name = doc_lines[0]                     ## First line from the documentation
        class_desc = "\n".join(doc_lines[1:]).strip() ## Essentially, all other text

        return Tool(
            name        = kwargs.get('name',        class_name),
            description = kwargs.get('description', class_desc),
            func        = kwargs.get('func',        self.run),
        )

    def run(self, command: str) -> str:
        ## The function that should be ran to execute the tool forward pass
        return command

class AskForInputTool(AutoTool):

    """Ask-For-Input Tool

    This tool asks the user for input, which you can use to gather more information.
    Use only when necessary, since their time is important and you want to give them a great experience! For example:
    Action-Input: What is your name?
    """

    def __init__(self, fn = input):
        self.fn = fn

    def run(self, command: str) -> str:
        response = self.fn(command)
        return response
    
class MyAgentBase(BaseSingleActionAgent):

    ###################################################################################
    ## IMPORTANT METHODS. Will be subclassed later

    @abstractmethod
    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any):
        '''
        Taking the "intermediate_steps" as the history of steps.
        Decide on the next action to take! Return the required action
        (returns a query from the action method)
        '''
        pass

    ###################################################################################
    ## Methods you should know about, but not modify

    def action(self, tool, tool_input, finish=False) -> Union[AgentAction, AgentFinish]:
        '''Takes the action associated with the tool and feeds it the necessary parameters'''
        if finish: return AgentFinish({"output": tool_input},           log = f"\nFinal Answer: {tool_input}\n")
        else:      return AgentAction(tool=tool, tool_input=tool_input, log = f"\nAgent: {tool_input.strip()}\n")
        # else:    return AgentAction(tool=tool, tool_input=tool_input, log = f"\nTool: {tool}\nInput: {tool_input}\n") ## Actually Correct

    async def aplan(self, intermediate_steps, **kwargs):
        '''The async version of plan. It has to be defined because abstractmethod'''
        return await self.plan(intermediate_steps, **kwargs)

    @property
    def input_keys(self):
        return ["input"]

class MyAgent(MyAgentBase):

    ## Instance methods that can be passed in as BaseModel arguments.
    ## Will be associated with self

    general_prompt : PromptTemplate
    llm            : Optional[BaseLanguageModel] #LLM #BaseLLM #abstract class

    general_chain  : Optional[RunnableSequence] #Optional[LLMChain]
    max_messages   : int                   = Field(10, gt=1)

    temperature    : float                 = Field(0.6, gt=0, le=1)
    max_new_tokens : int                   = Field(128, ge=1, le=2048)
    eos_token_id   : Union[int, List[int]] = Field(2, ge=0)
    
    gen_kw_keys: Optional[List[str]] = ['temperature', 'max_new_tokens', 'eos_token_id']
    gen_kw: Optional[Dict] = {}
    
    # gen_kw_keys: ClassVar[List[str]] = ['temperature', 'max_new_tokens']
    # gen_kw: Optional[Dict] = {}

    user_toxicity  : float = 0.5
    user_emotion   : str = "Unknown"
    
    # Inject conversation memory
    memory         : ConversationBufferMemory = Field(default_factory=ConversationBufferMemory)

    #pip install pydantic==1.10.2
    #
    #@model_validator(mode='before')
    @root_validator(pre=False, skip_on_failure=True)
    def validate_input(cls, values: Any) -> Any:
        '''Think of this like the BaseModel's __init__ method'''
        if not values.get('general_chain'):
            llm = values.get('llm')
            prompt = values.get("general_prompt")
            memory = values.get("memory") #new add
            #values['general_chain'] = LLMChain(llm=llm, prompt=prompt)  ## <- Feature stop
            values['general_chain'] = prompt | llm 
            values['gen_kw'] = {k:v for k,v in values.items() if k in values.get('gen_kw_keys')}
        return values
    
    def plan(self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any):
        '''Takes in previous logic and generates the next action to take!'''

        ## [Base Case] Default message to start off the loop. TO NOT OVERRIDE
        tool, response = "Ask-For-Input Tool", "Hello World! How can I help you?"
        if len(intermediate_steps) == 0:
            return self.action(tool, response)

        ## History of past agent queries/observations
        queries      = [step[0].tool_input for step in intermediate_steps]
        observations = [step[1]            for step in intermediate_steps]
        last_obs     = observations[-1]    # Most recent observation (i.e. user input)

        #############################################################################
        ## FOR THIS METHOD, ONLY MODIFY THE ENCLOSED REGION

        ## [!] Probably a good spot for your user statistics tracking

        ## [Stop Case] If the conversation is getting too long, wrap it up
        if len(observations) >= self.max_messages:
            response = "Thanks so much for the chat, and hope to see ya later! Goodbye!"
            return self.action(tool, response, finish=True)

        ## [!] Probably a good spot for your input-augmentation steps
        memory_variables = self.memory.load_memory_variables({})
        print(f"lkk: {memory_variables}")
        print(f"lkk last_obs: {last_obs}")
        if memory_variables is None:
            memory_variables = ""

        ## [Default Case] If observation is provided and you want to respond... do it!
        #with SetParams(llm, **self.gen_kw):
        response = self.general_chain.run(last_obs)

        ## [!] Probably a good spot for your output-postprocessing steps

        ## FOR THIS METHOD, ONLY MODIFY THE ENCLOSED REGION
        #############################################################################

        ## [Default Case] Send over the response back to the user and get their input!
        return self.action(tool, response)


    def reset(self):
        self.user_toxicity = 0
        self.user_emotion = "Unknown"
        if getattr(self.general_chain, 'memory', None) is not None:
            self.general_chain.memory.clear()  ## Hint about what general_chain should be...

if __name__ == "__main__":
    print("main")
    student_name = "Kaikai Liu"   ## TODO: What's your name

    llm = get_llm(llmtype='nvidia', test=False)
    llama_prompt, llama_hist_prompt = get_llamaprompt()
    memory = ConversationBufferMemory(memory_key="chat_history")
    
    agent_kw = dict(
        llm = llm,
        general_prompt = llama_prompt, #llama_hist_prompt
        memory = memory,  # Add memory here
        max_new_tokens = 128,
        eos_token_id = [2]   
    )

    agent_ex = AgentExecutor.from_agent_and_tools(
        agent = MyAgent(**agent_kw),
        tools=[AskForInputTool().get_tool()], 
        verbose=True
    )
    agent_ex("")
    print("Finished")
