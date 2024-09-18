from langchain.agents import AgentExecutor, create_tool_calling_agent, tool
from langchain_core.prompts import ChatPromptTemplate

import getpass
import os

if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    nvidia_api_key = getpass.getpass("Enter your NVIDIA API key: ")
    assert nvidia_api_key.startswith("nvapi-"), f"{nvidia_api_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvidia_api_key

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.memory import ConversationBufferMemory
# from langgraph.prebuilt import create_react_agent
# agent_executor = create_react_agent(llm, tools)
from langchain.agents import initialize_agent
from langchain.prompts import PromptTemplate
#from langchain.chains import LLMChain #Deprecated
from langchain.agents import Tool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, HumanMessage

def simplellmchain():
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

if __name__ == "__main__":
    # Uncomment the below code to list the availabe models
    # ChatNVIDIA.get_available_models()

    llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",
    api_key=os.environ["NVIDIA_API_KEY"],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    )
    response = llm.invoke([HumanMessage(content="hi!")])
    print(response.content)

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