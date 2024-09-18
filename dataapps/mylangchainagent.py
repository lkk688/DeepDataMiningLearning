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
    llm_chain = simplellmchain()
    llm_tool = Tool(
        name='Language Model',
        func=llm_chain.invoke,
        description='use this tool for general purpose queries and logic'
    )

    tools = [magic_function, llm_tool]

    agent = create_tool_calling_agent(llm, tools, prompt)
    
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
    result = conversational_agent(
        "What's my name?"
    )
    
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    result = agent_executor.invoke({"input": "what is the value of magic_function(3)?"})

    # Using with chat history
    from langchain_core.messages import AIMessage, HumanMessage
    agent_executor.invoke(
        {
            "input": "what's my name?",
            "chat_history": [
                HumanMessage(content="hi! my name is bob"),
                AIMessage(content="Hello Bob! How can I assist you today?"),
            ],
        }
    )
    print("Done")