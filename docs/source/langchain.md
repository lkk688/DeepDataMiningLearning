# LangChain

Install LangChain via [link](https://python.langchain.com/v0.2/docs/how_to/installation/)
```bash
conda install langchain -c conda-forge #pip install langchain
```
LangChain supports many different language model providers that you can use interchangably [link](https://python.langchain.com/v0.2/docs/tutorials/llm_chain/). If using OpenAI API, the key should be set to `OPENAI_API_KEY` environment variable, or directly inside the code.
```bash
pip install openai
pip install -qU langchain-openai
pip install -qU langchain-google-vertexai
pip install -qU langchain-nvidia-ai-endpoints
pip install -qU langchain-openai
```
Read the API key
```bash
import os
try: 
    print("openai_api_key:", os.environ['OPENAI_API_KEY']) 
except KeyError:  
    print("openai_api_key does not exist") 
#openai_api_key = os.environ.get('OPENAI_API_KEY', 'sk-XXX') #
os.environ["OPENAI_API_KEY"] = getpass.getpass() #use interactive command line to get the API key
```

## LLM models
The LLM class is designed as a standard interface to LLM providers like OpenAI, Cohere, HuggingFace etc
Uses the OpenAI text-davinci-003 model as an example (you cannot use a chat model like gpt-3.5-turbo here)
```bash
# Use the OpenAI LLM wrapper and text-davinci-003 model
from langchain.llms import OpenAI
llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)
```
## Chat models
Chat models are a variation of language models - they use language models under the hood, but interface with applications using chat messages instead of a text in / text out approach. 
SystemMessage: Helpful context for the chatbot
HumanMessage: Actual message from the user
AIMessage: Response from the chatbot

## Chain operator
We can easily create the chain using the | operator. The | operator is used in LangChain to combine two elements together.
```bash
chain = model | parser
chain.invoke(messages)
```
`test_chat(model) in dataapps/mylangchain.py` performs the basic langchain testing.

PromptTemplates are a concept in LangChain designed to assist with this transformation. They take in raw user input and return data (a prompt) that is ready to pass into a language model.

LangServe helps developers deploy LangChain chains as a REST API. You do not need to use LangServe to use LangChain
```bash
pip install "langserve[all]"
```

Run the FastAPI server:
```bash
python mylangchain.py
#use langserve_client() to call server
```

We can use a Message History class to wrap our model and make it stateful. This will keep track of inputs and outputs of the model, and store them in some datastore. Future interactions will then load those messages and pass them into the chain as part of the input. 
```bash
pip install langchain_community
```

Streamlit with Langchain
```bash
pip install streamlit
pip install openai
pip install tiktoken
streamlit run streamlit_langchain.py
```