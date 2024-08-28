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

## Text Embeddings
Embeddings are a measure of the relatedness of text strings, and are represented with a vector (list) of floating point numbers. The distance between two vectors measures their relatedness - the shorter the distance, the higher the relatedness. Embeddings are used for a wide variety of use cases - text classification, search, clustering, recommendations, anomaly detection, diversity measurement etc.

The LangChain Embedding class is designed as an interface for embedding providers like OpenAI, Cohere, HuggingFace etc. The base class exposes two methods `embed_query` and `embed_documents` - the former works over a single document, while the latter can work across multiple documents.

```bash
# Retrieve OpenAI text embeddings for a text input
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

text = "This is a sample query."
query_result = embeddings.embed_query(text)
print(query_result)
print(len(query_result))

text = ["This is a sample query.", "This is another sample query.", "This is yet another sample query."]
doc_result = embeddings.embed_documents(text)
print(doc_result)
print(len(doc_result))
```


# LangChain with Chroma
Chroma provides wrappers around the OpenAI embedding API, which uses the text-embedding-ada-002 second-generation model. By default, Chroma uses an in-memory DuckDB database; it can be persisted to disk in the persist_directory folder on exit and loaded on start (if it exists), but will be subject to the machine's available memory. Chroma can also be configured to run in a client-server mode, where the database runs from the disk instead of memory, [link](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/chroma.html?ref=alphasec.io#persistance)

## Summarize Documents
Use OpenAI API and LangChain to summarize documents:
```bash
% streamlit run dataapps/streamlit_langchainbasic.py
```

Summarize Documents with LangChain and Chroma: To summarize the document, we first split the uploaded file into individual pages, create embeddings for each page using the OpenAI embeddings API, and insert them into the Chroma vector database. Then, we retrieve the information from the vector database using a similarity search, and run the LangChain Chains module to perform the summarization over the input.
```bash
% streamlit run dataapps/streamlit_langchainchroma.py
```

 https://alphasec.io/summarize-documents-with-langchain-and-chroma/
 https://github.com/alphasecio/langchain-examples/tree/main/chroma-summary