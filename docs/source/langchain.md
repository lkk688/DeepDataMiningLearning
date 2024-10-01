# LLM Concepts

Large language models (LLMs) are a class of generative AI models built using transformer networks that can recognize, summarize, translate, predict, and generate language using very large datasets.

Tokenization is the first step to building a model, which involves splitting text into smaller units called tokens that become the basic building blocks for LLMs. These extracted tokens are used to build a vocabulary index mapping tokens to numeric IDs, to numerically represent text suitable for deep learning computations. During the encoding process, these numeric tokens are encoded into vectors representing each token’s meaning. During the decoding process, when LLMs perform generation, tokenizers decode the numeric vectors back into readable text sequences.  

Retrieval Augmented Generation: rather than just passing a user question directly to a language model, the system "retrieves" any documents that could be relevant in answering the question, and then passes those documents (along with the original question) to the language model for a "generation" step. 
* Retrieval is by using semantic search. In this process, a numerical vector (an embedding) is calculated for all documents, and those vectors are then stored in a vector database (a database optimized for storing and querying vectors). 
* Incoming queries are then vectorized as well, and the documents retrieved are those who are closest to the query in embedding space.
* Langchain supports two different query methods: one that just optimizes similarity, another with optimizes for maximal marginal relevance. Users often want to specify metadata filters to filter results before doing semantic search
* An index is a data structure that supports efficient searching, and a retriever is the component that uses the index to find and return relevant documents in response to a user's query. The index is a key component that the retriever relies on to perform its function.

LlamaIndex(previously known as GPT Index), is a data framework specifically designed for LLM apps. It builds on top of LangChain to provide “a central interface to connect your LLMs with external data.” Its primary focus is on ingesting, structuring, and accessing private or domain-specific data. LlamaIndex offers a set of tools that facilitate the integration of private data into LLMs.
* Feed relevant information into the prompt of an LLM. Instead of feeding it all the data, you try to pick out the bits useful to each query.
* Starting with your documents, you first load them into LlamaIndex. It comes with many ready-made readers for sources such as databases, Discord, Slack, Google Docs, Notion, and GitHub repos. `docs = loader.load_data(branch="main")`, where `docs` is a list of all the files and their text, we can move on to parsing them into nodes.
* Next, you use LlamaIndex to parse the documents into nodes — basically chunks of text. An index is constructed next so that, later, when we query the documents, LlamaIndex can quickly retrieve the relevant data. The index can be stored in different ways, e.g., Vector Store.
```bash
# 2. Parse the docs into nodes
from llama_index.node_parser import SimpleNodeParser
parser = SimpleNodeParser()
nodes = parser.get_nodes_from_documents(docs)
```
* Step3 Index Construction: create embeddings (check out this article for a visual explanation) for each node and store it in a Vector Store.
* Step4 Store the Index. The documents are read, parsed, and indexed, but the index will be in memory. LlamaIndex has many storage backends (e.g., Pinecone, JSON flat-files).
* Step5 Run query. 
```bash
from llama_index import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="index")

index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine()
response = query_engine.query(
    "What does load_index_from_storage do and how does it work?"
)
print(response)
```

LangChain vs LlamaIndex: LangChain is also suitable for building intelligent agents capable of performing multiple tasks simultaneously. if your main goal is smart search and retrieval, LlamaIndex is a great choice. It excels in indexing and retrieval for LLMs, making it a powerful tool for deep exploration of data.

# LangChain
LangChain is a framework for developing applications powered by language models. LangChain is a framework that enables the development of data-aware and agentic applications. It provides a set of components and off-the-shelf chains that make it easy to work with LLMs (such as GPT). LangChain is ideal if you are looking for a broader framework to bring multiple tools together. 

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