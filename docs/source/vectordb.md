# Pinecone
[Pinecone](https://www.pinecone.io/?ref=alphasec.io) is a fully managed vector database, making it easy to build high-performance vector search applications without infrastructure hassles. Pinecone provides long-term memory for high-performance AI applications. It’s a managed, cloud-native vector database with a streamlined API and no infrastructure hassles. Pinecone serves fresh, relevant query results with low latency at the scale of billions of vectors. On the free Starter plan, you get 1 project, 5 serverless indexes in the us-east-1 region of AWS, and up to 2 GB of storage.

Sign up for an account with Pinecone (Signup with Google account *.sjsu.edu), or log in if you already have an account, and create an API key. You can get your key from the Pinecone console: Open the Pinecone console, Select your project (Default), Go to API Keys, Copy your API key.

Install the Python client on the command line:
```bash
pip install pinecone #install without gRPC
pip install "pinecone-client[grpc]"
```
Use your API key to initialize your client:
```bash
PINECONE_API_KEY="YOUR_API_KEY"
from pinecone import Pinecone, ServerlessSpec
#from pinecone.grpc import PineconeGRPC as Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
```

Key concepts in Pinecone: https://docs.pinecone.io/guides/get-started/key-concepts. A organization is a group of one or more projects that use the same billing. A project belongs to an organization and contains one or more indexes. Each project belongs to exactly one organization, but only users who belong to the project can access the indexes in that project. API keys and Assistants are project-specific. 

In Pinecone, an index is the highest-level organizational unit of data, where you define the dimension of vectors (i.e., number of values in a vector) to be stored and the similarity metric to be used when querying them. Normally, you choose a dimension and similarity metric based on the embedding model used to create your vectors. In Pinecone, there are two types of indexes: serverless and pod-based. Each index runs on at least one pod, which are pre-configured units of hardware for running a Pinecone service." With the free Starter plan, you can create one pod with enough resources to support 100K vectors with 1536-dimensional embeddings and metadata.

A backup is a static copy of a serverless index. A collection is a static copy of a pod-based index. Both backups and collections only consume storage. They are non-queryable representations of a set of records. You can create a backup or collection from an index, and you can create a new index from that backup or collection.

You can create an index in the Pinecone web console. Click Create your first Index, provide the Index Name and Dimensions, select the Pod Type, and click Create Index.

You can also create a serverless index named "myindex" in code. that performs nearest-neighbor search using the cosine distance metric for 2-dimensional vectors:
```bash
index_name = "myindex"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536, # Replace with your model dimensions, max is 1536
        metric="cosine", # Replace with your model metric
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 
```

Within an index, vectors are stored in namespaces, and all upserts, queries, and other data operations always target one namespace. A namespace is a partition within an index. It divides records in an index into separate groups. Namespaces are essential for implementing multitenancy when you need to isolate the data of each customer/user. A record is a basic unit of data and consists of the following: Record ID (record’s unique ID), Dense vector (also referred to as a vector embedding or simply a vector), Metadata (optional, additional information that can be attached to vector embeddings to provide more context and enable additional filtering capabilities), and Sparse vector (optional). For example, the original text of the embeddings can be stored in the metadata. Each dimension in a sparse vector typically represents a word from a dictionary, and the non-zero values represent the importance of these words in the document.


For example, use the upsert operation to write six 2-dimensional vectors into 2 distinct namespaces. Use the describe_index_stats operation to check if the current vector count matches the number of vectors you upserted:
```bash
index = pc.Index(index_name)

index.upsert(
    vectors=[
        {"id": "vec1", "values": [1.0, 1.5]},
        {"id": "vec2", "values": [2.0, 1.0]},
        {"id": "vec3", "values": [0.1, 3.0]},
    ],
    namespace="example-namespace1"
)

index.upsert(
    vectors=[
        {"id": "vec1", "values": [1.0, -2.5]},
        {"id": "vec2", "values": [3.0, -2.0]},
        {"id": "vec3", "values": [0.5, -1.5]},
    ],
    namespace="example-namespace2"
)
print(index.describe_index_stats())
```

Query each namespace in your index for the 3 vectors that are most similar to an example 2-dimensional vector using the cosine similarity metric you specified for the index:
```bash
query_results1 = index.query(
    namespace="example-namespace1",
    vector=[1.0, 1.5],
    top_k=3,
    include_values=True
)

print(query_results1)

query_results2 = index.query(
    namespace="example-namespace2",
    vector=[1.0,-2.5],
    top_k=3,
    include_values=True
)

print(query_results2)
```
When you no longer need the index, use the delete_index operation to delete it:
```bash
pc.delete_index(index_name)
```

References:

https://docs.pinecone.io/examples/notebooks

https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/rag-getting-started.ipynb

https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/langchain-retrieval-augmentation.ipynb

https://colab.research.google.com/github/pinecone-io/examples/blob/master/docs/semantic-search.ipynb

https://docs.pinecone.io/examples/sample-apps/namespace-notes

https://docs.pinecone.io/guides/data/upsert-data

# Chroma DB
## Introduction to Chroma
[Chroma](https://www.trychroma.com/), is an open-source, lightweight embedding (or vector) database that can be used to store embeddings locally. 
```bash
pip install chromadb
```

If you are running Chroma in client-server mode [links](https://docs.trychroma.com/guides), you may not need the full Chroma library. You can install the chromadb-client package. This package is a lightweight HTTP client for the server with a minimal dependency footprint.
```bash
pip install chromadb-client
```

Basic python code for Chroma:
```bash
#Create a Chroma Client
import chromadb
chroma_client = chromadb.Client()
#Create a collection
#collection = chroma_client.create_collection(name="my_collection")
# switch `create_collection` to `get_or_create_collection` to avoid creating a new collection every time
collection = chroma_client.get_or_create_collection(name="my_collection")

#Add some text documents to the collection
# collection.add(
#     documents=[
#         "This is a document about pineapple",
#         "This is a document about oranges"
#     ],
#     ids=["id1", "id2"]
# )
# switch `add` to `upsert` to avoid adding the same documents every time
collection.upsert(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

#Query the collection
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)

```
Collections are where you'll store your embeddings, documents, and any additional metadata. Chroma will store your text and handle embedding and indexing automatically. 

## Chroma Server with Persistent Chroma Client
You can configure Chroma to save and load the database from your local machine. Data will be persisted automatically and loaded on start (if it exists).
```bash
import chromadb
client = chromadb.PersistentClient(path="/path/to/save/to")
client.heartbeat() # returns a nanosecond heartbeat. Useful for making sure the client remains connected
```

Running Chroma in client/server mode: To start the Chroma server, run the following command:
```bash
chroma run --path /db_path
```

Python Client code:
```bash
import chromadb
# Example setup of the client to connect to your chroma server
client = chromadb.HttpClient(host='localhost', port=8000)

import asyncio
# Or for async usage:
async def main():
    client = await chromadb.AsyncHttpClient(host='localhost', port=8000)
    collection = await client.create_collection(name="my_collection")
    await collection.add(
        documents=["hello world"],
        ids=["id1"]
    )
asyncio.run(main())
```

## Collections
Chroma lets you manage collections of embeddings, using the collection primitive. Chroma collections are created with a name and an optional embedding function. If you supply an embedding function, you must supply it every time you get the collection. If no embedding function is supplied, Chroma will use [sentence transformer](https://www.sbert.net/index.html) as a default.
```bash
collection = client.create_collection(name="my_collection", embedding_function=emb_fn)
collection = client.get_collection(name="my_collection", embedding_function=emb_fn)

collection = client.get_or_create_collection(name="test") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.
client.delete_collection(name="my_collection") # Delete a collection and all associated embeddings, documents, and metadata. ⚠️ This is destructive and not reversible
collection.peek()  # returns a list of the first 10 items in the collection
collection.count()  # returns the number of items in the collection
collection.modify(name="new_name") # Rename the collection
```

create_collection also takes an optional metadata argument which can be used to customize the distance method of the embedding space by setting the value of hnsw:space. Valid options for hnsw:space are "l2", "ip, "or "cosine". The default is "l2" which is the squared L2 norm [ref](https://docs.trychroma.com/guides). "ip" is Inner product.
```bash
 collection = client.create_collection(
        name="collection_name",
        metadata={"hnsw:space": "cosine"} # l2 is the default
    )

```

Add data to Chroma with .add. If Chroma is passed a list of documents, it will automatically tokenize and embed them with the collection's embedding function (the default will be used if none was supplied at collection creation). Chroma will also store the documents themselves. If the documents are too large to embed using the chosen embedding function, an exception will be raised.
```bash
collection.add(
    documents=["lorem ipsum...", "doc2", "doc3", ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```
Each document must have a unique associated id. Trying to .add the same ID twice will result in only the initial value being stored. An optional list of metadata dictionaries can be supplied for each document, to store additional information and enable filtering.

You can also store documents elsewhere, and just supply a list of embeddings and metadata to Chroma. You can use the ids to associate the embeddings with your documents stored elsewhere.
```bash
collection.add(
    documents=["doc1", "doc2", "doc3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    ids=["id1", "id2", "id3", ...]
)
```

Chroma collections can be queried in a variety of ways, using the .query method. You can query by a set of query_embeddings.
```bash
collection.query(
    query_embeddings=[[11.1, 12.1, 13.1],[1.1, 2.3, 3.2], ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
```
The query will return the n_results closest matches to each query_embedding, in order. An optional where filter dictionary can be supplied to filter by the metadata associated with each document. Additionally, an optional where_document filter dictionary can be supplied to filter by contents of the document.

You can also query by a set of query_texts. Chroma will first embed each query_text with the collection's embedding function, and then perform the query with the generated embedding.
```bash
collection.query(
    query_texts=["doc10", "thus spake zarathustra", ...],
    n_results=10,
    where={"metadata_field": "is_equal_to_this"},
    where_document={"$contains":"search_string"}
)
```

You can also retrieve items from a collection by id using .get. .get also supports the where and where_document filters. If no ids are supplied, it will return all items in the collection that match the where and where_document filters.
```bash
collection.get(
	ids=["id1", "id2", "id3", ...],
	where={"style": "style1"}
)
```

Any property of items in a collection can be updated using .update.
```bash
collection.update(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents=["doc1", "doc2", "doc3", ...],
)
```
Chroma also supports an upsert operation, which updates existing items, or adds them if they don't yet exist.
```bash
collection.upsert(
    ids=["id1", "id2", "id3", ...],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2], ...],
    metadatas=[{"chapter": "3", "verse": "16"}, {"chapter": "3", "verse": "5"}, {"chapter": "29", "verse": "11"}, ...],
    documents=["doc1", "doc2", "doc3", ...],
)
```

Chroma supports deleting items from a collection by id using .delete. The embeddings, documents, and metadata associated with each item will be deleted. ⚠️ Naturally, this is a destructive operation, and cannot be undone.
```bash
collection.delete(
    ids=["id1", "id2", "id3",...],
	where={"chapter": "20"}
)
```
## Embeddings
[Embeddings](https://docs.trychroma.com/guides/embeddings) can represent text, images, and soon audio and video. Chroma provides lightweight wrappers around popular embedding providers, making it easy to use them in your apps. You can set an embedding function when you create a Chroma collection, which will be used automatically, or you can call them directly yourself. By default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings. This embedding function runs locally on your machine.
```bash
from chromadb.utils import embedding_functions
default_ef = embedding_functions.DefaultEmbeddingFunction()
#call the function directly
val = default_ef(["foo"])
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
```
You can pass in an optional model_name argument, which lets you choose which Sentence Transformers model to use. You can see a list of all available models [here](https://www.sbert.net/docs/pretrained_models.html). You can create your own embedding function to use with Chroma, it just needs to implement the EmbeddingFunction protocol.

Chroma supports multi-modal embedding functions [link](https://docs.trychroma.com/guides/multimodal), which can be used to embed data from multiple modalities into a single embedding space. Chroma has the OpenCLIP embedding function built in, which supports both text and images.
```bash
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
embedding_function = OpenCLIPEmbeddingFunction()

```
