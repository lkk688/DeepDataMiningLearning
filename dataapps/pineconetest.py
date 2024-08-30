import getpass
import os
PINECONE_API_KEY = getpass.getpass() #os.environ["PINECONE_API_KEY"]

from pinecone import Pinecone, ServerlessSpec
#from pinecone.grpc import PineconeGRPC as Pinecone
#from pinecone import ServerlessSpec

#from pinecone.grpc import PineconeGRPC as Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

#List all indexes in a project
allindexes = pc.list_indexes()
print(allindexes)

index_name = 'myindex'
if index_name in allindexes.names():
    index = pc.Index(index_name)
    print(index.describe_index_stats())
else:
    print(f"{index_name} not exist in pinecone")

index_name = "example-index"
if index_name not in allindexes.names():
    pc.create_index(
        name=index_name,
        dimension=2,
        metric="cosine",
        spec=ServerlessSpec(
            cloud='aws', 
            region='us-east-1'
        ) 
    ) 

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

#Similarity Search
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


pc.delete_index(index_name)