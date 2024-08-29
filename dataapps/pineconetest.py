
PINECONE_API_KEY="d82a6dc6-078b-476b-92c5-b37cca6b8ad6"
from pinecone import Pinecone, ServerlessSpec
#from pinecone.grpc import PineconeGRPC as Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = 'myindex'
index = pc.Index(index_name)
print(index.describe_index_stats())

pc.delete_index(index_name)