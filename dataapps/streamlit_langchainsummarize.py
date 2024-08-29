import os, tempfile, streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader #pip install pypdf

USE_PINECONE = True

if USE_PINECONE:
    from langchain_pinecone import PineconeVectorStore #pip install langchain-pinecone
else:
    from langchain_chroma import Chroma #pip install langchain-chroma
    

#ref: https://github.com/alphasecio/langchain-examples/tree/main/chroma-summary
# % streamlit run dataapps/streamlit_langchainsummarize.py

# Streamlit app
st.subheader('Summarize Document with LangChain, Options of Pinecone or Chroma')

# Get OpenAI API key and source document input
with st.sidebar:
    st.subheader("Settings")
    openai_api_key = st.text_input("OpenAI API key", value="", type="password")
    if USE_PINECONE:
        pinecone_api_key = st.text_input("Pinecone API key", type="password")
        pinecone_index = st.text_input("Pinecone index name")
    else:
        pinecone_api_key = ""
        pinecone_index = ""
source_doc = st.file_uploader("Source Document", label_visibility="collapsed", type="pdf")

# If the 'Summarize' button is clicked
if st.button("Summarize"):
    # Validate inputs
    if not openai_api_key.strip() or not source_doc:
        st.error(f"Please provide the missing fields.")
    else:
        try:
            with st.spinner('Please wait...'):
              # Save uploaded file temporarily to disk, 
              with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                  tmp_file.write(source_doc.read())
              loader = PyPDFLoader(tmp_file.name) #load the pdf file
              pages = loader.load_and_split() #split the file into pages
              os.remove(tmp_file.name) #delete temp file

              # Create embeddings for the pages 
              embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
              if USE_PINECONE:
                  vectordb = PineconeVectorStore.from_documents(pages, embeddings, index_name=pinecone_index)
              else:
                  #and insert into Chroma database
                  vectordb = Chroma.from_documents(pages, embeddings)

              # Initialize the ChatOpenAI module, load and run the summarize chain
              llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
              chain = load_summarize_chain(llm, chain_type="stuff")
              
              search = vectordb.similarity_search(" ")#same API for Pinecone and Chroma
              summary = chain.run(input_documents=search, question="Write a concise summary within 200 words.")

              st.success(summary)
        except Exception as e:
            st.exception(f"An error occurred: {e}")