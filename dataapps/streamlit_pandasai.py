import streamlit as st
import pandas as pd
#from pandasai import PandasAI
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt
import os

#streamlit run dataapps/streamlit_pandasai.py

#Session State is a way to share variables between reruns, for each user session. In addition to the ability to store and persist state, Streamlit also exposes the ability to manipulate state using Callbacks. 
#https://docs.streamlit.io/library/api-reference/session-state

st.title("pandas-ai streamlit interface")

#test senssion state:
# Initialization
key = 'key'
if key not in st.session_state:
    st.session_state[key] = 'value'
    #st.session_state.key = 'value' # Session State also supports attribute based syntax

# Read the value of an item in Session State and display it 
st.write(st.session_state.key)

# Delete a single key-value pair
del st.session_state[key]

# Delete all the items in Session state
for key in st.session_state.keys():
    del st.session_state[key]

#Every widget with a key is automatically added to Session State:
st.text_input("Your name", key="name")
st.write(st.session_state.name)

st.write("A demo interface for [PandasAI](https://github.com/gventuri/pandas-ai)")
st.write(
    "Looking for an example *.csv-file?."
)

# if "openai_key" not in st.session_state:
#     with st.form("API key"):
#         key = st.text_input("OpenAI Key", value="", type="password")
#         if st.form_submit_button("Submit"):
#             st.session_state.openai_key = key
#             st.session_state.prompt_history = []
#             st.session_state.df = None
#             st.success('Saved API key for this session.')
st.session_state.df = None
st.session_state.prompt_history = []

#if "openai_key" in st.session_state:
#if st.session_state.df is None:
uploaded_file = st.file_uploader(
    "Choose a CSV file. This should be in long format (one datapoint per row).",
    type="csv",
)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df

with st.form("Question"):
    question = st.text_input("Question", value="", type="default")
    submitted = st.form_submit_button("Submit")
    if submitted:
        with st.spinner():
            llm = OpenAI(api_token=st.session_state.openai_key)
            #pandas_ai = PandasAI(llm)
            smartdf = SmartDataframe(st.session_state.df, config={"llm": llm})
            #x = pandas_ai.run(, prompt=question)
            x = df.chat(question)
            print(x)

            if os.path.isfile('temp_chart.png'):
                im = plt.imread('temp_chart.png')
                st.image(im)
                os.remove('temp_chart.png')

            if x is not None:
                st.write(x)
            st.session_state.prompt_history.append(question)

if st.session_state.df is not None:
    st.subheader("Current dataframe:")
    st.write(st.session_state.df)

st.subheader("Prompt history:")
st.write(st.session_state.prompt_history)


if st.button("Clear"):
    st.session_state.prompt_history = []
    st.session_state.df = None