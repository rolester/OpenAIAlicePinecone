import openai
import streamlit as st
from streamlit_chat import message
import pandas as pd
import pinecone
import numpy as np

openai.api_type = "azure"
openai.api_base = "https://openairal.openai.azure.com/"
openai.api_version = "2023-06-01-preview"
openai.api_key = open("Azurekey.txt","r").read()

#Function to search the pinecont database and return the text as a string for the prompt
def searchdatabase(searchterm = 'Is Alice Happy?', results =8):

    from openai.embeddings_utils import get_embedding, cosine_similarity

    pinecone.init(api_key=open("Pineconekey.txt","r").read(), environment="gcp-starter")
    index = pinecone.Index("alice")

    query_embedding = get_embedding(searchterm,engine="textembeddingada002v2")

    #pkl is faster to read that csv
    datafile_path = "data/Alice_with_embeddings.pkl"
    df = pd.read_pickle(datafile_path)

    df = df.rename(columns={"Unnamed: 0": "id", "embedding": "values", "paragraph": "paragraph"})
    df['id'] = df['id'].astype(int)

    ##Get the IDs
    xc = index.query(vector=query_embedding,
    top_k=results,
    include_values=False)
    c = [x["id"] for x in xc["matches"]]
    dfres = pd.DataFrame(c, columns=['id'])
    dfres['id'] = dfres['id'].astype(str)

    ### join the IDs back to get the text
    dfres['id'] = dfres['id'].astype(int)
    dffinalSearch = dfres.join(df, how='left', on='id',  lsuffix='_left', rsuffix='_right')

    return str(dffinalSearch['paragraph'].to_list())

st.title("💬 AliceBot - search base GPT model or using pinecone search Alice in Wonderland")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How may I help you? Please start typing!"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


with st.sidebar:

    slider = st.slider(
        label='Temp', min_value=.0,
        max_value=1.0, value=.0, key='my_slider')

    input_select = st.selectbox('select a model', 
                             options=('gpt35turbo', 'GPT4'),
                             key='model')
    
    checkbox_input = st.checkbox('Just Alice in Wonderland?', key='alicecb')


if prompt := st.chat_input():

    #Create a copy of the messages to send to the GPT model
    #so that we do not add loads of junk ot the history
    tosend = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    messages = st.session_state.messages

    if checkbox_input:
        searchres = searchdatabase(prompt)
        tosend.append({"role": "user", "content": prompt + " answer based only based on the following information:  " + searchres})
    else:
        tosend.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(engine=input_select,
                                            messages=tosend,
                                            temperature=slider,)
    msg = response.choices[0].message
    st.session_state.messages.append(msg)
    st.chat_message("assistant").write(msg.content)

with st.sidebar:

    df = pd.DataFrame(st.session_state.messages)
    "Memory"
    df

    if checkbox_input:
        "Search Index Data"
        if 'searchres' in locals():
            searchres
