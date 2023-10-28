import streamlit as st
import pickle
import pandas as pd
from pathlib import Path
from pprint import pprint
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings, CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.document_loaders import JSONLoader
import json
import os
import shutil
import requests
from PIL import Image
from io import BytesIO
from pprint import pprint

product_ikea_data = pd.read_csv('./ikea_ds.tsv', delimiter='\t')
youtube_ikea_data = pd.read_csv('./youtube_url_ikea.tsv', delimiter = '\t')

huggingface_key = st.secrets['HUGGINGFACE_KEY']
openai_key = st.secrets['OPENAI_API_KEY']

fs = LocalFileStore('cache_embed')
raw_embeddings = OpenAIEmbeddings()
embeddings = CacheBackedEmbeddings.from_bytes_store(
    raw_embeddings, fs, namespace=raw_embeddings.model
)
db = FAISS.load_local('faiss_index', embeddings=embeddings)
st.markdown('''
    <style>
            small {
            font-size: 13px !important;
            }
''', unsafe_allow_html=True)

def main() :
    st.title('Alfred')
    st.markdown('Yout best product searching!')
    prompt = st.text_input('What product would you like to search? : ')
    result = None
    if prompt :
        with st.spinner() :
            result = db.similarity_search_with_score(prompt, score_threshold=0.4)
    if result :
        if len(result) > 0 : show_product(result)
    else : st.markdown('**No Relevant Results!**')  

def show_product(result) :
    st.subheader('Best result') 
    response = requests.get(result[0][0].metadata['imgUrl'])
    img_thumb = Image.open(BytesIO(response.content))
    s1, s2, s3 = st.columns(3)
    with s1 :
        st.image(img_thumb, use_column_width=True)
    st.markdown(
        f"""
        <h3>{result[0][0].metadata['name']}</h3>
        <small>{result[0][0].metadata['description']}</small><br />
        """, unsafe_allow_html=True
    )
    for idx, row in youtube_ikea_data.loc[youtube_ikea_data['name'] == result[0][0].metadata['name']].iterrows() :
        st.markdown(f"<a href=\'{row['href']}\'> Video Reference : {row['title']} </a>", unsafe_allow_html=True)
    if len(result) > 1 :
        st.divider()
        st.subheader('Result Similiar')
    for result0 in result[1:] :
        res = result0[0]
        response = requests.get(result[0][0].metadata['imgUrl'])
        img_thumb = Image.open(BytesIO(response.content))
        s1, s2, s3 = st.columns(3)
        with s1 :
            st.image(img_thumb, use_column_width=True)
        st.markdown(
            f"""
            <h3>{res.metadata['name']}</h3>
            <small>{res.metadata['description']}</small><br />
            """, unsafe_allow_html=True
        )
        for idx, row in youtube_ikea_data.loc[youtube_ikea_data['name'] == res.metadata['name']].iterrows() :
            st.markdown(f"<a href=\'{row['href']}\'> Video Reference : {row['title']} </a>", unsafe_allow_html=True)
main()