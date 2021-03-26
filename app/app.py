#### Declaration Section ######
import streamlit as st
import awesome_streamlit as ast
import src.pages.page2
import src.pages.page3
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import math
import time
import re
import os
import seaborn as sns
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from matplotlib import gridspec
from scipy.sparse import hstack
import plotly
import plotly.figure_factory as ff
from plotly.graph_objs import Scatter, Layout
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from streamlit.hashing import _CodeHasher
import SessionState
import logging

session_state = SessionState.get(user_name='', rec_no='')
#st.beta_set_page_config(layout="wide")
plotly.offline.init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")


header = st.beta_container()
dataset = st.beta_container()
features = st.beta_container()
model_training = st.beta_container()
takeaway = st.beta_container()



ast.core.services.other.set_logging_format()

PAGES = {
    "Page2": src.pages.page2,
    "Page3": src.pages.page3,
}
    

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def remote_css(url):
    st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

def icon(icon_name):
    st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

with header:
    st.title('Apparel Recommendation System')
    

class _SessionState:

    def __init__(self, session, hash_funcs):
        """Initialize SessionState instance."""
        self.__dict__["_state"] = {
            "data": {},
            "hash": None,
            "hasher": _CodeHasher(hash_funcs),
            "is_rerun": False,
            "session": session,
        }

    def __call__(self, **kwargs):
        """Initialize state data once."""
        for item, value in kwargs.items():
            if item not in self._state["data"]:
                self._state["data"][item] = value

    def __getitem__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)
        
    def __getattr__(self, item):
        """Return a saved state value, None if item is undefined."""
        return self._state["data"].get(item, None)

    def __setitem__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value

    def __setattr__(self, item, value):
        """Set state value."""
        self._state["data"][item] = value
    
    def clear(self):
        """Clear session state and request a rerun."""
        self._state["data"].clear()
        self._state["session"].request_rerun()
    
    def sync(self):
        """Rerun the app with all state values up to date from the beginning to fix rollbacks."""

        # Ensure to rerun only once to avoid infinite loops
        # caused by a constantly changing state value at each run.
        #
        # Example: state.value += 1
        if self._state["is_rerun"]:
            self._state["is_rerun"] = False
        
        elif self._state["hash"] is not None:
            if self._state["hash"] != self._state["hasher"].to_bytes(self._state["data"], None):
                self._state["is_rerun"] = True
                self._state["session"].request_rerun()

        self._state["hash"] = self._state["hasher"].to_bytes(self._state["data"], None)

    
def _get_session():
    session_id = get_report_ctx().session_id
    session_info = Server.get_current()._get_session_info(session_id)

    if session_info is None:
        raise RuntimeError("Couldn't get your Streamlit Session object.")
    
    return session_info.session


def _get_state(hash_funcs=None):
    session = _get_session()

    if not hasattr(session, "_custom_session_state"):
        session._custom_session_state = _SessionState(session, hash_funcs)

    return session._custom_session_state


state = _get_state()   
    
local_css("style.css")
remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

plotly.offline.init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
data = pd.read_pickle('pickels/16k_apperal_data')
title_vectorizer = CountVectorizer()
title_features   = title_vectorizer.fit_transform(data['title'])   

def nlp_preprocessing(total_text, index, column):
    stop_words = set(stopwords.words('english'))
    if type(total_text) is not int:
        string = ""
        for words in total_text.split():
            # remove the special chars in review like '"#$@!%^&*()_+-~?>< etc.
            word = ("".join(e for e in words if e.isalnum()))
            # Conver all letters to lower-case
            word = word.lower()
            # stop-word removal
            if not word in stop_words:
                string += word + " "
        data[column][index] = string

#plotting code to understand the algorithm's decision.
def plot_heatmap(keys, values, labels, url, text):
        # we call dispaly_img based with paramete url
        #display_img(url, ax, fig)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        # we will display it in notebook 
        plt.imshow(img)

        # displays combine figure ( heat map and image together)
        plt.show()

def plot_heatmap_image(doc_id, vec1, vec2, url, text, model):

    # we find the common words in both titles, because these only words contribute to the distance between two title vec's
    intersection = set(vec1.keys()) & set(vec2.keys()) 

    # we set the values of non intersecting words to zero, this is just to show the difference in heatmap
    for i in vec2:
        if i not in intersection:
            vec2[i]=0

    # for labeling heatmap, keys contains list of all words in title2
    keys = list(vec2.keys())
    #  if ith word in intersection(lis of words of title1 and list of words of title2): values(i)=count of that word in title2 else 
    #values(i)=0 
    values = [vec2[x] for x in vec2.keys()]

    # labels: len(labels) == len(keys), the values of labels depends on the model we are using
        # if model == 'bag of words': labels(i) = values(i)
        # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
        # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))

    if model == 'bag_of_words':
        labels = values
    elif model == 'tfidf':
        labels = []
        for x in vec2.keys():
            # tfidf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # tfidf_title_features[doc_id, index_of_word_in_corpus] will give the tfidf value of word in given document (doc_id)
            if x in  tfidf_title_vectorizer.vocabulary_:
                labels.append(tfidf_title_features[doc_id, tfidf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)
    elif model == 'idf':
        labels = []
        for x in vec2.keys():
            # idf_title_vectorizer.vocabulary_ it contains all the words in the corpus
            # idf_title_features[doc_id, index_of_word_in_corpus] will give the idf value of word in given document (doc_id)
            if x in  idf_title_vectorizer.vocabulary_:
                labels.append(idf_title_features[doc_id, idf_title_vectorizer.vocabulary_[x]])
            else:
                labels.append(0)

    plot_heatmap(keys, values, labels, url, text)


# this function gets a list of wrods along with the frequency of each 
# word given "text"
def text_to_vector(text):
    word = re.compile(r'\w+')
    words = word.findall(text)
    # words stores list of all words in given string, you can try 'words = text.split()' this will also gives same result
    return Counter(words) # Counter counts the occurence of each word in list, it returns dict type object {word1:count}

def get_result(doc_id, content_a, content_b, url, model):
    text1 = content_a
    text2 = content_b

    # vector1 = dict{word11:#count, word12:#count, etc.}
    vector1 = text_to_vector(text1)

    # vector1 = dict{word21:#count, word22:#count, etc.}
    vector2 = text_to_vector(text2)

    plot_heatmap_image(doc_id, vector1, vector2, url, text2, model)

def bag_of_words_model(doc_id, num_results):

    pairwise_dist = pairwise_distances(title_features,title_features[doc_id])
     # np.argsort will return indices of the smallest distances
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    #pdists will store the smallest distances
    pdists  = np.sort(pairwise_dist.flatten())[0:num_results]

    #data frame indices of the 9 smallest distace's
    df_indices = list(data.index[indices])

    for i in range(0,len(indices)):
        # we will pass 1. doc_id, 2. title1, 3. title2, url, model
        get_result(indices[i],data['title'].loc[df_indices[0]], data['title'].loc[df_indices[i]],        
                   data['medium_image_url'].loc[df_indices[i]], 'bag_of_words')
        
        response = requests.get(data['medium_image_url'].loc[df_indices[i]])
        img = Image.open(BytesIO(response.content))
        #st.markdown("![Test Image](data['medium_image_url'])")
        st.image(img, width=None)
        st.write('Product ID:',data['asin'].loc[df_indices[i]])
        st.write('Brand:', data['brand'].loc[df_indices[i]])
        st.write('Title:', data['title'].loc[df_indices[i]])
        st.write('Euclidean Distance from the query:', pdists[i])
        #print('='*60)   

        
        print('Product ID:',data['asin'].loc[df_indices[i]])
        print ('Brand:', data['brand'].loc[df_indices[i]])
        print ('Title:', data['title'].loc[df_indices[i]])
        print ('Euclidean Distance:', pdists[i])
        print('='*60)   

    #write()

def write(index_val):
    #session_state = SessionState.get(user_name='', prod_no='')
    #st.write(session_state.user_name)
    #st.write(session_state.prod_no)
    #st.write(session_state.user_name)
    st.write(index_val)
    data = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
    stop_words = set(stopwords.words('english'))
    for index, row in data.iterrows():
        nlp_preprocessing(row['title'], index, 'title')
    title_vectorizer = CountVectorizer()
    title_features   = title_vectorizer.fit_transform(data['title'])     
    bag_of_words_model(index_val, 5)     
    
#if __name__ == "__main__":
    #main()   
### Read the Products 
col1, col2 = st.beta_columns(2)
df_data = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
data_new = data.reset_index()

#df_list = df_data['asin'].iloc[:5]
#df_list = df_data['asin'].iloc[:5]
df_list = df_data['brand'].value_counts(normalize =True).nlargest(10)
#st.sidebar.header('Choose your product')
brand_radio = st.sidebar.radio("Select Brand", (df_list.index[0],df_list.index[1], df_list.index[2],df_list.index[3],df_list.index[4]))
color_radio = st.sidebar.radio("Choose Color", ('Black','White','All'))
rec_no = st.sidebar.slider("No of Recommendations", 0,10,1)
if color_radio == 'Black' or color_radio == 'White':
    df_list = df_data['asin'][(df_data['brand'] == brand_radio) & (df_data['color'] == color_radio)].iloc[:5]
else:
    df_list = df_data['asin'][(df_data['brand'] == brand_radio)].iloc[:5]

#st.write(df_list)
options = list(range(len(df_list)))
try:
    with col2 : 
        button_image_clicked = st.button("Image Search", key = 1)
    #col2.write('Product ID:',df_data['asin'].loc[index_val])
        button_title_clicked = col2.button("Similar Products", key = 2)
    sel_index= int(st.sidebar.selectbox("Product Details", options,      format_func=lambda x: df_list.iloc[x]))
#st.write(sel_index)
    option = df_list.iloc[sel_index]
#st.write(option)

#option = st.sidebar.selectbox('Product Name',('',df_list.iloc[0],df_list.iloc[1],df_list.iloc[2],df_list.iloc[3],df_list.iloc[4]))
#session_state.user_name = option # Save the product selected in a session variable which I will use later.
    
    #index_val = df_data.index[df_data['asin'] == df_list.iloc[option]]
    index_val = data_new[data_new['asin'] == option].index[0]
    #index_val = df_data.index[df_data['asin'] == option].tolist()[0]
    #st.write(index_val)
    session_state.user_name = index_val
    session_state.rec_no = rec_no
    #st.write(data_new['medium_image_url'][data_new['index'] == index_val])
    response = requests.get(data_new['medium_image_url'].iloc[index_val])
    img = Image.open(BytesIO(response.content))
    #st.markdown("![Test Image](data['medium_image_url'])")
    #col1.header("Original")
    #col1.image(original, use_column_width=True)
    with col1 :
        st.image(img, width=None)
        st.write('Title:', data_new['title'].iloc[index_val])
        st.write('Brand:', df_data['brand'].iloc[index_val])
        st.write('Price:', df_data['formatted_price'].iloc[index_val])

    if  button_image_clicked:
        page = PAGES["Page3"]
    #    #state.sync()
        ast.shared.components.write_page(page)
    elif  button_title_clicked:
        #write(index_val)
        page = PAGES["Page2"]
        ast.shared.components.write_page(page)
except:
    st.write("No Products Available")    



