
import logging
import streamlit as st
import awesome_streamlit as ast
from sklearn.feature_extraction.text import CountVectorizer
from awesome_streamlit.core.services import resources
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
import SessionState


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

#def display_img(url,ax,fig):
    # we get the url of the apparel and download it
    #response = requests.get(url)
    #img = Image.open(BytesIO(response.content))
    # we will display it in notebook 
    #plt.imshow(img)
    #st.markdown("![Test Image](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif)")
    #st.image(url, width=None)

#plotting code to understand the algorithm's decision.
def plot_heatmap(keys, values, labels, url, text):
        # keys: list of words of recommended title
        # values: len(values) ==  len(keys), values(i) represents the occurence of the word keys(i)
        # labels: len(labels) == len(keys), the values of labels depends on the model we are using
                # if model == 'bag of words': labels(i) = values(i)
                # if model == 'tfidf weighted bag of words':labels(i) = tfidf(keys(i))
                # if model == 'idf weighted bag of words':labels(i) = idf(keys(i))
        # url : apparel's url

        # we will devide the whole figure into two parts
        #gs = gridspec.GridSpec(2, 2, width_ratios=[4,1], height_ratios=[4,1]) 
        #fig = plt.figure(figsize=(25,3))

        # 1st, ploting heat map that represents the count of commonly ocurred words in title2
        #ax = plt.subplot(gs[0])
        # it displays a cell in white color if the word is intersection(lis of words of title1 and list of words of title2), in black if not
        #ax = sns.heatmap(np.array([values]), annot=np.array([labels]))
        #ax.set_xticklabels(keys) # set that axis labels as the words of title
        #ax.set_title(text) # apparel title

        # 2nd, plotting image of the the apparel
        #ax = plt.subplot(gs[1])
        # we don't want any grid lines for image and no labels on x-axis and y-axis
        #ax.grid(False)
        #ax.set_xticks([])
        #ax.set_yticks([])
        #ax.set_title(text)

        # we call dispaly_img based with paramete url
        #display_img(url, ax, fig)
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        # we will display it in notebook 
        plt.imshow(img)

        # displays combine figure ( heat map and image together)
        plt.show()

def plot_heatmap_image(doc_id, vec1, vec2, url, text, model):

    # doc_id : index of the title1
    # vec1 : input apparels's vector, it is of a dict type {word:count}
    # vec2 : recommended apparels's vector, it is of a dict type {word:count}
    # url : apparels image url
    # text: title of recomonded apparel (used to keep title of image)
    # model, it can be any of the models, 
        # 1. bag_of_words
        # 2. tfidf
        # 3. idf

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
    st.write('='*60)
    st.write('Search Results')
    st.write('='*60) 
    for i in range(1,len(indices)):
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
        st.write('Price:', data['formatted_price'].loc[df_indices[i]])
        st.write('='*60) 
        #st.write('Euclidean Distance from the query:', pdists[i])
        #print('='*60)   

        
        print('Product ID:',data['asin'].loc[df_indices[i]])
        print ('Brand:', data['brand'].loc[df_indices[i]])
        print ('Title:', data['title'].loc[df_indices[i]])
        print ('Euclidean Distance:', pdists[i])
        print('='*60)   

    #write()

def write():
    session_state = SessionState.get(user_name='', rec_no='')
    #st.write(session_state.user_name)
    #st.write(session_state.prod_no)
    #st.write(session_state.user_name)
    data = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
    stop_words = set(stopwords.words('english'))
    for index, row in data.iterrows():
        nlp_preprocessing(row['title'], index, 'title')
    title_vectorizer = CountVectorizer()
    title_features   = title_vectorizer.fit_transform(data['title'])     
    bag_of_words_model(session_state.user_name, session_state.rec_no+1) 
         
if __name__ == "__main__":
    write()

    
