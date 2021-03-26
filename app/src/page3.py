"""Home page shown when the user enters the application"""
import streamlit as st

import awesome_streamlit as ast
import SessionState
import numpy as np
import pandas as pd
import requests
from io import BytesIO
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
#from IPython.display import display, Image, SVG, Math, YouTubeVideo

def get_similar_products_cnn(doc_id, num_results,asins,df_asins,bottleneck_features_train,data):
        doc_id = asins.index(df_asins[doc_id])
        pairwise_dist = pairwise_distances(bottleneck_features_train, bottleneck_features_train[doc_id].reshape(1,-1))

        indices = np.argsort(pairwise_dist.flatten())[0:num_results]
        pdists  = np.sort(pairwise_dist.flatten())[0:num_results]
        st.write('='*60)
        st.write('Search Results')
        st.write('='*60) 
        for i in range(1,len(indices)):
            rows = data[['medium_image_url','title']].loc[data['asin']==asins[indices[i]]]
            for indx, row in rows.iterrows():
                       response = requests.get(row['medium_image_url'])
                       img = Image.open(BytesIO(response.content))
                       st.image(img, width=None)
                       st.write('Product Title: ', row['title'])
                       st.write('Brand: ', data['brand'].iloc[indices[i]])
                       st.write('Price:', data['formatted_price'].iloc[indices[i]]) 
                       #st.write('Price ', row['formatted_price'])
                       st.write('='*60)  
                       #st.write('Brand:', row['brand']) 
                       #st.write('Amazon Url: www.amzon.com/dp/'+ asins[indices[i]])

def write():
    
    session_state = SessionState.get(user_name='', favorite_color='black')
    #st.write(session_state.user_name)
    bottleneck_features_train = np.load('16k_data_cnn_features.npy')
    asins = np.load('16k_data_cnn_feature_asins.npy')
    asins = list(asins)
    # load the original 16K dataset
    data = pd.read_pickle('pickels/16k_apperal_data_preprocessed')
    df_asins = list(data['asin'])
    #get similar products using CNN features (VGG-16)    
    get_similar_products_cnn(session_state.user_name,session_state.rec_no+1,asins,df_asins,bottleneck_features_train,data)

if __name__ == "__main__":
    write()