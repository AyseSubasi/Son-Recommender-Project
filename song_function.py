#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
from sklearn import datasets # sklearn comes with some toy datasets to practise
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import matplotlib.pyplot as plt
import pickle

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import config
import random


# In[122]:


#Initialize SpotiPy with user credentias
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id= config.client_id,
                                                           client_secret= config.client_secret))


# In[123]:


fin_df = pd.read_csv("./fin_df.csv", index_col=0)


# In[124]:


hot_songs = pd.read_csv("./df.csv", index_col=0)


# In[125]:


fin_df.head()


# ### scaling features

# In[126]:


fin_df.describe()


# In[127]:


#get numercial features

X = fin_df.select_dtypes(include=np.number)


# In[128]:


X.head()


# In[129]:


# drop columns that are irrelevant for clustering

# X = X.drop(columns=["duration_ms", "time_signature"],inplace=True)


# In[130]:


# X = np.array(X).reshape(1, -1)


# In[131]:


scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns = X.columns)
display(X.head())
print()
display(X_scaled_df.head())


# In[132]:


X_scaled_df.describe() # to check if features have the same weight


# ### Clustering

# In[133]:


kmeans = KMeans(n_clusters=12, random_state=1234)
kmeans.fit(X_scaled_df)


# In[134]:


clusters = kmeans.predict(X_scaled_df)
#clusters
pd.Series(clusters).value_counts().sort_index()


# In[135]:


# check which group was assigend 

X["cluster"] = clusters
X.head()


# ### playing with the parameters

# In[136]:


kmeans.inertia_ # decided k=12


# In[137]:


# K = range(2, 21)
# inertia = []

# for k in K:
#     print("Training a K-Means model with {} neighbours! ".format(k))
#     print()
#     kmeans = KMeans(n_clusters=k,
#                     random_state=1234)
#     kmeans.fit(X_scaled_df)
#     inertia.append(kmeans.inertia_)

# import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline

# plt.figure(figsize=(16,8))
# plt.plot(K, inertia, 'bx-')
# plt.xlabel('k')
# plt.ylabel('inertia')
# plt.xticks(np.arange(min(K), max(K)+1, 1.0))
# plt.title('Elbow Method showing the optimal k')


# elbow-> 8-9 checkin silouhette to decide..

# In[138]:


# K = range(2, 20)
# silhouette = []

# for k in K:
#     kmeans = KMeans(n_clusters=k,random_state=1234)
#     kmeans.fit(X_scaled_df)
#     silhouette.append(silhouette_score(X_scaled_df, kmeans.predict(X_scaled_df)))


# plt.figure(figsize=(16,8))
# plt.plot(K, silhouette, 'bx-')
# plt.xlabel('k')
# plt.ylabel('silhouette score')
# plt.xticks(np.arange(min(K), max(K)+1, 1.0))
# plt.title('Silhouette Method showing the optimal k')


# In[139]:


#here is the optimal k=3 the results are dissapointed
#for us is the optimal k between 8-9 -> k=12


# In[140]:


X.head(3)


# In[141]:


X['id'] = fin_df['id']


# In[142]:


#save X df  feat_clust.csv
X.to_csv("./feat_clust_id.csv") # X is the scaled features dataframe


# In[143]:


feat_clust = X.copy() # rename the DF to avoid confusion


# In[144]:


feat_clust.head()


# In[145]:


hot_songs.head()


# In[146]:


#rename columns:
hot_songs = hot_songs.rename(columns={'Rank': 'rank', 'Song Title': 'title', "Song Artist": "artist", "Year": "year", "Genre": "genre", "Decade": "decade"})


# In[147]:


def get_artists_from_id(id_):
    artists = sp.track(id_)['artists']
    artists_lst = [a['name'] for a in artists]
    return ", ".join(artists_lst)


# In[148]:


def get_trackname_from_id(id_):
    return sp.track(id_)['name']


# In[149]:


# X -> scaled DF (+ cluster labels)

def song_recommender():
    ask_song = input("Tell us a song you like: ").title()
    if hot_songs['title'].str.contains(ask_song).all():
        print("Nice, that is a hot song! Do you know " + hot_songs.sample(n=1)["title"].item() + " by " + hot_songs.sample(n=1)["artist"].item() + "? I'm sure you will like it!")
       

    else:
        print("Hey, that's not a hot song right now. Let me recommend another song for you!")
        results = sp.search(q = ask_song, limit=1)
        if results["tracks"]["items"]: 
            user_uri = results["tracks"]["items"][0]["uri"] #search after uri
            user_audio = sp.audio_features(tracks = user_uri) #get audio features
            user_df = pd.DataFrame(user_audio[0], index=[0])[feat_clust.drop(['cluster','id'], axis=1).columns]
            user_scaled = scaler.transform(user_df) # scale user input
            user_cluster = kmeans.predict(user_scaled)
            print(user_cluster)
            recommend_id = feat_clust[feat_clust['cluster'] == user_cluster[0]].sample(n=1)["id"].item()
            recommend_song_title = get_trackname_from_id(recommend_id)
            print("Nice song! Do you know " + recommend_song_title.upper() + "I'm sure you will like it!")
           
        else:
            print("No results.") 


# In[151]:


song_recommender()


# In[ ]:




