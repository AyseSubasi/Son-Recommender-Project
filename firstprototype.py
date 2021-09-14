#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup
import pandas as pd
import random


# In[2]:


top100=pd.read_csv("./top100.csv")


# Cleaning

# In[3]:


top100.head()


# In[ ]:


#delete index


# In[4]:


top100= top100.rename(columns={"Unnamed: 0": "ranking"})


# In[5]:


top100


# ## Frist Prototype
# 
# #input->if song in title -> recommend another song -> if not get audio features of the song -> recommend a song that sounds similar
# 

# In[9]:


#random choice
recommendations= random.choice(top100.title)


# In[10]:


def recommend_plus_artist():
    
    ask_song=input("Welcome to Gnoosic! Tell me your favorite song: ").lower().title()
#  ask_artist=input("Which artist?") # input artist
    
    if len(ask_song) == 0: #if input is empty ask again
        ask_song=input("Still waiting for your song: ").lower().title()
#typos:

#if same name of song then ask for artist:
        
    if ask_song in list(top100["title"]):
            return "Good choice! What do you think about:"  + " " + recommendations
        
        #if music doenst exist-> ask for artist
        

    if ask_song not in list (top100["title"]):
        ask_artist=input("Couldn't find a match. Maybe I can help you if you tell me the artist? Artist:").lower().title()
        
    if ask_artist in list(top100["artist"]):
        return "Good choice! What do you think about:"  + " " + recommendations
    
    else:
        return "Sorry there is no hot music for you"


# In[11]:


recommend_plus_artist()

