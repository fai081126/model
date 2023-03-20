#!/usr/bin/env python
# coding: utf-8

# In[4]:


#pip install git+https://github.com/pytube/pytube


# In[1]:


from pytube import YouTube


# In[2]:


url   = "https://www.youtube.com/watch?v=6hyLdfYIcxI"
yt = YouTube(url)


# In[3]:


stream = yt.streams
stream.get_highest_resolution().fps


# In[4]:


stream.get_highest_resolution().download()

