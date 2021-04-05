#!/usr/bin/env python
# coding: utf-8

# # Problem Statement

# ### Emotion dataset is a  dataset which is used to do sentiment analysis 
# 
# Task:-
# find the emotions from human sentiments , here we predict the emotion and check the accuracy of prediction by using machine learning algorithms.
# Collection of documents and its emotions, It helps greatly in NLP Classification tasks
# 
# Content
# sentiment and emotion. Dataset is split into train & test for building the machine learning model
# 
# Example :-
# i feel like I am still looking at a blank canvas blank pieces of paper; sadness
# 
# 
# 
# Using ALogorithms are:
# 1. Multinomial Naive Baise
# 2. Random Forest Classifier
# 
# 

# ## Import Libraries

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Load Data Train,Test,Valid Data

# In[3]:


train=pd.read_csv("project/train.txt",sep=';',names=['sentiment','emotion'])
test=pd.read_csv("project/test.txt",sep=';',names=['sentiment','emotion'])


# ## Exploring Data

# In[4]:


train.head()


# In[5]:


test.head()


# In[6]:


train.shape,test.shape  # shape function is used to find rows and column


# In[7]:


train.describe()


# In[8]:


test.describe()


# In[9]:


train.isnull().sum()


# In[10]:


test.isnull().sum()


# In[11]:


train.dtypes


# In[12]:


train.info


# In[13]:


train['emotion'].nunique()


# In[14]:


train.emotion.unique() # It represent the unique value of in emotion column


# In[15]:


# this is used to print five Joy statement
joy_sentiment = train[train["emotion"] == 'joy']["sentiment"].values 
for i in range(0,5):
    print(joy_sentiment[i], "\n")


# In[16]:


# this is used to print five Love statement
love_sentiment = train[train["emotion"]=="love"]["sentiment"].values
for i in range(0,5):
    print(love_sentiment[i],"\n")


# In[17]:


train.emotion.value_counts() # It counts the value of emotion's each


# In[18]:


test.emotion.value_counts()


# In[19]:


# Data visualization


# In[20]:


Emotion_count=train.groupby('emotion').count()
plt.bar(Emotion_count.index.values, Emotion_count['sentiment'],color=["orange","black","yellow","red","gray","blue"])
plt.xlabel('emotion')
plt.ylabel('Number of emotion')
plt.show()


# ## This bar plot graph is represent 
# #### 1. The count of each emotion
# #### 2. Diffrent colorbars represent different emotion 

# ### There are six emotions three emotions are good for health and rest three emotion are bad for health.

# In[21]:


Emotion_count=test.groupby('emotion').count()
plt.bar(Emotion_count.index.values, Emotion_count['sentiment'],color=["orange","black","yellow","red","gray","blue"])
plt.xlabel('emotion')
plt.ylabel('Number of emotion')
plt.show()


# In[22]:


catFeatures = ['sentiment', 'emotion']
for catFeature in catFeatures:
    print(train[catFeature].unique())


# In[23]:


import plotly.express as px
fig = px.sunburst(train, path=["emotion"]) 
fig.show()


# # this graph shows in pie chart form 
# ### It represent the count and percent of emotion in dataset
# ### this graph says the count of joy is max and second max value is sadness . this represent that joy is most common emotion in the dataset this is better for each person.
# ### the sadness is second max emotion it means most of time human mood swings in between joy and sadness feeling.
# ### mostly person dont get surprise oftenly thats why the percenatge of surprise is very less.
# ### Love Anger and Fear is also a emotion 
# 
# 

# In[24]:


import plotly.express as px
fig = px.sunburst(train, path=["emotion"]) 
fig.show()


# In[25]:


get_ipython().system('pip install nltk ')


# NLTK is a leading platform for building Python programs to work with human language data. Written by the creators of NLTK, it guides the reader through the fundamentals of writing Python programs, working with corpora, categorizing text, analyzing linguistic structure, and more. .

# In[26]:


import nltk


# In[27]:


nltk.download("stopwords")


# In[28]:


from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# Stopwords considered as noise in the text. Text may contain stop words such as is, am, are, this, a, an, the, etc.
# 
# In NLTK for removing stopwords, you need to create a list of stopwords and filter out your list of tokens from these words.

# In[29]:


get_ipython().system('pip install wordcloud')

#from wordcloud import WordCloud
#def get_wordcloud(text):
 #   return wordcloud;


# In[30]:


from wordcloud import WordCloud


# In[31]:


def get_wordcloud(text):
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=stopwords.words('english')).generate(str(text))
    
    return wordcloud;


# In[32]:


emotions = train["emotion"].unique()

figure, axes = plt.subplots(ncols=2, nrows=3,figsize=(30,25))
plt.axis('off')

# for each emotion
for emotion, ax in zip(emotions, axes.flat):
    wordcloud = get_wordcloud(train[train["emotion"]==emotion]['sentiment'])
    ax.imshow(wordcloud)
    ax.title.set_text(emotion)
    ax.title.set_size(70)
    
    ax.axis('off')
    
plt.subplots_adjust(wspace=0.25, hspace=0.10)


# ### A word cloud is a collection, or cluster, of words depicted in different sizes.
# ### Here there are six wordcloud each cloud shows diffrent emotion
# ### In each cloud there are clustered words which represent the same as the cloud name.

# # Cleaning Data

# In[33]:


# From https://towardsdatascience.com/detecting-bad-customer-reviews-with-nlp-d8b36134dc7e

# return the wordnet object value corresponding to the POS tag
from nltk.corpus import wordnet

def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
import string
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)


# In[34]:


#train["sentiment"] = train["sentiment"].apply(lambda x: clean_text(x))
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# In[35]:


train["sentiment"] = train["sentiment"].apply(lambda x: clean_text(x))


# In[36]:


train.head()


# In[37]:


# Test data
test["sentiment"] = test["sentiment"].apply(lambda x: clean_text(x))


# In[38]:


nltk.download('vader_lexicon')


# In[39]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
train["sentiments"] = train["sentiment"].apply(lambda x: sid.polarity_scores(x))
train = pd.concat([train.drop(['sentiments'], axis=1), train['sentiments'].apply(pd.Series)], axis=1)


# Test data
sid = SentimentIntensityAnalyzer()
test["sentiments"] = test["sentiment"].apply(lambda x: sid.polarity_scores(x))
test = pd.concat([test.drop(['sentiments'], axis=1), test['sentiments'].apply(pd.Series)],axis=1)


# In[40]:


train.head()


# In[41]:


test.head()


# In[42]:



from sklearn import preprocessing

import pickle


# # Label Encoding

# In[43]:


# Create encoder based on train data
labelEncoder = preprocessing.LabelEncoder()
labelEncoder.fit(train.emotion)

# Train data
train.emotion = labelEncoder.transform(train.emotion)
 
       
# Validation data
# validation_data.emotion = labelEncoder.transform(validation_data.emotion)

# Test data
test.emotion = labelEncoder.transform(test.emotion)


# In[44]:


train.emotion.unique()


# In[45]:


test.emotion.unique()


# In[46]:


train.emotion.value_counts()


# In[47]:


df=pd.concat([train,test])


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
## Vectorization


# In[49]:


count_vectorize = CountVectorizer(analyzer=clean_text) 
vectorized = count_vectorize.fit_transform(df['sentiment'])


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    vectorized, df['emotion'], test_size=0.3, random_state=123)


# # Multinomial Naive Bayes classifier

# In[51]:


from sklearn.naive_bayes import MultinomialNB
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[52]:


from sklearn.feature_extraction.text import TfidfVectorizer
tf=TfidfVectorizer()
text_tf= tf.fit_transform(df['sentiment'])


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    text_tf, df['emotion'], test_size=0.3, random_state=123)


# In[54]:


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
# Model Generation Using Multinomial Naive Bayes
clf = MultinomialNB().fit(X_train, y_train)
predicted= clf.predict(X_test)
print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[55]:


# We got a classificationrate of 66.03% using TF_IDF features which is not considered as good accuracy .


# # Random Forest Classifier

# In[56]:


from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)


# In[57]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("The Training Score of Random Forest Classifier is: {:.3f}%".format(model_rf.score(X_train, y_train)*100))
print("\n                                                                    \n")
print("The Confusion Matrix for Random Forest Classifier is: \n{}\n".format(confusion_matrix(y_test, y_pred)))
print("\n                                                                     \n")
print("The Classification report: \n{}\n".format(classification_report(y_test, y_pred)))
print("\n                                                                      \n") 
print("The Accuracy Score of Random Forest Classifier is: {:.3f}%".format(accuracy_score(y_test, y_pred)*100))


# ## We got Accuracy of Random Forest Classifier is 84.704% which is really better than Multinomial Naive Bayes.
# # Conclusion:
# #### 1. We explore the data of emotions. get diffrent types of emotion is Joy,love surprise,anger,fear,sadness.
# 
# #### 2. We visualise the data by bar plot and by sunburst pie chart and by wordcloud  
# #### 3. After that we clean the data using some technique are:-
# #####                                +Tokenization
# #####                                + POS tagging
# #####                                + lemmatization
# #####                                + stemming
# #####                                + stopwords
# #### 4. we use algorithms multinomial naive bayes and random forest classifier we got the accuracy of 84.704% from random forest classifer.
# #### 5. From this we can analys from the human sentiment or sentences we find emotion of human.

# In[ ]:




