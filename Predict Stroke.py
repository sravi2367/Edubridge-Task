#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# ### Healthcare-dataset-stroke-data.csv dataset is used to predict whether a patient is likely to get stroke based on the input parameters like gender, age, and various diseases and smoking status. A subset of the original train data is taken using the filtering method for Machine Learning and Data Visualization purposes

# In[1]:


# import important packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


#import data 
health=pd.read_csv("healthcare-dataset-stroke-data.csv")


# ## Exploring Data

# In[3]:


# print starting 5 rows of data
health.head()


# In[4]:


health.columns  # print the name of columns in data


# In[5]:


health.shape  # print Shape of data(rows,columns)


# In[6]:


health.dtypes  # find datatype of each column


# In[7]:


## Droping unnecessary columns 
df= health.drop("id",axis=1)
df.head()


# In[8]:


catFeatures = ['gender', 'hypertension', 'heart_disease','ever_married', 'work_type', 'Residence_type', 'smoking_status']
for catFeature in catFeatures:
    print(df[catFeature].unique())


# In[9]:


df.describe()


# In[10]:


df.corr()


# # Data Visualisation

# In[11]:



# scatter plot
plt.scatter(df.age,df.avg_glucose_level)


# ### this scatter plot graph show the realtion between age and average glucose level in human.

# In[12]:


sns.FacetGrid(data=df, hue='smoking_status', size=5).map(plt.scatter,'age', 'avg_glucose_level').add_legend()
plt.show()


# In[13]:


### this graph showing the relationship between avgerage glucose level and smoking person.


# In[14]:


# histogram
df.plot(kind="hist", y="avg_glucose_level")


# In[15]:


## this is histogram graph repesent frequency of glucose level


# In[16]:


df.plot(kind="hist", x="age", y="bmi")


# In[17]:


df.plot(kind="hist", x="age", y="heart_disease")


# In[18]:


sns.boxplot(data=df, y='avg_glucose_level')
plt.title('Boxplot of avg_glucose_level')
plt.show()


# ## this box plot graph shows outliers in avg glucose level column.

# In[19]:


# Plotting Countplots

for catFeature in catFeatures:
    plt.figure(figsize=(10,5))
    sns.countplot(x = df[catFeature])
    plt.show()


# ### countplot graph shows the count of each variable'unique values of categorical data

# In[21]:


sns.pairplot(df)
plt.show()


# ### this pairplot graph provide the relationship between each variable as pairwise.

# In[22]:


# Treating 'smoking status' for unknown as unknown replace as NaN
df["smoking_status"].replace("Unknown",np.nan,inplace=True)


# In[23]:


#  check the Missing Data

df.isnull().sum()


# ### filling the missing value or impute the missing value.
# #### BMI is a continuous data and the missing values around 4% .therefore we impute the values with mean of BMI
# #### Smoking Status is categorical data therefore it is replace by mode of data ,here data is 30% missing .if we remove the column then it is good in analysing data. but in predicting stroke smoking data is very useful so impute the values which may be show some biasing but useful.

# In[24]:


df["bmi"].fillna(df["bmi"].mean(), inplace =True)
df["smoking_status"].fillna(df["smoking_status"].mode()[0],inplace =True)


# In[25]:


df.isnull().sum()


# In[26]:


# After filling the msiisng values countplot.
sns.countplot(x=df["smoking_status"])


# In[27]:


plt.figure(figsize=(10,7))
sns.distplot(df["bmi"])
plt.show()


# ### this graph show the distribution of univariate data .here BMI is a data.

# In[28]:


matrix = np.triu(df.corr())
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot = True, cmap = 'Purples', fmt=".2f", mask = matrix, vmin = -1, vmax = 1, linewidths = 0.1, linecolor = 'white')
plt.show()


# ### this heatmap graph showing the correlation of each variable to each variable  that how one variable is related to another the color strength show the value of correlation betweenzero to 1.
# ##### from this heatmap graph we summarise that how age is related to stroke ,BMI ,heart_disease and hypertension and so on.

# In[29]:


## Distribution of stroke by gender 
stroke_vs_gender   = df.query('gender != "Other"').groupby(['gender', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
stroke_vs_gender.iloc[[0, 2], 1] = "didn't have a stroke"
stroke_vs_gender.iloc[[1, 3], 1] = "had a stroke"
stroke_vs_gender


# In[30]:


import plotly as py
import plotly.graph_objs as go
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected = True)


# In[31]:


fig = px.sunburst(stroke_vs_gender, path = ['gender', 'stroke'], values = 'count', color = 'gender',
                 color_discrete_map = {'Female': '#e381bc', 'Male': '#81a8e3'})

fig.update_layout(annotations = [dict(text = 'Distribution of stroke by gender', 
                                      x = 0.5, y = 1.15, font_size = 24, showarrow = False, 
                                      font_family = 'Arial Black',
                                      font_color = 'black')])

fig.update_traces(textinfo = 'label + percent parent')
                  
fig.show()


# Based on the distribution graph between stroke and gender in this data, gender does not affect the probability of stroke, but in fact men have a higher risk of stroke. howewer, women have a higher mortality rate from stroke. This is a medical fact.

# In[32]:


# Distribution of stroke by age group
plt.figure(figsize = (12, 9))
sns.set_style("dark")
sns.kdeplot(df.query('stroke == 1')['age'], color = '#c91010', shade = True, label = 'Had a stroke', alpha = 0.5)
sns.kdeplot(df.query('stroke == 0')['age'], color = '#1092c9', shade = True, label = "Didn't have a stroke", alpha = 0.5)
plt.ylabel('')
plt.xlabel('AGE')
plt.yticks([])
plt.legend(loc = 'upper left')
plt.show()


# In[33]:


## From this graph we can clearly seen that age effects on occurence the of strokes. strokes comes at any age age but it is more frequent after the of 50.


# In[34]:


df.index


# In[35]:


df['age_group'] = 0
for i in range(len(df.index)):
    if df.iloc[i, 1] < 2.0:
        df.iloc[i, 11] = 'baby'
    elif df.iloc[i, 1] < 17.0 and df.iloc[i, 1] >= 2.0:
        df.iloc[i, 11] = 'child'
    elif df.iloc[i, 1] < 30.0 and df.iloc[i, 1] >= 17.0:
        df.iloc[i, 11] = 'young adults'
    elif df.iloc[i, 1] < 60.0 and df.iloc[i, 1] >= 30.0:
        df.iloc[i, 11] = 'middle-aged adults'
    elif df.iloc[i, 1] < 80.0 and df.iloc[i, 1] >= 60.0:
        df.iloc[i, 11] = 'old-aged adults'
    else:
        df.iloc[i, 11] = 'long-lived'
        
df.head(5)


# In[36]:


stroke_vs_age = df.groupby(['age_group', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
stroke_vs_age.iloc[[0, 2, 4, 6, 8, 10], 1] = "didn't have a stroke"
stroke_vs_age.iloc[[1, 3, 5, 7, 9], 1] = "had a stroke"

stroke_vs_age


# In[37]:


fig = px.sunburst(stroke_vs_age, path = ['age_group', 'stroke'], values = 'count', color = 'age_group',
                 color_discrete_map = {'baby': '#c9b02e', 'child': '#007585', 'young adults': '#5b7521', 
                                       'middle-aged adults': '#21754c', 'old-aged adults': '#504c7a', 'long-lived': '#5e5e5e'},
                 width = 700, height = 700)

fig.update_layout(annotations = [dict(text = 'Distribution of stroke by age group', 
                                      x = 0.5, y = 1.1, font_size = 24, showarrow = False, 
                                      font_family = 'Arial Black',
                                      font_color = 'black')])

fig.update_traces(textinfo = 'label + percent parent')
                  
fig.show()


# ### There is a large correlation between age and stroke risk.
# #### Almost all doctors know that stroke is most often an age problem. 
# #### However, this can also happen with young people, in the available data in groups "baby" and "child", 2 people had a stroke.
# #### The risk of stroke mainly affect the people whose age is more than 55 or 60.
# #### In the pie graph we can see the whose is in the group of old age has 12% risk and who is long lived aged more than 80 there is 22% risk increase.

# In[38]:


#The affect of hypertension and heart diseases on stroke risk
hyper = df.groupby(['hypertension', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
hyper.iloc[[0, 1], 0] = 'No hypertension'
hyper.iloc[[2, 3], 0] = 'Yes hypertension'
hyper.iloc[[0, 2], 1] = "didn't have a stroke"
hyper.iloc[[1, 3], 1] = "had a stroke"
hyper


# In[39]:


fig = px.sunburst(hyper, path = ['stroke', 'hypertension'], values = 'count', color = 'stroke',
                 color_discrete_map = {"didn't have a stroke": '#1092c9', "had a stroke": '#c91010'},
                 width = 700, height = 700)

fig.update_layout(annotations = [dict(text = 'Affect of hypertension on stroke risk', 
                                      x = 0.5, y = 1.1, font_size = 24, showarrow = False, 
                                      font_family = 'Arial Black',
                                      font_color = 'black')])

fig.update_traces(textinfo = 'label + percent parent')
                  
fig.show()


# ### Based on the data, if a person have a stroke then 27% person has hypertension. 
# ### hypertension increase the risk of stroke so we can neglect the hypertension.

# In[40]:


heart = df.groupby(['heart_disease', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
heart.iloc[[0, 1], 0] = 'No heart diseases'
heart.iloc[[2, 3], 0] = 'Yes heart diseases'
heart.iloc[[0, 2], 1] = "didn't have a stroke"
heart.iloc[[1, 3], 1] = "had a stroke"
heart


# ### from this count table ,we can clearly seen that if a person has no heart disease then the percentage of rosk of stroke is 4.3%.
# ### while if a person has a heart disease then the risk of stroke increase 20.5% .
# ### therefore, we can conclude that if a person have heart disease then there is more chances of stroke other than those who do not have heart disease.

# In[41]:


fig = px.sunburst(heart, path = ['stroke', 'heart_disease'], values = 'count', color = 'stroke',
                 color_discrete_map = {"didn't have a stroke": '#1092c9', "had a stroke": '#c91010'},
                 width = 700, height = 700)

fig.update_layout(annotations = [dict(text = 'Affect of heart diseases on stroke risk', 
                                      x = 0.5, y = 1.1, font_size = 24, showarrow = False, 
                                      font_family = 'Arial Black',
                                      font_color = 'black')])

fig.update_traces(textinfo = 'label + percent parent')
                  
fig.show()


# ### from this graph we can clearly seen that heart disease increase the rate of stroke chances .if a person get stroke the there 19% person have heart disease from this observation we can say heart disease also major factor in getting stroke.

# In[42]:


# affect of living conditions on stroke risk
relation = df.groupby(['ever_married', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
relation.iloc[[0,1], 0] = 'Never married'
relation.iloc[[2,3], 0] = 'Married'
relation.iloc[[0, 2], 1] = "didn't have a stroke"
relation.iloc[[1, 3], 1] = "had a stroke"
relation


# In[43]:


fig = px.bar(relation, x = 'ever_married', y = 'count', color = 'stroke', title = 'Affect of marriage on stroke risk',             
    color_discrete_map = {"didn't have a stroke": '#1092c9', 'had a stroke': '#c91010'})


fig.update_layout(plot_bgcolor = 'white', title_font_family = 'Arial', title_font_color = '#221f1f', title_font_size = 20, title_x = 0.5)

fig.update_yaxes(showline = True, linecolor = '#f5f2f2', linewidth = 2,
                 showgrid = True, gridwidth = 1, gridcolor = '#f5f2f2',
                 title_font_size = 17, title_font_color = '#221f1f', tickfont_family = 'Arial', tickfont_color = '#221f1f')

fig.update_xaxes(showline = True, linecolor = '#f5f2f2', linewidth = 2, title = 'Ever married?',
                 title_font_size = 17, title_font_color = '#221f1f', tickfont_family = 'Arial', tickfont_color = '#221f1f')

fig.show()


# From this graph based on data, we can conclude that married person get more chances of stroke, but we can not concluded from marital status only .beacuse non married person age is comparatevely less than the married person.so there is age is also factor.

# In[44]:


work = df.groupby(['work_type', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
work.iloc[[0, 2, 3, 5, 7], 1] = "didn't have a stroke"
work.iloc[[1, 4, 6, 8], 1] = "had a stroke"
work


# From this table, we can see work type does not affect the stroke.so we can conclude that the work type does not affect the risk of stroke. but here we are not calculating the pressure of work which affect the risk of stroke .
# beacuse pressure of work create stress and stress is also factor of stroke.

# In[45]:


fig = px.bar(work, x = 'work_type', y = 'count', color = 'stroke', title = 'Affect of type of wok on stroke risk',
             color_discrete_map = {"didn't have a stroke": '#1092c9', 'had a stroke': '#c91010'})


fig.update_layout(plot_bgcolor = 'white', title_font_family = 'Arial', title_font_color = '#221f1f', title_font_size = 20, title_x = 0.5)

fig.update_yaxes(showline = True, linecolor = '#f5f2f2', linewidth = 2,
                 showgrid = True, gridwidth = 1, gridcolor = '#f5f2f2',
                 title_font_size = 17, title_font_color = '#221f1f', tickfont_family = 'Arial', tickfont_color = '#221f1f')

fig.update_xaxes(showline = True, linecolor = '#f5f2f2', linewidth = 2, title = 'Type of work',
                 title_font_size = 17, title_font_color = '#221f1f', tickfont_family = 'Arial', tickfont_color = '#221f1f')

fig.show()


# In[46]:


residence = df.groupby(['Residence_type', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
residence.iloc[[0, 2], 1] = "didn't have a stroke"
residence.iloc[[1, 3], 1] = "had a stroke"
residence


# In[ ]:





# In[47]:


fig = px.bar(residence, x = 'Residence_type', y = 'count', color = 'stroke', title = 'Affect of residence type on stroke risk',
             color_discrete_map = {"didn't have a stroke": '#1092c9', 'had a stroke': '#c91010'})


fig.update_layout(plot_bgcolor = 'white', title_font_family = 'Arial', title_font_color = '#221f1f', title_font_size = 20, title_x = 0.5)

fig.update_yaxes(showline = True, linecolor = '#f5f2f2', linewidth = 2,
                 showgrid = True, gridwidth = 1, gridcolor = '#f5f2f2',
                 title_font_size = 17, title_font_color = '#221f1f', tickfont_family = 'Arial', tickfont_color = '#221f1f')

fig.update_xaxes(showline = True, linecolor = '#f5f2f2', linewidth = 2, title = 'Residence type',
                 title_font_size = 17, title_font_color = '#221f1f', tickfont_family = 'Arial', tickfont_color = '#221f1f')

fig.show()


# ### Living in a rural or urban areas does not affect the risk of stroke in any way

# In[48]:


smoking = df.groupby(['smoking_status', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
smoking.iloc[[0, 2, 4], 1] = "didn't have a stroke"
smoking.iloc[[1, 3, 5], 1] = "had a stroke"
smoking


# In[49]:


fig = px.bar(smoking, x = 'smoking_status', y = 'count', color = 'stroke', title = 'Affect of smoking on stroke risk',
             color_discrete_map = {"didn't have a stroke": '#1092c9', 'had a stroke': '#c91010'})


fig.update_layout(plot_bgcolor = 'white', title_font_family = 'Arial', title_font_color = '#221f1f', title_font_size = 20, title_x = 0.5)

fig.update_yaxes(showline = True, linecolor = '#f5f2f2', linewidth = 2,
                 showgrid = True, gridwidth = 1, gridcolor = '#f5f2f2',
                 title_font_size = 17, title_font_color = '#221f1f', tickfont_family = 'Arial', tickfont_color = '#221f1f')

fig.update_xaxes(showline = True, linecolor = '#f5f2f2', linewidth = 2, title = 'Smoking_status',
                 title_font_size = 17, title_font_color = '#221f1f', tickfont_family = 'Arial', tickfont_color = '#221f1f')
fig.show()


# ### From this data we can see there is very little afffect of smoking on risk of stroke. but from other research ,we also observed that passive smoking increased the overall risk of stroke by 45%.  the risk of stroke increased by 12% for each increment of 5 cigarettes per day.so we never conclude that smoking does not increase the chances of stroke. for getting best results we have to get more correct data of smoking status and stroke .so we can conclude correctly.

# In[50]:


df.head(1)


# In[51]:


df['glucose_group'] = 0
for i in range(len(df.index)):
    if df.iloc[i, 7] < 100.0:
        df.iloc[i, 12] = 'Normal'
    elif df.iloc[i, 7] >= 100.0 and df.iloc[i, 7] < 125.0:
        df.iloc[i, 12] = 'Prediabetes'
    else:
        df.iloc[i, 12] = 'Diabetes'

df.head(5)


# In[52]:


glu = df.groupby(['glucose_group', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
glu.iloc[glu.query('stroke == 0').index.to_list(), 1] = "didn't have a stroke"
glu.iloc[glu.query('stroke == 1').index.to_list(), 1] = "had a stroke"
glu


# ### From this count table data, we can clearly seen how glucose level affect the risk of stroke. if a person is diabetic then the risk of stroke is 11%. and if a person is prediabetic then percentage is only 3.9%.if glucose is in normal level then the chances is 3.7%.

# In[53]:


fig = px.sunburst(glu, path = ['stroke', 'glucose_group'], values = 'count', color = 'stroke',
                 color_discrete_map = {"didn't have a stroke": '#1092c9', "had a stroke": '#c91010'},
                 width = 700, height = 700)

fig.update_layout(annotations = [dict(text = 'Affect of level of glucose on stroke risk', 
                                      x = 0.5, y = 1.1, font_size = 24, showarrow = False, 
                                      font_family = 'Arial Black',
                                      font_color = 'black')])

fig.update_traces(textinfo = 'label + percent parent')
                  
fig.show()


# #### Diabetes has a big impact on the risk of stroke. if the glucose level is very high then chances of getting stroke is very high .

# In[54]:


df['bmi_group'] = 0
for i in range(len(df.index)):
    if df.iloc[i, 8] < 18.5:
        df.iloc[i, 13] = 'Underweight'
    elif df.iloc[i, 8] < 25.0 and df.iloc[i, 8] >= 18.5:
        df.iloc[i, 13] = 'Normal weight'
    elif df.iloc[i, 8] < 30.0 and df.iloc[i, 8] >= 25.0:
        df.iloc[i, 13] = 'Overweight'
    else:
        df.iloc[i, 13] = 'Obese'
        
df.head(5)


# In[55]:


bmi = df.groupby(['bmi_group', 'stroke']).agg({'stroke': 'count'}).rename(columns = {'stroke': 'count'}).reset_index()
bmi.iloc[bmi.query('stroke == 0').index.to_list(), 1] = "didn't have a stroke"
bmi.iloc[bmi.query('stroke == 1').index.to_list(), 1] = "had a stroke"
bmi


# In[56]:


fig = px.sunburst(bmi, path = ['stroke', 'bmi_group'], values = 'count', color = 'stroke',
                 color_discrete_map = {"didn't have a stroke": '#1092c9', "had a stroke": '#c91010'},
                 width = 700, height = 700)

fig.update_layout(annotations = [dict(text = 'Affect of bmi on stroke risk', 
                                      x = 0.5, y = 1.1, font_size = 24, showarrow = False, 
                                      font_family = 'Arial Black',
                                      font_color = 'black')])

fig.update_traces(textinfo = 'label + percent parent')
                  
fig.show()


# ### From this Graph or data, We can easily conclude that if a person getting stroke then 46% person is overweight and 39% is obese.
# ### Because according to medical study ,Obesity/ Overweight are primary risk factors for stroke for men and women of all races. Degree of obesity, defined by body mass index, waist circumference, or waist-to-hip ratio, was a significant risk factor for ischemic stroke (stroke due to lack of blood flow, rather than due to clotting) regardless of sex or race.

# ### label Encoding

# In[57]:



from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()                  # le=label encoding variable

en_data = df.apply(le.fit_transform)  # en_data=encoded data variable

en_data.head()


# In[58]:


# feature selection
y = en_data['stroke']
X = en_data.drop('stroke', axis = 1)


# ### Splitting test and train set.

# In[59]:



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, stratify=y)


# In[60]:


# Applying Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[61]:


X_train.shape,X_test.shape


# # Training model using Random Forest Classifier algorithm

# In[62]:


from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)


# In[63]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("The Training Score of Random Forest Classifier is: {:.3f}%".format(model_rf.score(X_train, y_train)*100))
print("\n                                                                    \n")
print("The Confusion Matrix for Random Forest Classifier is: \n{}\n".format(confusion_matrix(y_test, y_pred)))
print("\n                                                                     \n")
print("The Classification report: \n{}\n".format(classification_report(y_test, y_pred)))
print("\n                                                                      \n") 
print("The Accuracy Score of Random Forest Classifier is: {:.3f}%".format(accuracy_score(y_test, y_pred)*100))


# ## Conclusion
# ### Random Forest classifier  have high precision. The number of false-positive is better handled by Random Forest.
# ### The accuracy of confusion matrix is 94.96%.
# ### Therefore we can use Random Forest to predict whether or not a patient will suffer from a stroke or not.

# In[ ]:




