#!/usr/bin/env python
# coding: utf-8

# ### GRIP at The Sparks Foundation
# ### Task-6 "Prediction using Decision Tree Algorithm"

# #### Author: Ashmita Roy Medha

# In[32]:


#Importing libraries and dataset
import numpy as np
import pandas as pd
df = pd.read_csv("E:/Internships/Grip/Prediction/data6.csv")
df


# In[33]:


df= df.set_index('Id')
df


# In[34]:


df.isnull().sum()


# In[35]:


df.describe()


# #### Vizualizing The Dataset
# ##### Using Matplotlib 

# In[36]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# ##### Linechart

# In[37]:


df.drop(['Species'], axis=1).plot.line(title='Iris Dataset')


# ##### Multiple Histograms

# In[38]:


df.plot.hist(subplots=True, layout=(2,2), figsize=(10, 10), bins=20)


# ##### Using Seaborn

# In[39]:


import seaborn as sns


# ##### Scatterplot

# In[40]:


sns.scatterplot(x='SepalLengthCm', y='SepalWidthCm', hue='Species', data=df)


# ##### Pairplot

# In[16]:


p_graph = sns.PairGrid(df, hue="Species")
p_graph.map_diag(sns.histplot)
p_graph.map_offdiag(sns.scatterplot)
p_graph.add_legend()


# ##### It’s also possible to use a different function in the upper and lower triangles to emphasize different aspects of the relationship

# In[24]:


graph = sns.PairGrid(df, hue="Species")
graph.map_upper(sns.scatterplot)
graph.map_lower(sns.kdeplot)
graph.map_diag(sns.kdeplot, lw=3, legend=False)


# ##### Heatmap

# In[43]:


sns.heatmap(df.corr(), annot=True)


# ###### Faceting

# In[44]:


g = sns.FacetGrid(df, col='Species')
g = g.map(sns.kdeplot, 'SepalLengthCm')


# ### Preparing for prediction

# In[45]:


inputs = df.drop('Species', axis='columns')
target = df['Species']


# In[46]:


inputs


# In[47]:


target


# #### Encoding the target 

# In[48]:


from sklearn.preprocessing import LabelEncoder


# In[49]:


laben = LabelEncoder()


# In[50]:


target = laben.fit_transform(target)


# In[51]:


target


# ### Iris-setosa: 0
# ### Iris-versicolor: 1
# ### Iris-virginica: 2

# #### Train Test Split

# In[56]:


from sklearn.model_selection import train_test_split


# In[83]:


X_train, X_test, y_train, y_test = train_test_split( inputs, target,test_size = 0.2, random_state = 42)


# #### Build the model

# In[84]:


from sklearn import tree


# In[85]:


model = tree.DecisionTreeClassifier()


# In[86]:


model.fit(X_train,y_train)


# In[87]:


model.score(X_train,y_train)


# In[88]:


model.predict([[6.7,3.0,5.2,2.3]]) #which will give the answer of Iris-virginica


# In[89]:


model.predict([[6.4,3.2,4.5,1.5]]) #which will give the answer of Iris-versicolor


# #### pip install graphviz 

# In[81]:


# Import necessary libraries for graph viz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz


# ### Visualize the Decision Tree 

# In[92]:


fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);


# In[ ]:




