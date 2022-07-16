#!/usr/bin/env python
# coding: utf-8

# # Churn Prediction

# 
# 
# The variables are:
# customerID: Customer ID gender: Gender of customer SeniorCitizen: Whether the customer is a senior citizen or not (1, 0) Partner: Whether the customer has a partner or not (Yes, No) Dependents: Whether the customer has dependents or not (Yes, No) tenure: Number of months the customer has stayed with the company PhoneService: Whether the customer has a phone service or not (Yes, No) MultipleLines: Whether the customer has multiple lines or not (Yes, No, No phone service) InternetService: Customer’s internet service provider (DSL, Fiber optic, No) OnlineSecurity: Whether the customer has online security or not (Yes, No, No internet service) OnlineBackup: Whether the customer has online backup or not (Yes, No, No internet service) DeviceProtection: Whether the customer has device protection or not (Yes, No, No internet service) TechSupport: Whether the customer has tech support or not (Yes, No, No internet service) StreamingTV: Whether the customer has streaming TV or not (Yes, No, No internet service) StreamingMovies: Whether the customer has streaming movies or not (Yes, No, No internet service) Contract: The contract term of the customer (Month-to-month, One year, Two year) PaperlessBilling: Whether the customer has paperless billing or not (Yes, No) PaymentMethod: The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)) MonthlyCharges: The amount charged to the customer monthly TotalCharges: The total amount charged to the customer Churn: Whether the customer churned or not (Yes or No)
# 
# At first glance, only customerID seems irrelevant to customer churn. Other variables may or may not have an effect on customer churn. We will figure out.
# 

# ### Importing the data set

# In[1]:



import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv("Telco-Customer-Churn.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# # Exploratory Data Analysis

# In[6]:


df.isna().sum().sum() #missing values in the data set


# There is no missing value in the data set so we can jump to explore it

# In[7]:


df.columns


# In[8]:


df.dtypes


# In[9]:


df.Churn.value_counts()


# In[10]:


df.info()


# Target variable has imbalanced class distribution. Negative class (Churn=No) is much less than positive class (churn=Yes). Imbalanced class distributions influence the performance of a machine learning model negatively. We will use upsampling or downsampling to overcome this issue. 

# It is always beneficial to explore the features (independent variables) before trying to build a model. Let's first discover the features that only have two values.

# In[11]:


columns = df.columns
binary_cols = []

for col in columns:
    if df[col].value_counts().shape[0] == 2:
        binary_cols.append(col)


# In[12]:


binary_cols # categorical features with two classes


# The remaining categorical variables have more than two values (or classes).

# In[13]:


# Categorical features with multiple classes
multiple_cols_cat = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract','PaymentMethod']


# ## Binary categorical features

# Let's check the class distribution of binary features.

# In[14]:


fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

sns.countplot("gender", data=df, ax=axes[0,0])
sns.countplot("SeniorCitizen", data=df, ax=axes[0,1])
sns.countplot("Partner", data=df, ax=axes[0,2])
sns.countplot("Dependents", data=df, ax=axes[1,0])
sns.countplot("PhoneService", data=df, ax=axes[1,1])
sns.countplot("PaperlessBilling", data=df, ax=axes[1,2])


# There is a high imbalance in SeniorCitizen and PhoneService variables. Most of the customers are not senior and similarly, most customers have a phone service.

# It is better to check how the target variable (churn) changes according to the binary features. To be able to make calculations, we need to change the values of target variable. "Yes" will be 1 and "No" will be 0.

# In[15]:


churn_numeric = {'Yes':1, 'No':0}
df.Churn.replace(churn_numeric, inplace=True)


# In[16]:


df[['gender','Churn']].groupby(['gender']).mean()


# Average churn rate for males and females are approximately the same which indicates gender variable does not bring a valuable prediction power to a model. Therefore, I will not use gender variable in the machine learning model.

# In[17]:


df[['SeniorCitizen','Churn']].groupby(['SeniorCitizen']).mean()


# In[18]:


df[['Partner','Churn']].groupby(['Partner']).mean()


# In[19]:


df[['Dependents','Churn']].groupby(['Dependents']).mean()


# In[20]:


df[['PhoneService','Churn']].groupby(['PhoneService']).mean()


# In[21]:


df[['PaperlessBilling','Churn']].groupby(['PaperlessBilling']).mean()


# The other binary features have an effect on the target variable. The phone service may also be skipped if you think 2% difference can be ignored. I have decided to use this feature in the model.
# 
# You can also use pandas pivot_table function to check the relationship between features and target variable.

# In[22]:


table = pd.pivot_table(df, values='Churn', index=['gender'],
                    columns=['SeniorCitizen'], aggfunc=np.mean)
table


# In[23]:


table = pd.pivot_table(df, values='Churn', index=['Partner'],
                    columns=['Dependents'], aggfunc=np.mean)
table


# ## Other Categorical Features

# It is time to explore other categorical features. We also have continuous features such as tenure, monthly charges and total charges which I will discuss in the next part.
# 
# There are 6 variables that come with internet service. There variables come into play if customer has internet service.

# ### Internet Service

# In[24]:


sns.countplot("InternetService", data=df)


# In[25]:


df[['InternetService','Churn']].groupby('InternetService').mean()


# Internet service variable is definitely important in predicting churn rate. As you can see, customers with fiber optic internet service are much likely to churn than other customers although there is not a big difference in the number of customers with DSL and fiber optic. This company may have some problems with fiber optic connection. However, it is not a good way to make assumptions based on only one variable. Let's also check the monthly charges.

# In[26]:


df[['InternetService','MonthlyCharges']].groupby('InternetService').mean()


# Fiber optic service is much more expensive than DSL which may be one of the reasons why customers churn.

# In[27]:


fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=True)

sns.countplot("StreamingTV", data=df, ax=axes[0,0])
sns.countplot("StreamingMovies", data=df, ax=axes[0,1])
sns.countplot("OnlineSecurity", data=df, ax=axes[0,2])
sns.countplot("OnlineBackup", data=df, ax=axes[1,0])
sns.countplot("DeviceProtection", data=df, ax=axes[1,1])
sns.countplot("TechSupport", data=df, ax=axes[1,2])


# In[28]:


df[['StreamingTV','Churn']].groupby('StreamingTV').mean()


# In[29]:


df[['StreamingMovies','Churn']].groupby('StreamingMovies').mean()


# In[30]:


df[['OnlineSecurity','Churn']].groupby('OnlineSecurity').mean()


# In[31]:


df[['OnlineBackup','Churn']].groupby('OnlineBackup').mean()


# In[32]:


df[['DeviceProtection','Churn']].groupby('DeviceProtection').mean()


# In[33]:


df[['TechSupport','Churn']].groupby('TechSupport').mean()


# All internet service related features seem to have different churn rates for their classes.

# ### Phone service

# In[34]:


df.PhoneService.value_counts()


# In[35]:


df.MultipleLines.value_counts()


# If a customer does not have a phone service, he/she cannot have multiple lines. MultipleLines column includes more specific data compared to PhoneService column. So I will not include PhoneService column as I can understand the number of people who have phone service from MultipleLines column. MultipleLines column takes the PhoneService column one step further.

# In[36]:


df[['MultipleLines','Churn']].groupby('MultipleLines').mean()


# ### Contract, Payment Method

# In[37]:


plt.figure(figsize=(10,6))
sns.countplot("Contract", data=df)


# In[38]:


df[['Contract','Churn']].groupby('Contract').mean()


# It seems like, as expected, customers with short-term contract are more likely to churn. This clearly explains the motivation for companies to have long-term relationship with their customers.

# In[39]:


plt.figure(figsize=(10,6))
sns.countplot("PaymentMethod", data=df)


# In[40]:


df[['PaymentMethod','Churn']].groupby('PaymentMethod').mean()


# ### Continuous Variables
# 
# The continuous features are tenure, monthly charges and total charges. The amount in total charges columns is proportional to tenure (months) multiplied by monthly charges. So it is unnecessary to include total charges in the model. Adding unnecassary features will increase the model complexity. It is better to have a simpler model when possible. Complex models tend to overfit and not generalize well to new, previously unseen observations. Since the goal of a machine learning model is to predict or explain new observations, overfitting is a crucial issue.
# 
# Let's also have a look at the distribution of continuous features.
# 

# In[41]:


fig, axes = plt.subplots(1,2, figsize=(12, 7))

sns.distplot(df["tenure"], ax=axes[0])
sns.distplot(df["MonthlyCharges"], ax=axes[1])


# In[42]:


df[['tenure','MonthlyCharges','Churn']].groupby('Churn').mean()


# It is clear that people who have been a customer for a long time tend to stay with the company. The average tenure in months for people who left the company is 20 months less than the average for people who stay. 
# 
# It seems like monthly charges also have an effect on churn rate. 
# 
# Contract and tenure features may be correlated because customer with long term contract are likely to stay longer with the company. Let's figure out.

# In[43]:


df[['Contract','tenure']].groupby('Contract').mean()


# As expected, contract and tenure are highly correlated. Customers with long contracts have been a customer for longer time than customers with short-term contracts. I think contract will add little to no value to tenure feature so I will not use contract feature in the model.

# After exploring the variables, I have decided not to use following variable because they add little or no informative power to the model:
# 1) Customer ID
# 2) Gender
# 3) PhoneService
# 4) Contract
# 5) TotalCharges

# In[44]:


df.drop(['customerID','gender','PhoneService','Contract','TotalCharges'], axis=1, inplace=True)


# In[45]:


df.head()


# # Data Preprocessing

# Categorical features need to be converted to numbers so that they can be included in calculations done by a machine learning model. The categorical variables in our data set are not ordinal (i.e. there is no order in them). For example, "DSL" internet service is not superior to "Fiber optic" internet service. An example for an ordinal categorical variable would be ratings from 1 to 5 or a variable with categories "bad", "average" and "good". 
# 
# When we encode the categorical variables, a number will be assigned to each category. The category with higher numbers will be considered more important or effect the model more. Therefore, we need to do encode the variables in a way that each category will be represented by a column and the value in that column will be 0 or 1.
# 
# We also need to scale continuous variables. Otherwise, variables with higher values will be given more importance which effects the accuracy of the model.

# In[46]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler


# In[47]:


cat_features = ['SeniorCitizen', 'Partner', 'Dependents',
        'MultipleLines', 'InternetService', 'OnlineSecurity',
       'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
       'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
X = pd.get_dummies(df, columns=cat_features, drop_first=True)


# In[48]:


sc = MinMaxScaler()
a = sc.fit_transform(df[['tenure']])
b = sc.fit_transform(df[['MonthlyCharges']])


# In[49]:


X['tenure'] = a
X['MonthlyCharges'] = b


# In[50]:


X.shape


# # Resampling

# As we briefly discussed in the beginning, target variables with imbalanced class distribution is not desired for machine learning models. I will use upsampling which means increasing the number of samples of the class with less samples by randomly selecting rows from it.

# In[51]:


sns.countplot('Churn', data=df).set_title('Class Distribution Before Resampling')


# In[52]:


X_no = X[X.Churn == 0]
X_yes = X[X.Churn == 1]


# In[53]:


print(len(X_no),len(X_yes))


# In[54]:


X_yes_upsampled = X_yes.sample(n=len(X_no), replace=True, random_state=42)
print(len(X_yes_upsampled))


# In[55]:


X_upsampled = X_no.append(X_yes_upsampled).reset_index(drop=True)


# In[56]:


sns.countplot('Churn', data=X_upsampled).set_title('Class Distribution After Resampling')


# # ML model

# We need to divide the data set into training and test subsets so that we are able to measure the performance of our model on new, previously unseen examples.

# In[57]:


from sklearn.model_selection import train_test_split


# In[58]:


X = X_upsampled.drop(['Churn'], axis=1) #features (independent variables)
y = X_upsampled['Churn'] #target (dependent variable)


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)


# ### Ridge Classifier

# I have decided to use ridge classifier as a base model. Then I will try a model that I think will perform better.

# In[60]:


from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[61]:


clf_ridge = RidgeClassifier() #create a ridge classifier object
clf_ridge.fit(X_train, y_train) #train the model


# In[62]:


pred = clf_ridge.predict(X_train)  #make predictions on training set


# In[63]:


accuracy_score(y_train, pred) #accuracy on training set


# In[64]:


confusion_matrix(y_train, pred)


# In[65]:


pred_test = clf_ridge.predict(X_test)


# In[66]:


accuracy_score(y_test, pred_test)


# The model achieved 75% accuracy on training set and 76% accuracy on test set. The model is not overfitting because accuracies on training and test sets are pretty close. However, 75% accuracy is not very good so we will try to get a better accuracy using a different model.

# ### Random Forests

# In[67]:


from sklearn.ensemble import RandomForestClassifier


# In[68]:


clf_forest = RandomForestClassifier(n_estimators=100, max_depth=10)


# In[69]:


clf_forest.fit(X_train, y_train)


# In[70]:


pred = clf_forest.predict(X_train)


# In[71]:


accuracy_score(y_train, pred)


# In[72]:


confusion_matrix(y_train, pred)


# In[73]:


pred_test = clf_forest.predict(X_test)


# In[74]:


print(accuracy_score(y_test, pred_test))





import joblib

filename='final_model.sav'
joblib.dump(clf_forest,filename)

# In[ ]:





# ## How to improve

# We can always try to improve the model. The fuel of machine learning models is data so if we can collect more data, it is always helpful in improving the model. We can also try a wider range of parameters in GridSearchCV because a little adjustment in a parameter may slighlty increase the model.
# 
# Finally, we can try more robust or advanced models. Please keep in mind that there will be a trade-off when making such kind of decisions. Advanced models may increase the accuracy but they require more data and more computing power. So it comes down to business decision.

# In[ ]:




