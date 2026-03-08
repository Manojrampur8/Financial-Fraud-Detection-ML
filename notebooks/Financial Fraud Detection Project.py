#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv("Cross_border.csv")
df.head(10)


# In[2]:


df.info()


# In[3]:


df.columns = df.columns.str.strip()
print(df.columns)


# In[4]:


df['Date'] = pd.to_datetime(df['Date'])
df['Time'] = pd.to_datetime(df['Time'])

df['hour'] = df['Time'].dt.hour
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month


# In[5]:


df = df.drop(['Date','Time'], axis=1)
df.head()


# In[6]:


df = df.drop(['Sender_account','Receiver_account'], axis=1)
df.head()


# In[7]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

categorical_columns = [
    'Payment_currency',
    'Received_currency',
    'Sender_bank_location',
    'Receiver_bank_location',
    'Payment_type'
]

for col in categorical_columns:
    df[col] = le.fit_transform(df[col])


# In[8]:


df.head()


# In[9]:


df = df.drop('Laundering_type', axis=1, errors='ignore')

X = df.drop('Is_laundering', axis=1)
y = df['Is_laundering']


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# In[15]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Original training shape:", X_train.shape)
print("Resampled training shape:", X_train_resampled.shape)


# In[11]:


print(X_train.shape)
print(X_test.shape)


# In[16]:


from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

model.fit(X_train_resampled, y_train_resampled)


# In[17]:


y_pred = model.predict(X_test)


# In[18]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


# In[19]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Money Laundering Detection")

plt.show()


# In[20]:


importance = pd.Series(model.feature_importances_, index=X.columns)

importance.sort_values().plot(kind='barh', figsize=(8,5))

plt.title("Feature Importance in Money Laundering Detection")
plt.show()


# In[21]:


from sklearn.metrics import roc_curve, auc

y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_prob)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label="ROC curve (AUC = %0.2f)" % roc_auc)
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve for Money Laundering Detection")
plt.legend()

plt.show()


# In[ ]:




