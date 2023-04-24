#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np

# Script to combine all signals
# Set the data path
DATA_PATH = r"C:\Users\pc\Desktop\Data Science A2\SAAD\Stress-Predict-Dataset-main"

# Set the save path
SAVE_PATH = r"C:\Users\pc\Desktop\Data Science A2\SAAD"

# Create the save path directory if it doesn't exist
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

final_columns = {
    'ACC': ['id', 'X', 'Y', 'Z', 'datetime'],
    'BVP': ['id', 'BVP', 'datetime'],
    'EDA': ['id', 'EDA', 'datetime'],
    'HR': ['id', 'HR', 'datetime'],
    'IBI': ['id', 'Initial', 'Interval', 'datetime'],
    'TEMP': ['id', 'TEMP', 'datetime']
}

names = {
    'ACC.csv': ['X', 'Y', 'Z'],
    'BVP.csv': ['BVP'],
    'EDA.csv': ['EDA'],
    'HR.csv': ['HR'], 
    'IBI.csv': ['Initial', 'Interval'],
    'TEMP.csv': ['TEMP']
}

desired_signals = ['ACC.csv', 'BVP.csv', 'EDA.csv', 'HR.csv', 'IBI.csv', 'TEMP.csv']

acc = pd.DataFrame(columns=final_columns['ACC'])
bvp = pd.DataFrame(columns=final_columns['BVP'])
eda = pd.DataFrame(columns=final_columns['EDA'])
hr = pd.DataFrame(columns=final_columns['HR'])
ibi = pd.DataFrame(columns=final_columns['IBI'])
temp = pd.DataFrame(columns=final_columns['TEMP'])


def process_df(df, file):
    start_timestamp = df.iloc[0,0]
    sample_rate = df.iloc[1,0]
    new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
    new_df['id'] =  file[-2:]
    new_df['datetime'] = [(start_timestamp + i/sample_rate) for i in range(len(new_df))]
    return new_df

#Combine data with same names in multiple folders
for folder in os.listdir(DATA_PATH):
    folder_path = os.path.join(DATA_PATH, folder)
    if os.path.isdir(folder_path):
        print(f'Processing {folder}')
        for sub_folder in os.listdir(folder_path):
            sub_folder_path = os.path.join(folder_path, sub_folder)
            if os.path.isdir(sub_folder_path):
                print(f'Processing {sub_folder} in {folder}')
                for signal in os.listdir(sub_folder_path):
                    if os.path.isfile(os.path.join(sub_folder_path, signal)):
                        if signal in desired_signals:
                            df = pd.read_csv(os.path.join(sub_folder_path, signal), names=names[signal], header=None)
                            if not df.empty:
                                if signal == 'ACC.csv':
                                    acc = pd.concat([acc, process_df(df, folder)])
                                    print("ACC df:", acc)
                                if signal == 'BVP.csv':
                                    bvp = pd.concat([bvp, process_df(df, folder)])
                                    print("BVP df:", bvp)
                                if signal == 'EDA.csv':
                                    eda = pd.concat([eda, process_df(df, folder)])
                                    print("EDA df:", eda)
                                if signal == 'HR.csv':
                                    hr = pd.concat([hr, process_df(df, folder)])
                                    print("HR df:", hr)
                                if signal == 'IBI.csv':
                                    ibi = pd.concat([ibi, process_df(df, folder)])
                                    print("IBI df:", ibi)
                                if signal == 'TEMP.csv':
                                    temp = pd.concat([temp, process_df(df, folder)])
                                    print("TEMP df:", temp)
                            else:
                                print(f"{signal} in {sub_folder} is empty")
                    else:
                        print(f"{signal} in {sub_folder} is not a file")
            else:
                print(f"{sub_folder} in {folder} is not a directory")
    else:
        print(f"{folder} is not a directory")

#Saving combined data
print('Saving Data ...')
acc.to_csv(os.path.join(SAVE_PATH, 'combined_acc.csv'), index=False)
bvp.to_csv(os.path.join(SAVE_PATH, 'combined_bvp.csv'), index=False)
eda.to_csv(os.path.join(SAVE_PATH, 'combined_eda.csv'), index=False)
hr.to_csv(os.path.join(SAVE_PATH, 'combined_hr.csv'), index=False)
ibi.to_csv(os.path.join(SAVE_PATH, 'combined_ibi.csv'), index=False)
temp.to_csv(os.path.join(SAVE_PATH, 'combined_temp.csv'), index=False)


# In[3]:


import pandas as pd
import os

# Set the data path
COMBINED_DATA_PATH = r"C:\Users\pc\Desktop\Data Science A2\SAAD"

# Set the save path
SAVE_PATH = r"C:\Users\pc\Desktop\Data Science A2\SAAD"

# Create the save path directory if it doesn't exist 
if COMBINED_DATA_PATH != SAVE_PATH:
    os.mkdir(SAVE_PATH)
                                          
print("Reading data ...")
acc, bvp, eda, hr, ibi, temp = None, None, None, None, None, None

signals = ['acc','bvp', 'eda', 'hr','ibi', 'temp']

results = []
for signal in signals:
    df = pd.read_csv(os.path.join(COMBINED_DATA_PATH, f"combined_{signal}.csv"), dtype={'id': str})
    results.append([signal, df])

for i in results:
    globals()[i[0]] = i[1]

# Merge data
print('Merging Data ...')
ids = eda['id'].unique()
columns = ['X', 'Y', 'Z','BVP', 'EDA', 'HR','IBI', 'TEMP', 'id', 'datetime']

results = []
for id in ids:
    print(f"Processing {id}")
    df = pd.DataFrame(columns=columns)

    acc_id = acc[acc['id'] == id]
    bvp_id = bvp[bvp['id'] == id].drop(['id'], axis=1)
    eda_id = eda[eda['id'] == id].drop(['id'], axis=1)
    hr_id = hr[hr['id'] == id].drop(['id'], axis=1)
    ibi_id = ibi[ibi['id'] == id].drop(['id'], axis=1)
    temp_id = temp[temp['id'] == id].drop(['id'], axis=1)

    df = acc_id.merge(bvp_id, on='datetime', how='outer')
    df = df.merge(eda_id, on='datetime', how='outer')
    df = df.merge(hr_id, on='datetime', how='outer')
    df = df.merge(ibi_id, on='datetime', how='outer')
    df = df.merge(temp_id, on='datetime', how='outer')

    #Filling null values with Forward and Backward value imputation
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)

    results.append(df)

print(results)
new_df = pd.concat(results, ignore_index=True)

print("Saving data ...")
new_df.to_csv(os.path.join(SAVE_PATH, "merged_data.csv"), index=False)


# # Preprocessing and merging of all CSV into single CSV is done

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings


# In[2]:


warnings.filterwarnings('ignore')


# In[3]:


#Loading Already preprocessed data for labeling
preprocessed_data = pd.read_csv(r"C:\Users\pc\Desktop\Data Science A2\SAAD\Stress-Predict-Dataset-main\Processed_data\Improved_All_Combined_hr_rsp_binary.csv")


# In[4]:


new_df = pd.read_csv('merged_data.csv')


# In[5]:


preprocessed_data.head()


# In[6]:


new_df.head()


# In[7]:


new_df.rename({'datetime':'Time(sec)'},axis = 1,inplace = True)


# In[8]:


#Merging Both dataframe for getting label based on time
final_df = new_df.merge(preprocessed_data[['Label','Time(sec)']],on = 'Time(sec)',how = 'inner')


# In[9]:


final_df.head()


# In[10]:


"""Code taken from chatgpt """

# Convert the Unix time feature to datetime
final_df['datetime'] = pd.to_datetime(final_df['Time(sec)'], unit='s')


# In[13]:


"""Code taken from chatgpt"""

fig, ax = plt.subplots()

ax.plot(final_df.datetime, final_df['X'])
ax.plot(final_df.datetime, final_df['Y'])
ax.plot(final_df.datetime, final_df['Z'])

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')

plt.show()


# In[14]:


"""Code taken from chatgpt"""

fig, ax = plt.subplots()

ax.plot(final_df.datetime, final_df['BVP'])

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')

plt.show()


# In[15]:


"""Code taken from chatgpt"""

fig, ax = plt.subplots()

ax.plot(final_df.datetime, final_df['EDA'])

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')

plt.show()


# In[16]:


"""Code taken from chatgpt"""

fig, ax = plt.subplots()

ax.plot(final_df.datetime, final_df['HR'])

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')

plt.show()


# In[18]:


"""Code taken from chatgpt"""

fig, ax = plt.subplots()

ax.plot(final_df.datetime, final_df['Interval'])

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')

plt.show()


# In[19]:


"""Code taken from chatgpt"""

fig, ax = plt.subplots()

ax.plot(final_df.datetime, final_df['TEMP'])

ax.set_xlabel('Date')
ax.set_ylabel('Value')
ax.set_title('Time Series Plot')

plt.show()


# In[9]:


#Dropping unecessary columns which are not required

final_df.drop({'id','Time(sec)'},axis = 1,inplace = True)


# In[10]:


final_df.head()


# In[11]:


#Checking null values in the dataframe
final_df.isnull().sum()


# In[12]:


#No null value so visulization is empty ####################################
null_count = pd.DataFrame(final_df.isnull().sum(), columns=['Count'])
null_count.plot(kind='bar')
plt.show()


# In[13]:


#Checking duplicates
final_df.duplicated().sum()


# In[14]:


#Visulize duplicated values
labels = ['Duplicated Values', 'Non Duplicated values']
sizes = [final_df.duplicated().sum(), final_df.shape[0] - final_df.duplicated().sum()]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
plt.title('Duplicates Visulization')
plt.show()


# In[15]:


final_df.drop_duplicates(inplace = True)


# In[16]:


#Checking if the data is imbalanced or not
final_df['Label'].value_counts()


# In[17]:


#From the ratio of 0 to 1 we can clearly see the data is highly imbalanced


# In[18]:


sns.countplot(final_df['Label'])


# In[19]:


#Have to resample the data to balance the data


# In[20]:


from sklearn.utils import resample


# In[21]:


#Seperate the minority which is 1 and majority which is 0 so that we can downsample 0 and match to 1's length
majority = final_df[final_df['Label'] == 0]
minority = final_df[final_df['Label'] == 1]


# In[22]:


#Downsampling majority which is 0
downsampled_majority_data = resample(majority,replace=False,n_samples=len(minority),random_state=100)


# In[23]:


#finally concating downsampled majority and minority
final_data = pd.concat([downsampled_majority_data, minority])


# In[24]:


#reset index because the random sample has jumbled the indexs
final_data.reset_index(inplace = True,drop = True)


# In[25]:


final_data.head()


# In[26]:


final_data['Label'].value_counts()


# In[27]:


#The data is balanced now


# In[28]:


sns.countplot(final_data['Label'])


# In[29]:


#Visulizaing histogram to check if the data is normally distributed
final_data.hist(figsize =  (12,12))


# In[30]:


#Checking correlation of each attriobute to the dependent variable
final_data.corr()


# In[31]:


sns.heatmap(final_data.corr(), cmap='coolwarm', annot=True, fmt='.2f')

# Show the plot
plt.title('Correlation Heatmap')
plt.show()


# # Splitting the data into two parts, training and testing

# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


X = final_data.drop('Label', axis=1)  # Features
y = final_data['Label']  # Target variable

#Splitting the data into training 70% and testing 30%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # Model Building

# # Logistic Regression

# In[34]:


from sklearn.linear_model import LogisticRegression


# In[35]:


model1 = LogisticRegression(random_state=42).fit(X_train, y_train)
model1_pred = model1.predict(X_test)


# In[36]:


from sklearn.metrics import precision_score,recall_score,f1_score

model1_precision = precision_score(y_test,model1_pred)
model1_recall = recall_score(y_test,model1_pred)
model1_f1 = f1_score(y_test,model1_pred)

print('Precision of model 1 is:- ',model1_precision)
print('Recall of model 1 is:- ',model1_recall)
print('F1_score of model 1 is:- ',model1_f1)


# In[37]:


labels = ['Precision', 'Recall', 'F1_Score']
sizes = [model1_precision, model1_recall, model1_f1]

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(labels, sizes)
plt.title('Metrics Visulization of Logistic Regression')
plt.show()


# # Random Forest

# In[38]:


from sklearn.ensemble import RandomForestClassifier


# In[39]:


model2 = RandomForestClassifier(max_depth=2, random_state=42).fit(X_train, y_train)
model2_pred = model2.predict(X_test)


# In[40]:


model2_precision = precision_score(y_test,model2_pred)
model2_recall = recall_score(y_test,model2_pred)
model2_f1 = f1_score(y_test,model2_pred)

print('Precision of model 2 is:- ',model2_precision)
print('Recall of model 2 is:- ',model2_recall)
print('F1_score of model 2 is:- ',model2_f1)


# In[41]:


labels = ['Precision', 'Recall', 'F1_Score']
sizes = [model2_precision, model2_recall, model2_f1]

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(labels, sizes)
plt.title('Metrics Visulization of Random Forest')
plt.show()


# # Naive Bayes

# In[42]:


from sklearn.naive_bayes import GaussianNB


# In[43]:


model3 = GaussianNB().fit(X_train, y_train)
model3_pred = model3.predict(X_test)


# In[44]:


model3_precision = precision_score(y_test,model3_pred)
model3_recall = recall_score(y_test,model3_pred)
model3_f1 = f1_score(y_test,model3_pred)

print('Precision of model 3 is:- ',model3_precision)
print('Recall of model 3 is:- ',model3_recall)
print('F1_score of model 3 is:- ',model3_f1)


# In[45]:


labels = ['Precision', 'Recall', 'F1_Score']
sizes = [model3_precision, model3_recall, model3_f1]

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(labels, sizes)
plt.title('Metrics Visulization of Naive Bayes')
plt.show()


# In[ ]:




