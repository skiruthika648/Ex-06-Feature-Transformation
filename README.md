# Ex-06-Feature-Transformation
# AIM
To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM
# STEP 1
Read the given Data

# STEP 2
Clean the Data Set using Data Cleaning Process

# STEP 3
Apply Feature Transformation techniques to all the features of the data set

# STEP 4
Save the data to the file

# CODE
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.head()

df.isnull().sum()

df.info()

df.describe()

df1 = df.copy()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()
```
# OUPUT
# Dataset:
![image](https://user-images.githubusercontent.com/128348968/232389381-75482699-32bf-40f9-85ac-48c6a3ab0173.png)
# Head:
![image](https://user-images.githubusercontent.com/128348968/232389863-b1b69142-43fe-430d-a6ca-4eb7233cbe6f.png)
# Null data:
![image](https://user-images.githubusercontent.com/128348968/232390069-0e92b21b-77cd-45ae-8a49-259505e0ba67.png)
# Information:
![image](https://user-images.githubusercontent.com/128348968/232390198-fdb57fa4-2b81-4f50-9688-d9b9855fd332.png)
# Description:
![image](https://user-images.githubusercontent.com/128348968/232391246-a53aa5d0-9133-4937-abe2-d8f97174c183.png)
# Highly Positive Skew:
![image](https://user-images.githubusercontent.com/128348968/232391398-d3c4344f-5680-4d9f-bf37-292387049c17.png)
# Highly Negative Skew:
![image](https://user-images.githubusercontent.com/128348968/232391533-7a4058d3-9b24-4411-bfd9-df27c28f8da9.png)
# Moderate Positive Skew:
![image](https://user-images.githubusercontent.com/128348968/232391668-9ab4cb3a-fada-4e0d-8aaf-df8551baa4a7.png)
# Moderate Negative Skew:
![image](https://user-images.githubusercontent.com/128348968/232391981-8b30920c-5c02-4f87-9ccb-c3dead3df8a8.png)
# Log of Highly Positive Skew:
![image](https://user-images.githubusercontent.com/128348968/232392078-f2c08fb3-37b0-4228-b2cc-30e8a3f55689.png)
# Log of Moderate Positive Skew:
![image](https://user-images.githubusercontent.com/128348968/232392188-b0ac2ba1-96c5-48ae-9d84-cdd87340744f.png)
# Reciprocal of Highly Positive Skew:
![image](https://user-images.githubusercontent.com/128348968/232392373-b00ff2e7-cc6e-4a84-9bc0-0ab4cb97cb00.png)
# Square root tranformation:
![image](https://user-images.githubusercontent.com/128348968/232392513-bf19d035-f543-4937-b61a-d8a39f039dd9.png)
# Power transformation of Moderate Positive Skew:
![image](https://user-images.githubusercontent.com/128348968/232392656-a0a5ea22-728a-44ec-8013-8401b37bf1cd.png)
# Power transformation of Moderate Negative Skew:
![image](https://user-images.githubusercontent.com/128348968/232392749-8171b3ab-53fa-4e6d-88de-0431a06132a0.png)
# Quantile transformation:
![image](https://user-images.githubusercontent.com/128348968/232392809-cbc51a18-5e2a-45e9-a7c9-7a41b233a41b.png)
# Result
Thus, Feature transformation is performed and executed successfully for the given dataset
