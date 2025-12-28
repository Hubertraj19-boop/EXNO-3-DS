## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
import pandas as pd 
df= pd.read_csv("Encoding Data.csv")
df
<img width="348" height="424" alt="image" src="https://github.com/user-attachments/assets/c66ac10b-0baa-40e3-991d-355081d2839d" />

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm= ['Hot','Warm','Cold']
e1= OrdinalEncoder (categories=[pm])
e1.fit_transform (df[["ord_2"]])
<img width="239" height="257" alt="image" src="https://github.com/user-attachments/assets/ac36cfbf-af6e-4065-8ec3-46c9554e0a86" />

df['bo2']= e1.fit_transform(df[["ord_2"]])
df
<img width="408" height="416" alt="image" src="https://github.com/user-attachments/assets/ef75411d-b1fd-4e4e-ba7c-e3141f666bd0" />

le= LabelEncoder()
dfc= df.copy()
dfc['ord_2']=le.fit_transform (dfc['ord_2'])
dfc
<img width="426" height="436" alt="image" src="https://github.com/user-attachments/assets/c5953ace-420e-419d-ac60-49c8c76fb336" />

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2= pd.concat([df2,enc],axis=1)
df2
<img width="532" height="422" alt="image" src="https://github.com/user-attachments/assets/7d96b0a2-7f87-4f8e-ac47-a2c358bb33cf" />

pd.get_dummies(df2,columns=["nom_0"])
<img width="809" height="440" alt="image" src="https://github.com/user-attachments/assets/5a0c809a-e8da-494b-870f-7f1449bd7bda" />

from category_encoders import BinaryEncoder
df= pd.read_csv("data.csv")
df
<img width="571" height="424" alt="image" src="https://github.com/user-attachments/assets/e6705bef-efef-4d1a-82fd-d34407f7fdbe" />

be= BinaryEncoder()
nd= be.fit_transform(df['Ord_2'])
df
<img width="600" height="435" alt="image" src="https://github.com/user-attachments/assets/138ea121-2d11-484b-adeb-3315de2825f5" />

dfb= pd.concat([df,nd],axis=1)
dfb
<img width="848" height="440" alt="image" src="https://github.com/user-attachments/assets/23468054-0976-42e3-8c0a-d33695ff348a" />

from category_encoders import TargetEncoder
te= TargetEncoder()
CC= df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC= pd.concat([CC,new],axis=1)
CC
<img width="666" height="428" alt="image" src="https://github.com/user-attachments/assets/8deb9f0f-2b18-4b62-bcaa-bc7efc1d468a" />

import pandas as pd 
import numpy as np
from scipy import stats 
df= pd.read_csv("Data_to_Transform.csv")
df
<img width="899" height="497" alt="image" src="https://github.com/user-attachments/assets/56d6d51f-9076-4eea-a482-3efc609ec685" />

df.skew()
<img width="416" height="145" alt="image" src="https://github.com/user-attachments/assets/ec86098e-3e4d-4d28-9b6a-41fab3c97c4b" />

np.log(df["Highly Positive Skew"])
<img width="661" height="311" alt="image" src="https://github.com/user-attachments/assets/f4377737-1bfa-431f-8d65-cf428ff346e6" />

np.reciprocal(df["Moderate Positive Skew"])
<img width="715" height="310" alt="image" src="https://github.com/user-attachments/assets/730faf2c-a0a3-4f2f-9fc7-4ce7dd6d5302" />

np.sqrt(df["Highly Positive Skew"])
<img width="701" height="298" alt="image" src="https://github.com/user-attachments/assets/ac65c3f1-84d3-4bb5-a425-036eda83fd29" />


np.square(df["Highly Positive Skew"])
<img width="672" height="314" alt="image" src="https://github.com/user-attachments/assets/b158e0ca-2e71-4b4f-b377-0e35819d3a14" />

df["Highly Positive Skew_boxcox"],parameters= stats.boxcox(df["Highly Positive Skew"])
df
<img width="1039" height="458" alt="image" src="https://github.com/user-attachments/assets/4ebc6d3c-df12-430b-a733-f323fa7ebf55" />

df.skew()
<img width="478" height="178" alt="image" src="https://github.com/user-attachments/assets/45774c31-e3e5-40df-af56-7ede6bb1f4a9" />

df["Highly Negative Skew_yeojohnson"],parameters= stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
<img width="519" height="196" alt="image" src="https://github.com/user-attachments/assets/fbd28d28-943a-42bf-97f2-43cada2b11f9" />
<img width="524" height="192" alt="image" src="https://github.com/user-attachments/assets/21cdaba6-cf58-43a5-99a9-d9908b76413a" />

 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal')
 df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 df
 <img width="1039" height="389" alt="image" src="https://github.com/user-attachments/assets/99ba3e73-da17-4417-bb5f-4765c5035ab5" />

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
<img width="873" height="631" alt="image" src="https://github.com/user-attachments/assets/3e66700f-b939-4331-a249-f19673e04059" />


sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
<img width="860" height="609" alt="image" src="https://github.com/user-attachments/assets/c44f02be-3e13-4763-b848-24e15d0f4bba" />

 from sklearn.preprocessing import QuantileTransformer
 qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
 df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
 sm.qqplot(df["Moderate Negative Skew"],line='45')
 plt.show()
 <img width="862" height="622" alt="image" src="https://github.com/user-attachments/assets/a1d8b742-4f3e-4a41-b9f9-0f223a25181a" />

 df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
 sm.qqplot(df["Highly Negative Skew"],line='45')
 plt.show()
 <img width="887" height="623" alt="image" src="https://github.com/user-attachments/assets/95c96f01-8ae7-4308-8ae3-f7690206b7b6" />

dt =pd.read_csv("titanic_dataset.csv")
dt
dt=pd.read_csv("titanic_dataset.csv")
dt
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
dt["Age_1"]=qt.fit_transform(dt[["Age"]])
sm.qqplot(dt['Age'],line='45') 
plt.show()
<img width="881" height="627" alt="image" src="https://github.com/user-attachments/assets/55591b90-833e-40a8-ae27-3aede1baa5c2" />

sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
<img width="884" height="626" alt="image" src="https://github.com/user-attachments/assets/42c305bd-5e46-4b52-888c-c2a2d41c5ed5" />

# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully

       
