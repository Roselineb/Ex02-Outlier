# Ex02-Outlier

You are given bhp.csv which contains property prices in the city of banglore, India. You need to examine price_per_sqft column and do following,

(1) Remove outliers using IQR 

(2) After removing outliers in step 1, you get a new dataframe.

(3) use zscore of 3 to remove outliers. This is quite similar to IQR and you will get exact same result

(4) for the data set height_weight.csv find the following

    (i) Using IQR detect weight outliers and print them

    (ii) Using IQR, detect height outliers and print them
# Explanation
An Outlier is an observation in a given dataset that lies far from the rest of the observations. That means an outlier is vastly larger or smaller than the remaining values in the set. An outlier is an observation of a data point that lies an abnormal distance from other values in a given population. (odd man out).Outliers badly affect mean and standard deviation of the dataset. These may statistically give erroneous results.Most machine learning algorithms do not work well in the presence of outlier. So it is desirable to detect and remove outliers.Outliers are highly useful in anomaly detection like fraud detection where the fraud transactions are very different from normal transactions.
# ALGORITHM
## STEP 1
Read the given Data

## STEP 2
Get the information about the data

## STEP 3
Detect the Outliers using IQR method and Z score

## STEP 4
Remove the outliers

## STEP 5
Plot the datas using Box Plot

# CODE
## (1) & (2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
```
import pandas as pd
import numpy as np
import seaborn as sns

df = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\bhp.csv")
df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.shape

sns.boxplot(x="price_per_sqft",data=df)
```
```
q1 = df['price_per_sqft'].quantile(0.25)
q3 = df['price_Aper_sqft'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df1 =df[((df['price_per_sqft']>=ll)&(df['price_per_sqft']<=ul))]
df1

df1.shape

sns.boxplot(x="price_per_sqft",data=df1)
```
## (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
```
from scipy import stats

z = np.abs(stats.zscore(df['price_per_sqft']))
df2 = df[(z<3)]
df2

print(df2.shape)
sns.boxplot(x="price_per_sqft",data=df2)
```
## (4)(i) For the data set height_weight.csv detect weight outliers using IQR method
```
df3 = pd.read_csv("C:\Users\chief\OneDrive\Documents\Ex02-Outlier\height_weight.csv")
df3

df3.head()

df3.info()

df3.describe()

df3.isnull().sum()

df3.shape
```
```
sns.boxplot(x="weight",data=df3)

q1 = df3['weight'].quantile(0.25)
q3 = df3['weight'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df4 =df3[((df3['weight']>=ll)&(df3['weight']<=ul))]
df4

df4.shape

sns.boxplot(x="weight",data=df4)
```
## (4)(ii) For the data set height_weight.csv detect height outliers using IQR method
```
sns.boxplot(x="height",data=df3)

q1 = df3['height'].quantile(0.25)
q3 = df3['height'].quantile(0.75)
print("First Quantile =",q1,"\nSecond Quantile =",q3)

IQR = q3-q1
ul = q3+1.5*IQR
ll = q1-1.5*IQR

df5 =df3[((df3['height']>=ll)&(df3['height']<=ul))]
df5

df5.shape

sns.boxplot(x="height",data=df5)
```
# OUTPUT
## (1)(2) Examine price_per_sqft column and use IQR to remove outliers and create new dataframe
### Dataset
# ![image](https://user-images.githubusercontent.com/128909895/228261620-347d0194-dc7d-4db4-a55b-4438401fe60f.png)
### Dataset Head
# ![image](https://user-images.githubusercontent.com/128909895/228261964-66532bf1-7165-4109-9242-131558e94e64.png)
### Dataset Info
# ![image](https://user-images.githubusercontent.com/128909895/228262077-293e6483-044d-490d-a279-04562ce3fd72.png)
### Dataset Describe
# ![image](https://user-images.githubusercontent.com/128909895/228262287-fed1a1e0-d558-4c41-9ab3-fa6093bc3179.png)
### Null Values
# ![image](https://user-images.githubusercontent.com/128909895/228262411-5a7c8149-5707-4d0f-ad1b-febb53a7b7c5.png)
### Dataset Shape
# ![image](https://user-images.githubusercontent.com/128909895/228262621-495044f9-2669-45fb-b0ba-294573e711b9.png)
### Box plot of price_per_sqft column with outliers
# ![image](https://user-images.githubusercontent.com/128909895/228262747-f1bb2979-570b-451c-9acc-2c0ee45035fd.png)
### price_per_sqft - Dataset after removing outliers
# ![image](https://user-images.githubusercontent.com/128909895/228262881-52eac7ba-569e-49f8-99c5-0c518ba060cd.png)
### price_per_sqft - Shape of Dataset after removing outliers
# ![image](https://user-images.githubusercontent.com/128909895/228263776-c2687f80-9386-4dd7-85bd-57060fe7c2d2.png)
### Box Plot of price_per_sqft column without outliers
# ![image](https://user-images.githubusercontent.com/128909895/228263908-ea639a48-e4f8-45c3-bb70-a2c6bf46f7c7.png)
## (3) Examine price_per_sqft column and use zscore of 3 to remove outliers.
### Dataset after removal of outlier using z score
# ![image](https://user-images.githubusercontent.com/128909895/228264127-980a4fbd-a575-400e-9463-39b739786dfb.png)
### Shape of Dataset after removal of outlier using z score
# ![image](https://user-images.githubusercontent.com/128909895/228264260-a315ee92-1904-4203-9357-da69fccbffd5.png)
### price_per_sqft column after removing outliers
# ![image](https://user-images.githubusercontent.com/128909895/228264430-20480634-cf08-439e-8930-eb6daa00623a.png)
## (4) For the data set height_weight.csv detect weight and height outliers using IQR method
### Dataset
# ![image](https://user-images.githubusercontent.com/128909895/228264683-e7263c90-fb68-48eb-9146-bd6845646d67.png)
### Dataset Head
# ![image](https://user-images.githubusercontent.com/128909895/228264788-c699d423-95be-4be0-a754-5180fcc35079.png)
### Dataset Info
# ![image](https://user-images.githubusercontent.com/128909895/228264899-f1cdfbb0-b3dc-4966-9600-8f55163fb9ff.png)
### Dataset Describe
# ![image](https://user-images.githubusercontent.com/128909895/228265023-80c66cb8-10d2-4273-9b26-7f1de638c181.png)
### Null Values
# ![image](https://user-images.githubusercontent.com/128909895/228265122-a1f989c0-56b9-40e6-9050-ed40cb6ed787.png)
### Dataset Shape
# ![image](https://user-images.githubusercontent.com/128909895/228265245-57c8a91a-a8df-4f15-a74d-aa1428ff5c1c.png)
### Weight - With outliers
# ![image](https://user-images.githubusercontent.com/128909895/228265374-2a522421-45c2-4618-a6b8-c9f092baa29a.png)
### Weight - Dataset after removing Outliers using IQR method
# ![image](https://user-images.githubusercontent.com/128909895/228265544-ba0e5b93-0e73-41e2-836a-0bea91acee72.png)
### Weight - Shape of Dataset after removing Outliers using IQR method
# ![image](https://user-images.githubusercontent.com/128909895/228265654-280e240c-d716-4ac7-a875-0577b3b28060.png)
### Weight - Without Outliers using IQR method
# ![image](https://user-images.githubusercontent.com/128909895/228265781-e3fe78af-1561-4e79-b6d2-09ba846008c6.png)
### Height - With outliers
# ![image](https://user-images.githubusercontent.com/128909895/228265888-ee5b8f55-25b1-4c9d-9fbe-1c017a735603.png)
### Height - Dataset after removing Outliers using IQR method
# ![image](https://user-images.githubusercontent.com/128909895/228265995-30494d6a-aed3-400e-a1ea-b9a92444de77.png)
### Height - Shape of Dataset after removing Outliers using IQR method
# ![image](https://user-images.githubusercontent.com/128909895/228267159-6be2edd8-4b8a-4c41-82ad-af94781991b5.png)

### Height - Without Outliers using IQR method
# ![image](https://user-images.githubusercontent.com/128909895/228266212-45f7a444-39fb-4eaa-ba49-a796972341bd.png)
# RESULT
The given datasets are read and outliers are detected and are removed using IQR and z-score methods. nd print them































