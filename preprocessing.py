##The objective for this project is to correctly classify which potential customers would purchase transmission services
##In addition, the company would like to know the likelihood of purchase for each.

#Import data set
import pandas as pd
df = pd.read_excel('Ralph's Transmission Vehicle Breakdown_Data.xlsx')

#Display information summary and first rows
print(df.shape)
print(df.info())
print(df.head())

#Correct Dtypes: 'DateIn' and 'ComletedDate' change to datetime objects
df['DateIn'] = pd.to_datetime(df.DateIn)
df['CompletedDate'] = pd.to_datetime(df.CompletedDate)
print(df.info())

#Engineer new columns: Wait_Time = time between datein and completed date
#Buy = True/False is 'total earned' > 0
#Car_Age = Time between Year and DateIn
#Drop How_Long, Parts, Labor, and Tax
#Drop used date fields, otherwise LogisticRegression will error

df['Wait_Time'] = pd.DatetimeIndex(df['CompletedDate']).day - pd.DatetimeIndex(df['DateIn').day
df['Car_Age'] = pd.DatetimeIndex(df['DateIn']).year - df.Year
df['Buy'] = df['Total Earned'] > 0
df = df.drop(['How Long', 'Job1 Labor', 'Job1 Parts', 'Jobs 2-3-4 Labor', 'Jobs 2-3-4 Parts', 
              'Sublet', 'Tax', 'CompletedDate', 'DateIn'], axis = 1)
df.info()

#View the proportion and count of Nulls in each column
print(df.isnull().mean().sort_values(ascending = False))
print(df.isnull().sum().sort_values(ascending = False))

#To impute MilesIn let's use the average MilesIn, grouped by Car_Age
df['MilesIn'] = df.groupby('Car_Age')['MilesIn'].transform(lambda x: x.fillna(x.mean()))
df_drop = df.dropna()
print(df_drop.isnull().sum().sort_values(ascending = False))

##Note. In future versions we can similarly impute axle, engine and transmission by the most common entries grouped by make and model,
##but this is slightly more difficult with categorical data. So for now we have dropped the rest of rows with null values.
