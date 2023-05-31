# %%
# IMPORT PACKAGES
import pandas as pd
import numpy as np

# %%
# READ DATASET
df_cab = pd.read_csv('rideshare_kaggle.csv')
df_cab.head()

# %%
df_cab.shape

# %%
# Checking missing data
total_na = df_cab.isnull().sum().sum()
print(f'Total Missing Data: {total_na}')

# %%
# DROP THE MISSING DATA
df_clean_cab = df_cab.dropna(axis=0)

# %%
df_clean_cab.shape

# %%
df_clean_cab.dtypes

# %%
# TRANSFORM NOMINAL FEATURES
car_model = np.unique(df_clean_cab['name'])
car_model

# %%
from sklearn.preprocessing import LabelEncoder
# LabelEncoder based on alphabetical order
labels = LabelEncoder()
car_labels = labels.fit_transform(df_clean_cab['name'])
car_mapping = {index: label for index, label in enumerate(labels.classes_)}

car_mapping

# %%
df_clean_cab['car_label'] = car_labels
df_clean_cab[['price','distance','name','car_label']].iloc[1:10]

# %%
# DATASET DOESNT CONTAIN ANY ORDINAL FEATURES, 
# SKIP ORDINAL TRANSFORMATION

# %%
# ENCODE CATEGORICAL FEATURES
# ONE HOT ENCODING (OHE) on Source
source_le = LabelEncoder()
source_labels = source_le.fit_transform(df_clean_cab['source'])
source_mapping = {index: label for index, label in enumerate(source_le.classes_)}

source_mapping


# %%
df_clean_cab['source_label'] = source_labels

# %%
cab_df_sub = df_clean_cab[['source','source_label']]

# %%
cab_df_sub

# %%
from sklearn.preprocessing import OneHotEncoder

source_ohe = OneHotEncoder()
source_feature_arr = source_ohe.fit_transform(df_clean_cab[['source_label']]).toarray()
source_feature_labels = list(source_le.classes_)
source_feature = pd.DataFrame(source_feature_arr, columns=source_feature_labels)

# %%
df_ohe = pd.concat([cab_df_sub, source_feature], axis=1)

# %%
df_ohe.iloc[8:15]

# %%
new_source_df = pd.DataFrame([['Malika','North End'],
                            ['Mika','West End'],
                            ['Ikea','Fenway']],
                            columns=['cust_name','source'])
new_source_df

# %%
new_source_labels = source_le.transform(new_source_df['source'])
new_source_df['source_label'] = new_source_labels

new_source_df

# %%
new_source_feature_arr = source_ohe.transform(new_source_df[['source_label']]).toarray()
new_source_features = pd.DataFrame(new_source_feature_arr, columns=source_feature_labels)

new_source_ohe = pd.concat([new_source_df, new_source_features],axis=1)

# %%
new_source_ohe

# %%
# DUMMY CODING
source_ohe_features = pd.get_dummies(df_clean_cab['source'])
pd.concat([df_clean_cab[['source']], source_ohe_features], axis=1)

# %%
# DUMMY CODING DROP FIRST
source_ohe_features = pd.get_dummies(df_clean_cab['source'])
source_dummy_features = source_ohe_features.iloc[:, :-1]
pd.concat([df_clean_cab[['source']], source_dummy_features], axis=1)

# %%
# EFFECT CODING SCHEME
source_ohe_features = pd.get_dummies(df_clean_cab['source'])
source_effect_features = source_ohe_features.iloc[:,:-1]
source_effect_features.loc[np.all(source_effect_features == 0, axis=1)] = -1

# %%
pd.concat([df_clean_cab[['source']], source_effect_features],axis=1)

# %%
# FEATURE HASHING SCHEME
unique_weather = np.unique(df_clean_cab[['short_summary']])
print("Total weather category: ", len(unique_weather))
print(unique_weather)

# %%
weather = df_clean_cab['short_summary']

# %%
df_weather = pd.DataFrame(weather, columns=['short_summary'])
df_weather

# %%
from category_encoders import HashingEncoder

encoder = HashingEncoder(cols='short_summary', n_components=4)
hashed_features = encoder.fit_transform(df_weather)

# %%
df_clean_cab['short_summary']

# %%
hashed_features.iloc[45:60]

# %%
data = pd.concat([df_clean_cab['short_summary'], hashed_features], axis=1)

# %%
data.iloc[45:60]

# %%
encoder.get_params()


