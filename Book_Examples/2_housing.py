import os
datapath = os.path.join("Book_Examples","dataset", "housing")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from zlib import crc32
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

#############################
# Where to save the figures
PROJECT_ROOT_DIR = os.path.abspath(os.getcwd())
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH)

##################### Load Data
def load_housing_data(housing_path=datapath):
    csv_path = os.path.join(PROJECT_ROOT_DIR, housing_path, "housing.csv")
    print("Reading data from path", csv_path, "\n")
    return pd.read_csv(csv_path)

##################### Read the data and understand its structure

housing=load_housing_data()
print("\nFunc head() prints the first 5 rows of the dataset\n")
print(housing.head())

print("\n\nFunc info() prints - data description, no of rows, attribute type, no of non-null values\n")
print(housing.info())

print("\n\nCount different objects in the column [ocean_proximity] ")
print(housing["ocean_proximity"].value_counts())

print("\n\nFunc describe() summarizes the numerical values. NULL values are ignored\n")
print(housing.describe())

housing.hist(bins=50, figsize=(20,15))
#plt.show()


###################### Split the training data to save some for validation
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing, 0.2)
print("\nSize of trainning set =", len(train_set))
print("\nSize of trainning set =", len(test_set))

print("\nBetter way of splitting data")
def set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio


housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]# Adds the latitude to the ID to get more stable index
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)



##############################################################################################################################################################################################################################################################
####################################################################################################################################################################################################################################

############################## Discover and visualize the data to gain insights

housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=housing["population"]/100, label="population", figsize=(10,7), c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, sharex=False)
plt.legend()
plt.show()
