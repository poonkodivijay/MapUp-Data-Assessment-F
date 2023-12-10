#!/usr/bin/env python
# coding: utf-8

# In[123]:


import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta



# In[9]:


dataset_1=pd.read_csv("C:/Users/ADMIN/Desktop/Narashima/Mapup/dataset-1.csv")


# In[10]:


dataset_1


# In[11]:


dataset_3=pd.read_csv("C:/Users/ADMIN/Desktop/Narashima/Mapup/dataset-3.csv")


# In[12]:


dataset_3


# In[34]:


#Under the function named generate_car_matrix write a logic that takes the dataset-1.csv as a DataFrame. Return a new DataFrame that follows the following rules:

#values from id_2 as columns
#values from id_1 as index
#dataframe should have values from car column
#diagonal values should be 0.



# In[23]:


def generate_car_matrix(file_path):
 
    df = pd.read_csv("C:/Users/ADMIN/Desktop/Narashima/Mapup/dataset-1.csv")

    # Pivot the DataFrame to create the desired matrix
    car_matrix = df.pivot(index='id_1', columns='id_2', values='car').fillna(0)

    # Set diagonal values to 0
    for i in car_matrix.index:
        if i in car_matrix.columns:
            car_matrix.loc[i, i] = 0

    return car_matrix

# Example usage:
file_path = 'dataset-1.csv'
result_matrix = generate_car_matrix(file_path)
print(result_matrix)


# In[ ]:





# In[ ]:


#Create a Python function named get_type_count that takes the dataset-1.csv as a DataFrame. Add a new categorical column car_type based on values of the column car:

low for values less than or equal to 15,
medium for values greater than 15 and less than or equal to 25,
high for values greater than 25.
Calculate the count of occurrences for each car_type category and return the result as a dictionary. Sort the dictionary alphabetically based on keys.


# In[25]:


def get_type_count(df):
    # Add a new column 'car_type' based on the conditions
    df['car_type'] = pd.cut(df['car'], bins=[float('-inf'), 15, 25, float('inf')],
                            labels=['low', 'medium', 'high'], right=False)

    # Calculate the count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().to_dict()

    # Sort the dictionary alphabetically based on keys
    sorted_type_counts = dict(sorted(type_counts.items()))

    return sorted_type_counts

# Example usage:
file_path = 'dataset-1.csv'
df = pd.read_csv("C:/Users/ADMIN/Desktop/Narashima/Mapup/dataset-1.csv")
result_dict = get_type_count(df)
print(result_dict)


# In[ ]:


#Create a Python function named get_bus_indexes that takes the dataset-1.csv as a DataFrame. The function should identify and return the indices as a list (sorted in ascending order) where the bus values are greater than twice the mean value of the bus column in the DataFrame.


# In[26]:


def get_bus_indexes(df):
    # Calculate the mean value of the 'bus' column
    mean_bus_value = df['bus'].mean()

    # Identify indices where the 'bus' values are greater than twice the mean
    bus_indexes = df[df['bus'] > 2 * mean_bus_value].index.tolist()

    # Sort the indices in ascending order
    bus_indexes.sort()

    return bus_indexes

# Example usage:
file_path = 'dataset-1.csv'
df = pd.read_csv("C:/Users/ADMIN/Desktop/Narashima/Mapup/dataset-1.csv")
result_indices = get_bus_indexes(df)
print(result_indices)


# In[ ]:


#Create a python function filter_routes that takes the dataset-1.csv as a DataFrame. The function should return the sorted list of values of column route for which the average of values of truck column is greater than 7.


# In[27]:


def filter_routes(df):
    # Calculate the average value of the 'truck' column for each route
    route_avg_truck = df.groupby('route')['truck'].mean()

    # Filter routes where the average of 'truck' values is greater than 7
    selected_routes = route_avg_truck[route_avg_truck > 7].index.tolist()

    # Sort the list of selected routes
    selected_routes.sort()

    return selected_routes

# Example usage:
file_path = 'dataset-1.csv'
df = pd.read_csv("C:/Users/ADMIN/Desktop/Narashima/Mapup/dataset-1.csv")
result_routes = filter_routes(df)
print(result_routes)


# In[ ]:


#Create a Python function named multiply_matrix that takes the resulting DataFrame from Question 1, as input and modifies each value according to the following logic:

If a value in the DataFrame is greater than 20, multiply those values by 0.75,
If a value is 20 or less, multiply those values by 1.25.
The function should return the modified DataFrame which has values rounded to 1 decimal place.


# In[28]:


def multiply_matrix(input_matrix):
    # Create a deep copy of the input matrix to avoid modifying the original
    modified_matrix = input_matrix.copy()

    # Apply the specified logic to each value in the matrix
    for i in modified_matrix.index:
        for j in modified_matrix.columns:
            if modified_matrix.loc[i, j] > 20:
                modified_matrix.loc[i, j] *= 0.75
            else:
                modified_matrix.loc[i, j] *= 1.25

    # Round the values to 1 decimal place
    modified_matrix = modified_matrix.round(1)

    return modified_matrix

# Example usage:
# Assuming result_matrix is the DataFrame from Question 1
# result_matrix = generate_car_matrix('dataset-1.csv')
modified_result_matrix = multiply_matrix(result_matrix)
print(modified_result_matrix)


# In[ ]:




