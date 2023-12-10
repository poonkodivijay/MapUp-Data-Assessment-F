#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix 
 


# In[3]:


df=pd.read_csv("C:/Users/ADMIN/Desktop/Narashima/Mapup/dataset-3.csv")


# In[14]:


df


# In[73]:


df.columns


# In[74]:


from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.spatial import distance_matrix


# In[75]:


pd.DataFrame(distance_matrix(df.values,df.values),index=df.id_start,columns=df.id_end)


# #Create a function unroll_distance_matrix that takes the DataFrame created in Question 1. The resulting DataFrame should have three columns: columns id_start, id_end, and distance.
# 
# All the combinations except for same id_start to id_end must be present in the rows with their distance values from the input DataFrame.

# In[77]:


def create_distance_matrix(dataframe, id_start_col='id_start', id_end_col='id_end'):
    """
    Create a distance matrix based on the given DataFrame.

    Parameters:
    - dataframe: pd.DataFrame
        The DataFrame containing the data for which the distance matrix is to be calculated.
    - id_start_col: str, optional (default='id_start')
        The name of the column representing the start IDs.
    - id_end_col: str, optional (default='id_end')
        The name of the column representing the end IDs.

    Returns:
    - pd.DataFrame
        The distance matrix with index and columns set from the specified start and end ID columns.
    """
    if id_start_col not in dataframe.columns or id_end_col not in dataframe.columns:
        raise ValueError(f"Columns '{id_start_col}' and '{id_end_col}' must be present in the DataFrame.")

    # Extract relevant columns for distance matrix calculation
    subset_df = df[[id_start_col, id_end_col]]

    # Drop duplicates to get unique IDs
    unique_ids = pd.concat([subset_df[id_start_col], subset_df[id_end_col]]).unique()

    # Create a DataFrame with unique IDs as index and columns
    distance_matrix_df = pd.DataFrame(index=unique_ids, columns=unique_ids)

    # Fill the distance matrix with calculated distances
    distances = distance_matrix(subset_df.values, subset_df.values)
    distance_matrix_df.loc[:, :] = distances

    return distance_matrix_df

# Example usage:
# Replace 'your_dataframe' with the actual name of your DataFrame
# distance_matrix_result = create_distance_matrix(your_dataframe)


# In[78]:


def unroll_distance_matrix(distance_matrix_df):
    """
    Unroll a distance matrix DataFrame into a long-format DataFrame.

    Parameters:
    - distance_matrix_df: pd.DataFrame
        The distance matrix DataFrame.

    Returns:
    - pd.DataFrame
        Long-format DataFrame with columns 'id_start', 'id_end', and 'distance'.
    """
    # Ensure that the input DataFrame is a square matrix
    if distance_matrix_df.shape[0] != distance_matrix_df.shape[1]:
        raise ValueError("Input DataFrame must be a square distance matrix.")

    # Extract unique IDs from the index
    unique_ids = distance_matrix_df.index

    # Initialize lists to store unrolled data
    id_start_list = []
    id_end_list = []
    distance_list = []

    # Iterate over unique IDs to create combinations
    for id_start in unique_ids:
        for id_end in unique_ids:
            # Exclude combinations where id_start equals id_end
            if id_start != id_end:
                id_start_list.append(id_start)
                id_end_list.append(id_end)
                distance_list.append(distance_matrix_df.loc[id_start, id_end])

    # Create the unrolled DataFrame
    unrolled_df = pd.DataFrame({'id_start': id_start_list, 'id_end': id_end_list, 'distance': distance_list})

    return unrolled_df


# In[79]:


def unroll_distance_matrix(distance_matrix):
    distance_df = distance_matrix.rename_axis('id_start').reset_index()
    unrolled_df = distance_df.melt(id_vars='id_start', var_name='id_end', value_name='distance')
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]
    return unrolled_df.reset_index(drop=True)



# In[80]:


def find_ids_within_ten_percentage_threshold(df, reference_value):
    avg_distance_reference = df[df['id_start'] == reference_value]['distance'].mean()
    lower_bound = avg_distance_reference * 0.9
    upper_bound = avg_distance_reference * 1.1
    within_threshold = df[(df['id_start'] != reference_value) & 
                          (df['distance'] >= lower_bound) & 
                          (df['distance'] <= upper_bound)]['id_start'].unique()
    return sorted(within_threshold)


# In[81]:


def calculate_toll_rate(distance_matrix):
    toll_df = distance_matrix.copy()
    toll_df['moto'] = toll_df.apply(lambda row: row * 0.8 if row.name != row.index else 0, axis=1)
    toll_df['car'] = toll_df.apply(lambda row: row * 1.2 if row.name != row.index else 0, axis=1)
    toll_df['rv'] = toll_df.apply(lambda row: row * 1.5 if row.name != row.index else 0, axis=1)
    toll_df['bus'] = toll_df.apply(lambda row: row * 2.2 if row.name != row.index else 0, axis=1)
    toll_df['truck'] = toll_df.apply(lambda row: row * 3.6 if row.name != row.index else 0, axis=1) 
    return toll_df



# In[82]:


def calculate_time_based_toll_rates(df):
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    weekday_morning = pd.to_datetime('10:00:00').time()
    weekday_evening = pd.to_datetime('18:00:00').time()
    def apply_discount(row):
        if row['start_time'].weekday() < 5:  # Weekdays (Monday - Friday)
            if row['start_time'].time() < weekday_morning:
                return row * 0.8
            elif row['start_time'].time() < weekday_evening:
                return row * 1.2
            else:
                return row * 0.8
        else: 
            return row * 0.7
    vehicles = ['moto', 'car', 'rv', 'bus', 'truck']
    for vehicle in vehicles:
        df[vehicle] = df[vehicle].apply(apply_discount)
    days_of_week = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    df['start_day'] = df['start_time'].dt.weekday.map(days_of_week)
    df['end_day'] = df['end_time'].dt.weekday.map(days_of_week)
    df['start_time'] = df['start_time'].dt.time
    df['end_time'] = df['end_time'].dt.time
    return df


# In[ ]:





# In[ ]:




