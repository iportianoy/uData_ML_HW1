#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import copy
from scipy import stats
from sklearn.linear_model import LinearRegression


# ### Delete nulls

# In[ ]:


def delete_nulls(df, axis='row'):
    true_headers = df.columns.values.tolist()
    matrix = copy.deepcopy(df.values)
    
    if axis == 'row':
        matrix_without_nulls = []
        for row in matrix:
            not_nulls = [item == item for item in row]
    
            if all(not_nulls):
                matrix_without_nulls.append(row)
                
        return pd.DataFrame(data=np.array(matrix_without_nulls), columns=true_headers)
    
    elif axis == 'col':
        matrix_without_nulls = []
        headers = []
        
        for col, index in zip(matrix.T, range(0, matrix.shape[1])):
            not_nulls = [item == item for item in col]
            
            if all(not_nulls):
                matrix_without_nulls.append(col)
                headers.append(true_headers[index])

        return pd.DataFrame(data=matrix_without_nulls, columns=headers)    
    
    else:
        print('Unknown axis name. Must be row or col')
        return df


# ### Replace nan

# In[4]:


def replace_nulls(df, value='mean'):
    headers = df.columns.values.tolist()
    matrix = copy.deepcopy(df.values)
    print(matrix)
    value_for_replace = None
    
    if value == 'mean':
        value_for_replace = np.nanmean(matrix, axis=0)
        
    elif value == 'median':
        value_for_replace = np.nanmedian(matrix, axis=0)    
        
    elif value == 'mode':
        value_for_replace = stats.mode(matrix)[0][0]
        
    else:
        print('Unknown mode name. Must be mean, mode or median')
        return df
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] != matrix[i, j]:
                matrix[i, j] = value_for_replace[j]
    
    return pd.DataFrame(data=matrix, columns=headers)  


# ### Replace nan with linear regression

# In[17]:


def replace_nulls_with_lr(df):
    headers = df.columns.values.tolist()
    matrix = copy.deepcopy(df.values)
    value_for_replace = None
    print(matrix[1, 1])
    
    for j in range(matrix.shape[1]):
        for i in range(matrix.shape[0]):
            print(i, j)
            if matrix[i, j] != matrix[i, j]:
                X_train, X_test, y_train, y_test = get_train_test_set(matrix, j)
                y_pred = get_pred(X_train, X_test, y_train)
                matrix = get_new_matrix(list(X_train), list(X_test), list(y_train), list(y_pred), j)
    
    return pd.DataFrame(data=matrix, columns=headers) 

def get_train_test_set(matrix, j):
    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(matrix.shape[0]):
        if matrix[i, j] != matrix[i, j]: # nan == nan (False)
            X_test.append(list(matrix[i, :j]) + list(matrix[i, j+1:]))
            y_test.append(matrix[i, j])
        else:
            X_train.append(list(matrix[i, :j]) + list(matrix[i, j+1:]))
            y_train.append(matrix[i, j])
            
    return X_train, X_test, y_train, y_test

def get_pred(X_train, X_test, y_train): 
    lr = LinearRegression()
    lr = lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    
    return y_pred

def get_new_matrix(X_train, X_test, y_train, y_pred, j):
    matrix = []
    for i in range(len(X_train)):
        X_train[i].insert(j, y_train[i])
        matrix.append(X_train[i])
        
    for i in range(len(X_test)):
        X_test[i].insert(j, y_pred[i])
        matrix.append(X_test[i])
        
    print('Matrix: ')
    print(matrix)
    return np.array(matrix)


# Заміна nan на значення встановлене за допомогою лінійної регресії поки що працює у випадку, коли nan є тільки в стовпчику target, а у всіх інших немає nan, так як LinearRegression() по-іншому не працює.

# ### KNN

# In[ ]:


def euclidian_dist(x_known,x_unknown):
    num_pred = x_unknown.shape[0]
    num_data = x_known.shape[0]

    dists = np.zeros((num_pred, num_data))

    for i in range(num_pred):
        for j in range(num_data):
            dists[i,j] = np.sqrt(np.sum((x_unknown[i] - x_known[j])**2))

    return dists

def k_nearest_labels(dists, y_known, k):
    num_pred = dists.shape[0]
    n_nearest = []
    
    for j in range(num_pred):
        dst = dists[j]
        closest_y = y_known[np.argsort(dst)[:k]]
        
        n_nearest.append(closest_y)
    return np.asarray(n_nearest) 


# In[ ]:


class KNearest_Neighbours(object):
    def __init__(self, k):
        
        self.k = k
        self.test_set_x = None
        self.train_set_x = None
        self.train_set_y = None
        
    def fit(self, train_set_x, train_set_y):
        self.train_set_x = train_set_x
        self.train_set_y = train_set_y
        
    def predict(self, test_set_x):
        dists = euclidian_dist(self.train_set_x, test_set_x)
        knl = k_nearest_labels(dists, self.train_set_y, self.k)
        prediction = []
        for k_nearest in knl:
            counts = np.bincount(k_nearest)
            prediction.append(np.argmax(counts))
        return prediction


# ### Standartize

# In[13]:


def standartize(df):
    headers = df.columns.values.tolist()
    matrix = copy.deepcopy(df.values)
    
    mean = matrix.mean(axis=1, keepdims=True)
    std = matrix.std(axis=1, keepdims=True)
    matrix = (matrix - mean) / std
    
    return pd.DataFrame(data=matrix, columns=headers)


# ### Scaling

# In[119]:


def scale(df, columns=None):
    headers = df.columns.values.tolist()
    matrix = copy.deepcopy(df.values)

    columns_indexes = [headers.index(column_name) for column_name in columns]
    min_columns_values = matrix.min(axis=0)
    max_columns_values = matrix.max(axis=0)

    for col_index in columns_indexes:
        diff = matrix.T[col_index] - min_columns_values[col_index]
        max_min_diff = max_columns_values[col_index] - min_columns_values[col_index]
        matrix[:, col_index] = diff / max_min_diff

    return pd.DataFrame(data=matrix, columns=headers)

