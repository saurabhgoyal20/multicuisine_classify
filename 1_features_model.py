#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:11:51 2018

@author: saurabh
"""

"""ReadMe

This is a Kaggle problem, which involves classifying a list of ingredients into one of given 20 cuisine(Italian, Indian, French, Chinese etc.).
For Eg: A dish with Ingredients as "Penne Pasta", "chopped tomatoes", "fresh basil", "garlic", "extra-virgin olive oil","kosher salt","flat leaf parsley",
i.e., (Pasta) will be classified under cuisine 'Italian'

Every dish comes with a list of ingredients where number of ingredients is not fixed.
To build a predictive based model, these ingredients need to be converted into informative numbers

To quantify ingredients, I find how important and particular is an ingredient to a cuisine, for instance, "Penne Pasta" will be particular to 'Italian', 
however, ingredients like "Salt", "Sugar", "Water" will be important to almost every cuisine but will be common to every cuisine.

Ingredients importance(to a cuisine) = No. of dishes in a cuisine containing that ingredients/(No of dishes in all cuisines containing that ingredients * No. of dishes in that cuisine in training data)

Building Features of a dish:  Importance(to a cuisine) of all Ingredients present in that dish are added up.
This is being done for all 20 cuisines.

Thus the feature vector has a length of 20 where every dimension in a way suggests that  how important is that set of ingredients to a cuisine 

I then use Support Vector Machine to built a predictive model


"""


from sklearn import svm
import pickle
import gensim
from gensim.models import Word2Vec
import numpy as np
import json
import collections
import csv
import time
import warnings
import statistics
warnings.filterwarnings("ignore", category=DeprecationWarning)

start = time.clock()


#Loading data
with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all/train.json','r', encoding='utf-8') as f:
    json_dict = json.load(f)

id_list = []             #List of dish IDs
cuisine = []             #List of IDs Labels(cuisine)
ingredients_data = []    #List of ingredients of a dish in training data 
ingredients_all = []     #List of all ingredients



for data in json_dict:
    id_list.append(data['id'])
    cuisine.append(data['cuisine'])
    ingredients_data.append(data['ingredients'])
    ingredients_all.extend(data['ingredients'])

#groups_cuisine contains number of dishes per cuisine in training data
groups_cuisine =dict(collections.Counter(cuisine))

#Converting groups_cuisine to a list
cuisine_strength = []
for key, value in groups_cuisine.items():
    k = [key, value]
    cuisine_strength.append(k)

##cuisine_all is used to build a list cuisine_all_prop, which contains ingredients and their frequency for a cuisine
cuisine_all = []    
for i in range(len(cuisine_strength)):
    k = [cuisine_strength[i][0], []]
    cuisine_all.append(k)


for data in json_dict:
    k = data['cuisine']
    
    for j in range(len(cuisine_all)):
        if k == cuisine_all[j][0]:
            cuisine_all[j][1].extend(data['ingredients'])

cuisine_all_prop = cuisine_all


for i in range(len(cuisine_all)):
    cuisine_all_prop[i][0] = cuisine_all[i][0]
    cuisine_all_prop[i][1] = dict(collections.Counter(cuisine_all[i][1]))
    
    
#groups_ingredients contains the frequency of an ingredient in all cuisines
groups_ingredients = dict(collections.Counter(ingredients_all))
ingredients_list =[]
ingredients_sum  =[]



for key, value in groups_ingredients.items():
    ingredients_list.append(key)
    ingredients_sum.append(value)

#ingredients_content contains ingredient importance
ingredients_contents = []
for i in range(len(ingredients_list)):
    k = [(ingredients_list[i]).encode('utf-8')]
    A = [0] * len(cuisine_strength)
    k.extend(A)
    ingredients_contents.append(k)


for i in range(0, len(ingredients_contents)):
    ingr =  ingredients_list[i]
    
    for key, value  in groups_ingredients.items():
        
        den1 = 1
        if key == ingr:
            den1 = value
            break

    for j in range(1, (len(cuisine_all_prop)+ 1)):
        n1 = 0
        n2 = 0
        cuis = cuisine_all_prop[j-1][0]            
        den2 = cuisine_strength[j-1][1]
            
        for key, value in cuisine_all_prop[j-1][1].items():
            if key == ingr:
                n2 = value
            
        num = max(n1, n2) 
        no = (num/(den1*den2))*40000
        ingredients_contents[i][j] = no
                

        
header = ['ingredient']

for i in range(len(cuisine_strength)):
    ingr = cuisine_strength[i][0].encode('utf-8')
    header.append(ingr)
    
"""To make common ingredients("Salt", "Sugar") less important and unique ingredients("Penne Pasta") more importance, coefficient of variation(Standard deviation/Average) is computed for every ingredient 
Which it is multiplied with ingredient importance -> ingredients_contents_2"""

ingr_variance = []

for i in range(0, len(ingredients_contents)):
    
    std = statistics.stdev(map(float, ingredients_contents[i][1:len(ingredients_contents[i])]))
    avg = statistics.mean(map(float, ingredients_contents[i][1:len(ingredients_contents[i])]))
    avg = 1
    coeff_varn = std/avg
    k =[std, avg, coeff_varn]
    ingr_variance.append(k)
    


ingredients_contents_2 = []

for i in range(0, len(ingredients_contents)):
    k = [ingredients_contents[i][0]]
    for j in range(1, len(ingredients_contents[i])):
        var = ingredients_contents[i][j]*ingr_variance[i][2]
        k.append(var)
        
    ingredients_contents_2.append(k)
    
#Training Data formation

X = []
Y = []

for i in range(0, len(ingredients_data)):

    initial_list = [0]*len(cuisine_strength)
    
    
    for j in range(len(ingredients_data[i])):
        ingr= ingredients_data[i][j].encode('utf-8')

        for k in range(len(ingredients_contents_2)):
           
            if ingredients_contents_2[k][0] == ingr:
                initial_list = [(x + y) for x, y in zip(initial_list, ingredients_contents_2[k][1: len(ingredients_contents_2[k])])]
                
        
    no_ingr = len(ingredients_data[i])
    initial_list.append(no_ingr)
    X.append(initial_list)
    
    Y.append(cuisine[i])
    

print('Time Taken in data formation' , (time.clock() - start)/60, 'mins')

print('Data length: train data length,  features, outcome length', len(X), len(X[0]), len(Y))

#Running predictive model
clf_SVClinear = svm.SVC(C = 1, kernel = 'linear', decision_function_shape = 'ovo')
clf_SVClinear.fit(X,Y)
clf_SVClinear = pickle.dumps(clf_SVClinear)

print('Modelformed')

clf = pickle.loads(clf_SVClinear)


accuracy =0
matrix = []

#Computing training data accuracy
for i in range(0, len(X)):
    k = clf.predict(X[i])
    
        
    if k == cuisine[i]:
        accuracy = accuracy + 1/len(cuisine)
        mismatch =0
    else:
        mismatch = 1
        
        
    A =[]
    A.append(id_list[i])
    A.append(cuisine[i])
    A.append(str(k).encode(encoding = 'utf-8'))
    A.append(mismatch)
    A.extend(X[i])
    matrix.append(A) 
    
print('Accuracy of SVC linear ovo model is ', accuracy)

header = ['id', 'cuisine', 'predicted', 'mismatch']

for i in range(0, len(cuisine_strength)):
    header.append(cuisine_strength[i][0])

header.append('no_ingr')
    

with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/LogisticRegression/train_scoring_ub20.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(header)
    writer.writerows(matrix)
 
print('Time Taken in model formation' , (time.clock() - start)/60, 'mins')


#Scoring
with open("/Users/minnie/Desktop/Kaggle/Cooking/Data/all/test.json","r", encoding="utf-8") as f:
    json_dict = json.load(f)
    

    
testdata_list = []
id_list=[]

for data in json_dict:
    id_list.append(data['id'])
    testdata_list.append(data['ingredients'])
    
test_vector = []

t1 = 0
t2 = 0
t3 = 0
z = 0

for i in range(0, len(testdata_list)):

    initial_list = [0]*len(cuisine_strength)
    t1 = t1+1
    
    for j in range(len(testdata_list[i])):
        ingr= testdata_list[i][j].encode('utf-8')
        t2 = t2 + 1
        
        for k in range(len(ingredients_contents_2)):
            t3 = t3+1
           
            if ingredients_contents_2[k][0] == ingr:
                z = z+1
                initial_list = [(x + y) for x, y in zip(initial_list, ingredients_contents_2[k][1: len(ingredients_contents_2[k])])]
                
        
    no_ingr = len(testdata_list[i])
    initial_list.append(no_ingr)
    test_vector.append(initial_list)

print('test data formed', t1, t2, t3, z)    

print('length of testdata is', len(test_vector), len(test_vector[0]))

clf = pickle.loads(clf_SVClinear)
    
testcuisine = []
mismatch_matrix = []

for i in range(0, len(test_vector)):
    
    k = clf.predict(test_vector[i])
    result = [id_list[i], k[0]]

    testcuisine.append(result)
    
print('Scoring done')

with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/LogisticRegression/LR_test_20_1.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['id', 'cuisine'])
    writer.writerows(testcuisine)
    
with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/LogisticRegression/LR_test_20_1.csv', 'r') as f:
    reader = csv.reader(f)
    your_list = list(reader)

print('csv done')

with open('/Users/minnie/Desktop/Kaggle/Cooking/Data/all2/kaggle_submission.csv', 'r') as f:
    reader = csv.reader(f)
    list_kaggle = list(reader)
    
for i in range(0,len(list_kaggle)):
    for j in range(0,len(your_list)):
        if list_kaggle[i][0] == your_list[j][0]:
            list_kaggle[i][1] = your_list[j][1]
            break


with open('/Users/minnie/Desktop/Kaggle/Cooking/Coding/LogisticRegression/LR_test_20_1.csv', 'w',newline='') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
    writer.writerows(list_kaggle)

print('Time Taken by process is' , (time.clock() - start)/60, 'mins')

