import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import random
import math

topk = 100
num_latent = 100

num_target # number of users in target dataset
num_shadow # number of users in shadow dataset
num_vector # number of items

f_target = open("file name of target interactions", 'r') # interactions for target

fr_target = open("file name of target recommendations", 'r') # recommendations for target
fr_vector = open("file name of target vectorizations", 'r') # vector for items

interaction_target = [] # interactions for target
recommend_target = []   # recommendations for target
vector_target = []  # vectors for target
vectors = []    # vectors for items
label_target = []

def dis(v1, v2):
    return math.sqrt(sum((v1-v2)**2))

def initCenter(num_k):
    center = []
    for i in range(num_k):
        index = int(random.uniform(0, num_target))
        center.append(vector_target[index])
    return center

# vectors for items
for i in range(num_vector):
    vectors.append([])
    line = fr_vector.readline()
    arr = line.split('\t')
    for j in range(100):
        arr[j] = float(arr[j])
        vectors[i].append(arr[j])
    vectors[i] = np.array(vectors[i])

# init for target
for i in range(num_target):
    recommend_target.append([])
    interaction_target.append([])
# read recommendations
line = fr_target.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    recommend_target[int(arr[0])].append(int(arr[1]))
    line = fr_target.readline()
# read interactions
line = f_target.readline()
while line != '' and line is not None and line != ' ':
    arr = line.split('\t')
    interaction_target[int(arr[0])].append(int(arr[1]))
    line = f_target.readline()

# vectorization for target
for i in range(num_target):
    if i < num_target // 2:
        # member
        label_target.append(1)
    else:
        # random
        label_target.append(0)
    temp_vector = []
    for j in range(100):
        temp_vector.append(0.0)
    temp_vector = np.array(temp_vector)
    # the center of the ineractions
    len_target = len(interaction_target[i])
    for j in range(len_target):
        temp_vector = temp_vector + vectors[interaction_target[i][j]]
    temp_vector = temp_vector / len_target
    vector_target.append(temp_vector)
    temp_vector = []
    for j in range(100):
        temp_vector.append(0.0)
    temp_vector = np.array(temp_vector)
    #the center of the recommendations
    for j in range(topk):
        temp_vector = temp_vector + (0.01)*vectors[recommend_target[i][j]]
    vector_target[i] = vector_target[i] - temp_vector

# cluster
# k-means
clusterChanged = True
centers = initCenter(2)
temp_res = []
num_k = 2
for i in range(num_target):
    temp_res.append(-1)

while clusterChanged:
    clusterChanged = False

    for i in range(num_target):
        minDist = 9999999999.0
        minIndex = 0

        for j in range(num_k):
            distance = dis(centers[j], vector_target[i])
            if distance < minDist:
                minDist = distance
                minIndex = j
        if minIndex != temp_res[i]:
            clusterChanged = True
            temp_res[i] = minIndex

    for j in range(num_k):
        temp_center = []
        temp_num = 0
        for jj in range(100):
            temp_center.append(0)
        temp_center = np.array(temp_center)
        for jj in range(num_target):
            if temp_res[jj]==j:
                temp_num = temp_num + 1
                temp_center = temp_center + vector_target[jj]
            if temp_num > 0:
                temp_center = temp_center/temp_num
                centers[j] = temp_center

num_correct = 0
TP = 0
FN = 0
TN = 0
FP = 0
for i in range(num_target):
    if label_target[i]==1:
        if temp_res[i]==1:
            num_correct += 1
            TP += 1
        else:
            FN += 1
    else:
        if temp_res[i]==0:
            num_correct += 1
            TN += 1
        else:
            FP += 1

acc_ans = num_correct/num_target

print('accuracy:')
print(acc_ans)
print('precision:')
print((TP/(TP+FP)))
print('recall:')
print((TP/(TP+FN)))
TPRate = TP / (TP+FN)
FPRate = FP / (FP+TN)
area = 0.5*TPRate*FPRate+0.5*(TPRate+1)*(1-FPRate)
print('AUC:')
print(area)
