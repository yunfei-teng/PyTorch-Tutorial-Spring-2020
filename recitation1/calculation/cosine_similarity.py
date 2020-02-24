# PyTorch tutorial codes for course Advanced Machine Learning
import datetime
import torch
import torch.nn.functional as F

print('\nThis experiemnt shows the efficiency of different algorithms calculating consine similarity of columns of a square matrix:')
print('Algorithm 1: Using forloop \nAlgorithm 2: Using Matrix on CPU \nAlgorithm 3: Using Matrix on GPU')

########## Experiment 1 ##########
size = 1000
vectors = torch.randn(size, size)
vectors.cuda() # This step does nothing but check if GPU can work
print('\nMatrix of type', size)

# Algorithm 1: Using forloop
start_time = datetime.datetime.now()
result1 = torch.zeros(size, size)
for i in range(vectors.size(0)):
    for j in range(i, vectors.size(1)):
        result1[i][j] = F.cosine_similarity(vectors[:, i], vectors[:, j], dim = 0, eps=1e-12).item()
        result1[j][i] = result1[i][j]

current_time = datetime.datetime.now()
print('Algorithm 1 Time Interval:', current_time - start_time)

# Algorithm 2: Using Matrix on CPU
start_time = datetime.datetime.now()
_norm  = torch.norm(vectors, 2, 0)
_vectors = vectors / _norm
result2 = torch.mm(_vectors.t(), _vectors) 
current_time = datetime.datetime.now()
print('Algorithm 2 Time Interval:', current_time - start_time)

# Algorithm 3: Using Matrix on GPU
start_time = datetime.datetime.now()
vectors = vectors.cuda()
_norm  = torch.norm(vectors, 2, 0)
_vectors = vectors / _norm
result3 = torch.mm(_vectors.t(), _vectors) 
vectors = vectors.cpu()
result3 = result3.cpu()
current_time = datetime.datetime.now()
print('Algorithm 3 Time Interval:', current_time - start_time)

# You can check that they have the same results. However, due to the different precision of 
# CPU Tensors and GPU Tensors, the results may vary a little bit.
# print((result1 - result2).abs().max())
# print((result1 - result3).abs().max())

########## Experiment 2 ##########
size = 10000
vectors = torch.randn(size, size)
vectors.cuda() # This step does nothing but check if GPU can work
print('\nMatrix of type', size)

# Algorithm 2: Using Matrix on CPU
start_time = datetime.datetime.now()
_norm  = torch.norm(vectors, 2, 0)
_vectors = vectors / _norm
result4 = torch.mm(_vectors.t(), _vectors) 
current_time = datetime.datetime.now()
print('Algorithm 2 Time Interval:', current_time - start_time)

# Algorithm 3: Using Matrix on GPU
start_time = datetime.datetime.now()
vectors = vectors.cuda()
_norm  = torch.norm(vectors, 2, 0)
_vectors = vectors / _norm
result5 = torch.mm(_vectors.t(), _vectors)
vectors = vectors.cpu()
result5 = result5.cpu()
current_time = datetime.datetime.now()
print('Algorithm 3 Time Interval:', current_time - start_time)
