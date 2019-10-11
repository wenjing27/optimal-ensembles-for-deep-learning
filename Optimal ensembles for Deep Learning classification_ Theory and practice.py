import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import random
import scipy
from scipy import stats
import torchvision.datasets as datasets
import torchvision.models as models
import copy
import pickle
import os

#make sure scripts are run on gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#specifications for parameters
lambda_=torch.tensor([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]).to(device)
epochs =10 
learning_rate = 0.001
ensemble_size=3

# Data Transformations
input_size=224
transform_with_aug =transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])
transform_no_aug   =transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
    ])

# Downloading/Louding CIFAR10 data
trainset  = CIFAR10(root='./data', train=True , download=True)
testset   = CIFAR10(root='./data', train=False, download=True)
classDict = {'plane':0, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

# Separating trainset/testset data/label
x_train = trainset.data
x_test = testset.data
y_train = trainset.targets
y_test = testset.targets

# Define a function to separate CIFAR classes by class index
def get_class_i(x, y, i):
    """
    x: trainset.data 
    y: trainset.targets
    i: class label, a number between 0 to 9
    return: x_i
    """
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:,0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]  
    return x_i

class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc = transform_no_aug):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths  = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc
    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)
    
    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index  = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class



# Let's choose planes (class 0 of CIFAR) and cars (class 1 of CIFAR) as trainset/testset
trainset = DatasetMaker(
        [get_class_i(x_train, y_train, classDict['plane']), get_class_i(x_train, y_train, classDict['car'])],
        transform_with_aug
    )
testset  = DatasetMaker(
        [get_class_i(x_test , y_test , classDict['plane']), get_class_i(x_test , y_test , classDict['car'])],
        transform_no_aug
    )

kwargs = {'num_workers': 0, 'pin_memory': True}

#load trainset and testset by data loader
cifarsubset_trainloader = torch.utils.data.DataLoader(trainset,shuffle=True, **kwargs, batch_size=64)
cifarsubset_testloader = torch.utils.data.DataLoader(testset,shuffle=False, **kwargs, batch_size=64)    
    
classes = ('plane','car')



#Define a function to return correlation matrix from given data
def cor(m, rowvar=False):
    '''Return a correlation matrix given data.
    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column represents 
            a single observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The correlation matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    m = m.type(torch.float) 
    fact = 1.0 / (m.size(1) - 1)
    m=m-torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    c=fact * m.matmul(mt).squeeze() #covariance matrix
    #normalize covariance matrix
    d_ = torch.diag(c)
    stddev = torch.pow(d_, 0.5)+1e-7  #Add 1e-7 to make the experiments numerically stable
    c = c.div(stddev.expand_as(c))
    c = c.div(stddev.expand_as(c).t())
    return c #correlation matrix


#Define a function to return d (i.e. averaged learner-learner correlations) from a correlation matrix
def d_func(matrix):
    Correlation=cor(matrix)
    L=Correlation.shape[1]-1
    d=(torch.sum(Correlation[1:L+1,1:L+1])-L)/2/(L*(L-1)/2)
    return d #averaged learner-learner correlations

#Define a function to return a (i.e. averaged truth-learner correlations) from a correlation matrix
def a_func(matrix):
    Correlation=cor(matrix)
    L=Correlation.shape[1]-1
    a=(torch.sum(Correlation[0,:])-1)/L
    return a #averaged truth-learner correlations



#Define our baseline pretrained net: squeezenet1_1
def Net():
    net=models.squeezenet1_1(pretrained=True)
    num_of_output_classes=2 
# change the last conv2d layer
    net.classifier._modules["1"] = nn.Conv2d(512, num_of_output_classes, kernel_size=(1, 1))
# change the internal num_classes variable rather than redefining the forward pass
    net.num_classes = num_of_output_classes
    return  net



#Model 1: single model with cross entropy loss

net=Net().to(device)
torch.backends.cudnn.benchmark=True
criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
train_loss=[]
best_acc = 0  # best test accuracy

for epoch in range(epochs):
#Training:    
    net.train()
    for batch_idx, (inputs, targets) in enumerate(cifarsubset_trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss.append(loss)

#Testing:
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(cifarsubset_testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(net, './checkpoint/ckpt.pth')
        best_acc = acc
single_test_accuracy=best_acc   #accuracy of model 1  

#save the model for future use
saved_model = torch.load('./checkpoint/ckpt.pth')


#save results from Model 1
data1 = [single_test_accuracy, device,epochs]

with open('result1.pkl','wb') as outfile:
    pickle.dump(data1, outfile)

# with open('result1.pkl','rb') as infile:
#     result1 = pickle.load(infile)




# Model 2: Ensemble of Model with Entropy Loss

#load saved model
def Net2():
    net=copy.deepcopy(saved_model)
    return net

torch.backends.cudnn.benchmark=True
model_list=[]
for _ in range(ensemble_size):
    model_list.append(Net2().to(device))
params=[]
for model in model_list:
    params.extend(list(model.parameters()))
optimizer = torch.optim.Adam(params, lr=learning_rate) 
criterion=nn.CrossEntropyLoss()
train_loss=[]
best_acc=0

for epoch in range(epochs):
#training:    
    for j in range(ensemble_size):
        model_list[j].train()
    for batch_iter, data in enumerate(cifarsubset_trainloader):
    # get the inputs
        inputs_train, labels_train = data
        inputs_train, labels_train = inputs_train.to(device), labels_train.to(device)
        optimizer.zero_grad()
        outputs_train=[[] for i in range(ensemble_size)]
        for j in range(ensemble_size):
            outputs_train[j] = model_list[j](inputs_train)
        outputs_train_ave=np.sum(outputs_train)/ensemble_size  #take the averaged scores as the score for ensemble 
        loss_train=criterion(outputs_train_ave, labels_train)
        train_loss.append(loss_train.item())
        loss_train.backward()
        optimizer.step()

    #testing:
    
    total=0
    correct=0
    for j in range(ensemble_size):
        model_list[j].eval()
    with torch.no_grad():
        for batch_iter, data in enumerate(cifarsubset_testloader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs_test=[[] for i in range(ensemble_size)]
            predicted=[[] for i in range(ensemble_size)]
            for j in range(ensemble_size):
                outputs_test[j] = model_list[j](images)   
                predicted[j] = torch.max(outputs_test[j].data, 1)[1].cpu().numpy()
            total += labels.size(0)
            correct += (scipy.stats.mode(predicted)[0] == labels.cpu().numpy()).sum().item()
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
 
            
ensemble_cross_entropy_test_accuracy=best_acc  #accuracy of model 2

#save results of model 2
data2 = [ensemble_cross_entropy_test_accuracy, device, epochs]

with open('result2.pkl','wb') as outfile:
    pickle.dump(data2, outfile)   


    
    
    

# Model 3: Loss = truth-learner correlation + lambda_*learner-learner correlation

#load saved model
def Net3():
    net=copy.deepcopy(saved_model)
    return net

torch.backends.cudnn.benchmark=True
model_list=[]
for _ in range(ensemble_size):
    model_list.append(Net3().to(device))
params=[]
for model in model_list:
    params.extend(list(model.parameters()))
optimizer = torch.optim.Adam(params, lr=learning_rate) 
best_acc=0
loss_train=[]
train_ave_truth_learner_corr=[]
train_ave_learner_learner_corr=[]
train_acc_all=[]
for epoch in range(epochs):
#training:    
    for j in range(ensemble_size):
        model_list[j].train()
    for batch_iter, data in enumerate(cifarsubset_trainloader):#, 0):
    # get the inputs
        inputs_train, y_train = data
        labels_train=torch.tensor(y_train, dtype=torch.float)
        inputs_train,labels_train=inputs_train.to(device),labels_train.to(device)
        y_train=y_train.to(device)
        optimizer.zero_grad()
        matrix_train=torch.zeros([len(y_train),ensemble_size+1], dtype=torch.float) #matrix with truth and all the outputs from three nets 
        matrix_train=matrix_train.to(device)
        matrix_train[:,0]=torch.cuda.FloatTensor(torch.squeeze(labels_train))
        matrix_hard=torch.zeros([len(y_train),ensemble_size], dtype=torch.float)
        matrix_hard=matrix_hard.to(device)
        outputs_train=[[] for k in range(ensemble_size)]
        for j in range(ensemble_size):
            outputs_train[j]=model_list[j](inputs_train)
            matrix_train[:,j+1]=torch.cuda.FloatTensor(torch.squeeze(outputs_train[j][:,1]))
            matrix_hard[:,j]=torch.max(outputs_train[j],1)[1]# 0,1 label
        majority_vote_label=scipy.stats.mode(matrix_hard.detach().cpu().numpy().T)[0]
        train_accuracy=np.sum(majority_vote_label.ravel() == labels_train.cpu().numpy())/len(labels_train) #majority vote accuracy
        a_train=a_func(matrix_train) #averaged truth-learner correlations
        d_train=d_func(matrix_train) #averaged learner-learner correlations
        train_ave_truth_learner_corr.append(a_train.item())
        train_ave_learner_learner_corr.append(d_train.item())
        train_acc_all.append(train_accuracy.item())
        loss=-a_train+lambda_*d_train  #our novel loss function
        loss_train.append(loss.item())
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
    #testing:
    total=0
    correct=0
    for j in range(ensemble_size):
        model_list[j].eval()
    with torch.no_grad():
        for batch_iter, test_data in enumerate(cifarsubset_testloader):
            inputs_test, y_test = test_data
            inputs_test, y_test = inputs_test.to(device), y_test.to(device)
            matrix_hard_test=torch.zeros([len(y_test),ensemble_size], dtype=torch.float)
            for j in range(ensemble_size):
                output_test=model_list[j](inputs_test)
                matrix_hard_test[:,j]=torch.max(output_test,1)[1]# 0,1 label
            majority_vote_label_test=scipy.stats.mode(matrix_hard_test.cpu().numpy().T)[0]
            total += y_test.size(0)
            correct += (majority_vote_label_test == y_test.cpu().numpy()).sum().item()
            torch.cuda.empty_cache()
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        
majority_vote_accu_test=best_acc  #accuracy of model 3

#save the results from model 3
data3 = ['majority_vote_accu_test, device,epochs,lambda_,train_ave_truth_learner_corr,train_ave_learner_learner_corr,train_acc_all',
         majority_vote_accu_test, device,epochs,lambda_,train_ave_truth_learner_corr,train_ave_learner_learner_corr,train_acc_all]

with open('result3.pkl','wb') as outfile:
    pickle.dump(data3, outfile)

# with open('result3.pkl','rb') as infile:
#     result3 = pickle.load(infile)  

