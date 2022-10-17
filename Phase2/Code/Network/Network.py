"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 2 Starter Code


Colab file can be found at:
    https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute


Code adapted from CMSC733 at the University of Maryland, College Park.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def confusion_array(outputs,labels):

    _, preds = torch.max(outputs, dim=1)

    return confusion_matrix(preds,labels)

def loss_fn(out, labels):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################


    criterion = nn.CrossEntropyLoss()

    # print(out)
    # print(labels)

    loss=criterion(out,labels)

    return loss 

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = loss_fn(out, labels) # Calculate loss
        acc= accuracy(out,labels)
        # conf_matrix=confusion_array(out,labels)
        return loss,acc
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = loss_fn(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'loss': loss.detach(), 'acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'loss': epoch_loss.item(), 'acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], loss: {:.4f}, acc: {:.4f}".format(epoch, result['loss'], result['acc']))






class CIFAR10Model(ImageClassificationBase):

    def __init__(self,InputSize,OutputSize):

        
#       """
#       Inputs: 
#       InputSize - Size of the Input
#       OutputSize - Size of the Output
#       """
#       #############################
#       # Fill your network initialization of choice here!
#       #############################

        # Code understood with shapes and parameters
        #  and changed from Pytorch official tutorials : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

        super().__init__()
        self.conv1 = nn.Conv2d(3, 9, 5)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(9, 18, 5)
        self.BN1=nn.BatchNorm2d(9)
        self.BN2=nn.BatchNorm2d(18)
        self.fc1 = nn.Linear(5 * 5 * 18, 100)
        self.fc2=nn.Linear(100,OutputSize)
        

    

          
    def forward(self, xb):

        """
      Input:
      xb is a MiniBatch of the current image
      Outputs:
      out - output of the network
      """
      #############################
      # Fill your network structure of choice here!
      #############################

        # Code adopted from Pytorch official tutorials : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

        x = self.pool(self.BN1(F.relu(self.conv1(xb))))
        x = self.pool(self.BN2(F.relu(self.conv2(x))))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



      

    



