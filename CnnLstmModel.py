#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:02:56 2020

@author: nami
"""

import torch.nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from pprint import pprint
#from efficientnet_pytorch import EfficientNet

class CnnLstmModel(torch.nn.Module):
    def __init__(self, hiddenSize, numLayers, outputSize, bidirectional, device):
        super(CnnLstmModel, self).__init__()
        # Hidden dimensions
        self.hidden_size = hiddenSize
        # Number of hidden layers
        self.num_layers = numLayers
        self.bidirectional = bidirectional
        self.device = device

        # Building the CNN + LSTM
        #Initialize CNN:   
            
        # für Resnet18/34/50
        '''
        self.cnn = models.resnet18(pretrained=True) 
        self.inFeatures = self.cnn.fc.in_features * 49 
        self.cnn.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7)) 
        self.cnn.fc = torch.nn.Identity()
        '''
        # für EfficientNet
        '''
        self.cnn = models.efficientnet_b1(pretrained=True)
        self.inFeatures = self.cnn.classifier[1].in_features * 16
        self.cnn.avgpool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.cnn.classifier = torch.nn.Identity()
        '''
        # für EfficientNet V2
        
        #pprint(timm.list_models(pretrained=True))
        self.cnn = timm.create_model('efficientnetv2_rw_s', pretrained=True)    
        #self.cnn = timm.create_model('tf_efficientnetv2_m_in21k', pretrained=True)  
        self.inFeatures = self.cnn.classifier.in_features * 16
        self.cnn.global_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.cnn.classifier = torch.nn.Identity()
        
        # für RegNet
        '''
        self.cnn = models.regnet_y_16gf(pretrained=True)
        self.inFeatures = self.cnn.fc.in_features * 9 
        self.cnn.avgpool = torch.nn.AdaptiveAvgPool2d((3, 3)) 
        self.cnn.fc = torch.nn.Identity()
        '''

        
        #self.cnn.fc = torch.nn.Linear(self.cnn.fc.in_features, int(self.cnn.fc.in_features))
            
        
        self.layerNorm = torch.nn.LayerNorm(self.inFeatures)
        
        ##self.lstm = torch.nn.LSTM(input_size=self.inFeatures, hidden_size=hiddenSize, num_layers=numLayers, bidirectional=self.bidirectional, dropout=0.5, batch_first=True)
        self.gru = torch.nn.GRU(input_size=self.inFeatures, hidden_size=hiddenSize, num_layers=numLayers, bidirectional=self.bidirectional, dropout=0.5, batch_first=True)
        
        # Readout layer
        if bidirectional:
            self.fc = torch.nn.Linear(2 * hiddenSize, outputSize)
        else:
            self.fc = torch.nn.Linear(hiddenSize, outputSize)

        
    def forward(self, image_sequence):
        length = image_sequence.shape[1]
        if length % 3 == 2 :
                image1 = image_sequence[:, -1, :, :].view(-1, 1, 512, 512)
                image_sequence = torch.cat((image_sequence, image1), dim=1)
                length += 1
                
        elif length % 3 == 1 :
                image1 = image_sequence[:, -1, :, :].view(-1, 1, 512, 512)
                image_sequence = torch.cat((image_sequence, image1, image1), dim=1)
                length += 2

        image = image_sequence.view(int(length/3), 3, 512, 512)
        output_cnn = self.cnn(image.to(device=self.device, dtype=torch.float32))
        output_cnn = torch.flatten(output_cnn, start_dim=1)
        output_cnn = F.leaky_relu(self.layerNorm(output_cnn))
        out, (hn) = self.gru(output_cnn.view(-1, int(length/3), self.inFeatures))
        return self.fc(out[:, -1, :])
        

    