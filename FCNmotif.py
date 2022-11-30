# -*- coding: utf8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import sys


def upsample(x, out_size):
    return F.interpolate(x, size=out_size, mode='linear', align_corners=False)


def bn_relu_conv(in_, out_, kernel_size=3, stride=1, bias=False):
    padding = kernel_size // 2
    return nn.Sequential(nn.BatchNorm1d(in_),
                         nn.ReLU(inplace=True),
                         nn.Conv1d(in_, out_, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))


dev = qml.device("default.qubit", wires=16)

@qml.qnode(dev)
def circuit_expval(theta):
    qml.RX(theta, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.expval(qml.PauliY(0))

@qml.qnode(dev)
def circuit_var(theta):
    qml.RX(theta, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    return qml.var(qml.PauliY(0))

@qml.qnode(dev)
def circuit(phi):
    N_phi = len(phi)

    # Encoding of 'N_phi' classical input values
    for j in range(N_phi):
        qml.RY(np.pi * phi[j], wires=j)

    # Random quantum circuit
    N_layers = 1
    N_rand_params = 20
    rand_params = np.random.uniform(high=2 * np.pi, size=(N_layers, N_rand_params))
    RandomLayers(rand_params, wires=range(N_phi))

    # Measurement producing 'N_phi' classical output values
    return [qml.expval(qml.PauliZ(j)) for j in range(N_phi)]


def qonv(data): # batch: 1 x N_base x batch_size-kernel_size+1
    # Constants
    N = len(data[0][0])
    K = 16
    slide = 1
    N_base = 4

    # Output Size
    out_x = 1
    out_y = N_base * K
    out_c = N-K+1
    out = np.zeros((out_x, out_y, out_c))

    for y in range(N_base):
        batch_y = data[0][y]

    out_tmp = np.zeros((out_x, K, out_c))
    for c in range(0, out_c, slide):
        batch_k = batch_y[c:c+K]
        q_results = circuit(batch_k) # same size as the batch (K)

        for i in range(K):
            out_tmp[0][i][c]= q_results[i]

    for i in range(K):
        for j in range(out_c):
            out[0][i+y*K][j] = out_tmp[0][i][j]

    return out



class FCNwithQ(nn.Module):
    """FCN for motif mining"""
    def __init__(self, motiflen=13):
        super(FCNwithQ, self).__init__()
        # encode process
        # self.qonv = Qonv(kernel_size=16, slide=1, N_base=4)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # decode process
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        print("data : ", data.shape)
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = qonv(data)
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        # decode process
        up3 = upsample(skip4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        print("out_dense : ", out_dense.shape)

        return out_dense


class FCN(nn.Module):
    """FCN for motif mining"""
    def __init__(self, motiflen=13):
        super(FCN, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # decode process
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        print("data : ", data.shape)
        # encode process
        skip1 = data
        out1 = self.conv1(data)

        print("out1 : ", out1.shape)
        
        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        # decode process
        up3 = upsample(skip4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        return out_dense


class FCNA(nn.Module):
    """FCNA for motif mining"""
    def __init__(self, motiflen=13):
        super(FCNA, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        # decode process
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        print("data : ", data.shape)
        # encode process
        skip1 = data
        out1 = self.conv1(data)

        print("out1 : ", out1.shape)

        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        up5 = self.aap(out1)
        # decode process
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        print("out_dense : ", out_dense.shape)

        return out_dense


class FCNAR(nn.Module):
    """A slightly-modified FCNA for refining motif mining"""
    def __init__(self, motiflen=13):
        super(FCNAR, self).__init__()
        # encode process
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=64, kernel_size=motiflen, padding=motiflen//2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.aap = nn.AdaptiveAvgPool1d(1)
        # decode process
        self.blend4 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend3 = bn_relu_conv(64, 64, kernel_size=3)
        self.blend2 = bn_relu_conv(64, 4, kernel_size=3)
        self.blend1 = bn_relu_conv(4, 1, kernel_size=3)
        # general functions
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self._init_weights()

    def _init_weights(self):
        """Initialize the new built layers"""
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)

    def forward(self, data):
        """Construct a new computation graph at each froward"""
        b, _, _ = data.size()
        # encode process
        skip1 = data
        out1 = self.conv1(data)

        out1 = self.relu(out1)
        out1 = self.pool1(out1)
        out1 = self.dropout(out1)
        skip2 = out1
        out1 = self.conv2(out1)
        out1 = self.relu(out1)
        out1 = self.pool2(out1)
        out1 = self.dropout(out1)
        skip3 = out1
        out1 = self.conv3(out1)
        out1 = self.relu(out1)
        out1 = self.pool3(out1)
        out1 = self.dropout(out1)
        skip4 = out1
        up5 = self.aap(out1)
        # decode process
        up4 = upsample(up5, skip4.size()[-1])
        up4 = up4 + skip4
        up4 = self.blend4(up4)
        up3 = upsample(up4, skip3.size()[-1])
        up3 = up3 + skip3
        up3 = self.blend3(up3)
        up2 = upsample(up3, skip2.size()[-1])
        up2 = up2 + skip2
        up2 = self.blend2(up2)
        up1 = upsample(up2, skip1.size()[-1])
        up1 = up1 + skip1
        up1 = self.blend1(up1)
        out_dense = self.sigmoid(up1)
        out_dense = out_dense.view(b, -1)

        return out_dense
