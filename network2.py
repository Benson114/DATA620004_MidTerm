#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:23:39 2024

@author: zzz
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import json
from sklearn.model_selection import ParameterGrid
import random

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='tanh'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        
        # 激活函数及其导数
        if self.activation == 'tanh':
            self.activation_func = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x)**2
        elif self.activation == 'sigmoid':
            self.activation_func = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: self.activation_func(x) * (1 - self.activation_func(x))
        else:
            raise ValueError("Unsupported activation function.")
        
    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation_func(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, X, y, learning_rate, reg_lambda=0.01):
        # 反向传播
        batch_size = X.shape[0]
        delta3 = self.probs
        delta3[range(batch_size), y] -= 1
        delta3 /= batch_size
        
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = np.dot(delta3, self.W2.T) * self.activation_derivative(self.z1)
        
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # 添加L2正则化项的梯度
        dW2 += reg_lambda * self.W2
        dW1 += reg_lambda * self.W1
        
        # 参数更新
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
    def predict(self, X):
        # 预测
        return np.argmax(self.forward(X), axis=1)
    
    def save_weights(self, file_path):
        # 保存模型权重
        model_params = {
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
            'activation': self.activation
        }
        with open(file_path, 'w') as f:
            json.dump(model_params, f)
            
    def load_weights(self, file_path):
        # 加载模型权重
        with open(file_path, 'r') as f:
            model_params = json.load(f)
        self.W1 = np.array(model_params['W1'])
        self.b1 = np.array(model_params['b1'])
        self.W2 = np.array(model_params['W2'])
        self.b2 = np.array(model_params['b2'])
        self.activation = model_params['activation']
        if self.activation == 'tanh':
            self.activation_func = np.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x)**2
        elif self.activation == 'sigmoid':
            self.activation_func = lambda x: 1 / (1 + np.exp(-x))
            self.activation_derivative = lambda x: self.activation_func(x) * (1 - self.activation_func(x))

def train_model(X_train, y_train, X_val, y_val, params):
    input_size = X_train.shape[1]
    hidden_size = params['hidden_size']
    output_size = len(np.unique(y_train))
    activation = params['activation']
    learning_rate = params['learning_rate']
    reg_lambda = params['reg_lambda']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    
    model = NeuralNetwork(input_size, hidden_size, output_size, activation)
    
    best_val_acc = 0.0
    best_model = None
    
    for epoch in range(num_epochs):
        # Shuffle training data
        shuffle_index = np.random.permutation(len(X_train))
        X_train_shuffled = X_train.iloc[shuffle_index]
        y_train_shuffled = y_train.iloc[shuffle_index]
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Forward and backward pass
            model.forward(X_batch)
            model.backward(X_batch, y_batch, learning_rate, reg_lambda)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        
        # Save the best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model
    
    return best_model, best_val_acc

# 加载MNIST数据集
mnist = fetch_openml('mnist_784')
X, y = mnist.data.astype('float32'), mnist.target.astype('int')

# 数据归一化
X /= 255.0

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# 定义参数搜索空间
param_grid = {
    'hidden_size': [64, 128],
    'activation': ['tanh', 'sigmoid'],
    'learning_rate': [0.01, 0.1],
    'reg_lambda': [0.001, 0.01],
    'num_epochs': [10],
    'batch_size': [64, 128]
}

# 参数搜索和模型训练
best_model = None
best_val_acc = 0.0

evaluation = []

for params in ParameterGrid(param_grid):
    model, val_acc = train_model(X_train, y_train, X_val, y_val, params)
    print(f"Params: {params}, Validation Accuracy: {val_acc}")
    evaluation.append({"params":params,"accuracy":val_acc})
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model = model

# 保存最优模型
best_model.save_weights('best_model.json')

# 加载并测试最优模型
loaded_model = NeuralNetwork(input_size=X_test.shape[1], hidden_size=best_model.hidden_size,
                             output_size=len(np.unique(y_train)), activation=best_model.activation)
loaded_model.load_weights('best_model.json')

y_pred_test = loaded_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy}")
