#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:17:55 2024

@author: zzz
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# 加载MNIST数据集
mnist = fetch_openml('mnist_784')
X, y = mnist.data.astype('float32'), mnist.target.astype('int')

# 数据归一化
X /= 255.0

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对标签进行独热编码
encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def forward(self, X):
        # 前向传播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        exp_scores = np.exp(self.z2)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return self.probs
    
    def backward(self, X, y, learning_rate=0.01):
        # 反向传播
        batch_size = X.shape[0]
        delta3 = self.probs
        delta3[range(batch_size), y] -= 1
        delta3 /= batch_size
        
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        
        delta2 = np.dot(delta3, self.W2.T) * (1 - np.power(self.a1, 2))
        
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # 参数更新
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        
    def predict(self, X):
        # 预测
        return np.argmax(self.forward(X), axis=1)

    def train(self, X, y, batch_size=64, num_epochs=1000, learning_rate=0.01):
        for epoch in range(num_epochs):
            # 按照 batch_size 遍历训练集
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # 前向传播
                probs = self.forward(X_batch)
                
                # 计算损失（交叉熵损失）
                loss = -np.sum(np.log(probs[range(len(X_batch)), y_batch])) / len(X_batch)
                
                # 反向传播
                self.backward(X_batch, y_batch, learning_rate)
                
            # 每10次迭代输出一次损失
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# 创建神经网络模型
input_size = X_train.shape[1]
hidden_size = 128
output_size = 10
model = NeuralNetwork(input_size, hidden_size, output_size)

# 设置训练参数并训练模型
batch_size = 64
num_epochs = 10
learning_rate = 0.1
model.train(X_train, y_train, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate)

# 使用测试集评估模型
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Accuracy on test set: {accuracy}')

