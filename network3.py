import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import ParameterGrid
import json
from tqdm import tqdm

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
    
    train_acc_history = []
    val_acc_history = []
    
    for epoch in tqdm(range(num_epochs)):
        # Shuffle training data
        shuffle_index = np.random.permutation(len(X_train))
        X_train_shuffled = X_train.iloc[shuffle_index]
        y_train_shuffled = y_train.iloc[shuffle_index]
        
        # Mini-batch training
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_shuffled.iloc[i:i+batch_size].values
            y_batch = y_train_shuffled.iloc[i:i+batch_size].values
            
            # Forward and backward pass
            model.forward(X_batch)
            model.backward(X_batch, y_batch, learning_rate, reg_lambda)
        
        # Evaluate on training set
        y_train_pred = model.predict(X_train.values)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_acc_history.append(train_acc)
        
        # Evaluate on validation set
        y_val_pred = model.predict(X_val.values)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_acc_history.append(val_acc)
        file_path = f"model/d_{params['hidden_size']}_lr{params['learning_rate']}_reg{params['reg_lambda']}_fuc_{params['activation']}_batch_{params['batch_size']}_ep_{epoch}"
        model.save_weights(file_path)
        
        
        # Print and save accuracy for each epoch
        print(f"Epoch {epoch + 1}/{num_epochs}: Train Accuracy = {train_acc:.4f}, Val Accuracy = {val_acc:.4f}")
    
    return train_acc_history, val_acc_history

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
    'hidden_size': [128],
    'activation': ['tanh'],
    # 'learning_rate': [1e-5],
    'learning_rate': [1e-5,1e-4,1e-3,1e-2,1e-1],
    'reg_lambda': [0.001],
    'num_epochs': [10],
    'batch_size': [128]
}

# 记录准确率的字典
accuracy_dict = {}

for params in ParameterGrid(param_grid):
    train_acc_history, val_acc_history = train_model(X_train, y_train, X_val, y_val, params)
    print(f"Params: {params}, Final Training Accuracy: {train_acc_history[-1]}, Final Validation Accuracy: {val_acc_history[-1]}")
    
    # 将准确率记录到字典中
    accuracy_dict[json.dumps(params)] = {
        'train_accuracy': train_acc_history,
        'val_accuracy': val_acc_history
    }

# 打印所有参数组合下的准确率
for params, acc_history in accuracy_dict.items():
    print(f"Params: {params}")
    print(f"Final Training Accuracy: {acc_history['train_accuracy'][-1]}")
    print(f"Final Validation Accuracy: {acc_history['val_accuracy'][-1]}")
    print()
    
with open("result.json","w") as f:
    json.dump(accuracy_dict, f, ensure_ascii=False, indent=2)
    
    
    
    
    

