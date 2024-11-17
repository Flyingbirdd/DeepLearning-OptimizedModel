import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import random

# 设置随机种子以确保结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用GPU

# 定义优化后的模型类，增加了更深的层和Leaky ReLU激活函数
class OptimizedModel(nn.Module):
    def __init__(self):
        super(OptimizedModel, self).__init__()
        self.fc1 = nn.Linear(10, 512)  # 增加了更大的隐藏层以更好地表示数据
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 1)  # 输出层
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)  # 使用Leaky ReLU
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)
        x = self.dropout(x)
        return torch.sigmoid(self.fc5(x))

# 创建一个更大的随机数据集，样本数为20000
class RandomDataset(Dataset):
    def __init__(self, size=20000):  # 增加数据集的大小为20000
        self.data = torch.randn(size, 10)  # 随机特征
        self.target = torch.randint(0, 2, (size, 1)).float()  # 二分类标签

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.target[index]

# 带有早停和更好的学习率调度器的训练循环
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10, patience=3):
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        scheduler.step()  # 调整学习率
        print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

        # 早停
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

# 生成预测结果并保存为CSV
def generate_predictions(model, train_loader, filename="predictions.csv"):
    model.eval()
    predictions = []

    with torch.no_grad():
        for inputs, _ in train_loader:
            outputs = model(inputs)
            predictions.extend(outputs.squeeze().cpu().numpy())

    df = pd.DataFrame({'label': predictions})
    df.to_csv(filename, index=False)
    print(f"Predictions saved to {filename}")

# 计算评估指标
def calculate_metrics(true_labels, predicted_scores):
    auc = roc_auc_score(true_labels, predicted_scores)
    novelty = -np.sum(predicted_scores * np.log(predicted_scores + 1e-10))  # 添加小的epsilon避免log(0)
    print(f"AUC: {auc:.4f}")
    print(f"Novelty: {novelty:.4f}")
    print(f"Sample label mean: {true_labels.mean():.4f}")
    print(f"Predicted score mean: {predicted_scores.mean():.4f}")

# 设置随机种子确保结果可复现
set_seed()

# ------------------- 数据加载与处理 -------------------

# 定义文件路径
train_file_path = r"C:\Program Files\recommend-practice\recommend_trx_simple\train.csv"
test_file_path = r"C:\Program Files\recommend-practice\recommend_trx_simple\test.feature.csv"

# 加载数据
train_data = pd.read_csv(train_file_path)
test_data = pd.read_csv(test_file_path)

# 检查数据的列名并确保没有错误
print("Train Data Columns:", train_data.columns)
print("Test Data Columns:", test_data.columns)

# 如果 'target' 列存在，则从数据集中删除它
if 'target' in train_data.columns:
    train_features = train_data.drop(columns=['target'])
else:
    print("'target' column not found in train data. Check the column names.")
    train_features = train_data  # 保留数据集原样

# 获取标签
train_labels = train_data['target'] if 'target' in train_data.columns else None

# 如果没有目标列，处理可能的错误
if train_labels is None:
    print("Warning: 'target' column is missing in the train dataset. Ensure proper labeling.")
    # This would need to be handled appropriately depending on your use case.

# ------------------- 模型训练与评估 -------------------

# 初始化优化后的模型
model = OptimizedModel()

# 准备数据集和数据加载器
train_dataset = RandomDataset(size=20000)  # 设置样本数为20000
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义损失函数和优化器
criterion = nn.BCELoss()  # 二元交叉熵损失
optimizer = optim.AdamW(model.parameters(), lr=0.001)  # AdamW优化器

# 学习率调度器，带有预热和衰减
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

# 训练模型（神经网络）
train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10)

# 从神经网络生成预测并保存为CSV
generate_predictions(model, train_loader, filename="recommend_trx_optimized.csv")

# 获取神经网络的真实标签和预测得分
true_labels = train_dataset.target.squeeze().cpu().numpy()
predicted_scores_nn = pd.read_csv("recommend_trx_optimized.csv")['label'].values

# 计算神经网络的评估指标
calculate_metrics(true_labels, predicted_scores_nn)

# ------------------------ 随机森林 ------------------------

# 准备随机森林数据集（将数据转换为NumPy格式以适应sklearn）
train_data_np = train_dataset.data.numpy()
train_labels_np = train_dataset.target.numpy().squeeze()

# 初始化并训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
rf_model.fit(train_data_np, train_labels_np)

# 从随机森林生成预测
predicted_scores_rf = rf_model.predict_proba(train_data_np)[:, 1]  # 预测为正类的概率

# 保存随机森林预测结果为CSV
df_rf = pd.DataFrame({'label': predicted_scores_rf})
df_rf.to_csv("recommend_trx_rf.csv", index=False)
print(f"Random Forest predictions saved to recommend_trx_rf.csv")

# 计算随机森林的评估指标
calculate_metrics(true_labels, predicted_scores_rf)
