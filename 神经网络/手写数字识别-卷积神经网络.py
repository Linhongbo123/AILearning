import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5,padding=2),  # 卷积层
            nn.BatchNorm2d(32),                   # 批归一化层
            nn.ReLU(),                             # 激活函数
            nn.MaxPool2d(kernel_size=2),          # 池化层
        )

        self.fc = nn.Linear(32 * 14 * 14,10)  #全连接层 这里只有一层

    def forward(self, x):
            x = self.feature_extractor(x)  # 特征提取
            x = x.view(x.size(0), -1)      # 展平
            x = self.fc(x)                 # 全连接层分类
            return x
        
def get_dataloaders(is_train):
    to_tensor= transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST('mnist', is_train, transform=to_tensor, download=True)
    #一批数据有64个样本
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataloader


def evaluate(test_data,net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (images,labels) in test_data:
            # images = images.cuda()
            # labels = labels.cuda()
            outputs = net.forward(images)
            for i, output in enumerate(outputs):
                if torch.argmax(output) == labels[i]:
                    n_correct += 1
                n_total += 1
    accuracy = n_correct / n_total
    return accuracy

def main():
    train_data = get_dataloaders(is_train=True)
    test_data = get_dataloaders(is_train=False)

    cnn = CNN()
    # cnn.cuda()  #如果有GPU的话就用GPU加速训练
    optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)

    loss_func = nn.CrossEntropyLoss()

    for epoch in range(5):
        for index,(images,labels) in enumerate(train_data):
            # images = images.cuda()
            # labels = labels.cuda()
            optimizer.zero_grad() 
            outputs = cnn.forward(images)
            loss = loss_func(outputs,labels)
            loss.backward()
            optimizer.step()
        print(f"第{epoch+1}轮训练完成，正确率: {evaluate(test_data,cnn)}")

    for (n,(x,_)) in enumerate(test_data):
        if n >=5:
            break
        # predict = torch.argmax(cnn.forward(x[0].cuda()))
        predict = torch.argmax(cnn.forward(x[0].unsqueeze(0)))
        plt.figure(0)
        plt.imshow(x[0].view(28,28),cmap='gray')
        plt.title(f"预测结果: {predict.item()}")
    plt.show()

if __name__ == "__main__":
    main()

