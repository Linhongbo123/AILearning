import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28*28,64)
        self.fc2 = torch.nn.Linear(64,64)
        self.fc3 = torch.nn.Linear(64,64)
        self.fc4 = torch.nn.Linear(64,10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
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
        for (x,y) in test_data:
            outputs = net.forward(x.view(-1,28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    accuracy = n_correct / n_total
    return accuracy

def main():
    train_data = get_dataloaders(is_train=True)
    test_data = get_dataloaders(is_train=False)

    net = Net()
    print("初始网络的正确率", evaluate(test_data,net))
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(5):
        for (x,y) in train_data:
            net.zero_grad() #会清零模型所有参数
            # optimizer.zero_grad()  #只清零对应优化器管理的参数的梯度（推荐）
            outputs = net.forward(x.view(-1,28*28))
            loss = torch.nn.functional.nll_loss(outputs,y)
            loss.backward()
            optimizer.step()
        print(f"第{epoch+1}轮训练完成，正确率: {evaluate(test_data,net)}")

    for (n,(x,_)) in enumerate(test_data):
        if n >=5:
            break
        predict = torch.argmax(net.forward(x[0].view(-1,28*28)))
        plt.figure(0)
        plt.imshow(x[0].view(28,28),cmap='gray')
        plt.title(f"预测结果: {predict.item()}")
    plt.show()

if __name__ == "__main__":
    main()