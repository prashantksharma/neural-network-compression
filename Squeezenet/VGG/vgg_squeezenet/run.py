from vgg import vggNET19
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
# from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import visdom

viz = visdom.Visdom()
trial = 5

cuda0 = torch.cuda.set_device(1)
device = torch.device('cpu')

##### Uncomment for using MNIST Data-Set and make adequate changes in vgg.py file
# data_train = MNIST('./data/mnist',
#                    download=True,
#                    transform=transforms.Compose([
#                        transforms.Resize((32, 32)),
#                        transforms.ToTensor()]))
# data_test = MNIST('./data/mnist',
#                   train=False,
#                   download=True,
#                   transform=transforms.Compose([
#                       transforms.Resize((32, 32)),
#                       transforms.ToTensor()]))
# data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
# data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
###

## Different Transformations that can be applied on Data-set

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
     
     ])
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
# ])

data_train = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
data_train_loader = torch.utils.data.DataLoader(data_train, batch_size=40,
                                          shuffle=True, num_workers=8)

data_test = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
data_test_loader = torch.utils.data.DataLoader(data_test, batch_size=40,
                                         shuffle=False, num_workers=8)
##
net = vggNET19()
net = net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=2e-3)

cur_batch_win = None
cur_batch_win_opts = {
    'title': 'Epoch Loss Trace',
    'xlabel': 'Batch Number',
    'ylabel': 'Loss',
    'width': 1200,
    'height': 600,
}


def train(epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()

        images = images.to(device)

        output = net(images)

        labels = labels.to(device)

        loss = criterion(output, labels)

        loss_list.append(loss.detach().cpu().item())
        batch_list.append(i+1)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        # Update Visualization
        if viz.check_connection():
            cur_batch_win = viz.line(torch.Tensor(loss_list), torch.Tensor(batch_list),
                                     win=cur_batch_win, name='current_batch_loss',
                                     update=(None if cur_batch_win is None else 'replace'),
                                     opts=cur_batch_win_opts)

        loss.backward()
        optimizer.step()
    torch.save(net,"model_vgg"+str(trial)+".bin")

def test():
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
    	images = images.to(device)
    	
    	output = net(images)
    	labels = labels.to(device)
    	avg_loss += criterion(output, labels).sum()
    	pred = output.detach().max(1)[1]
    	total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), float(total_correct) / len(data_test)))


def train_and_test(epoch):
    train(epoch)
    test()


def main():
    for e in range(1, 20):
        train_and_test(e)


if __name__ == '__main__':
    main()
