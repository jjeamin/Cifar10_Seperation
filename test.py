import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
from src.vgg import VGG
from src.loader import Custom_CIFAR10
from src.utils import get_logger

sys.path.append('.')

if torch.cuda.is_available():
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')

configs = {
    'task': 'classify',
    'model': 'VGG16',
    'dataset': 'CIFAR10',
    'classes': 10,
    'batch_size': 32,
    'root_path': './datasets/cifar-10-img'
}

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

test_transformer = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

test_datasets = Custom_CIFAR10(root_path=configs['root_path'],
                               class_names=['airplane', 'dog', 'horse'],
                               dtype='test',
                               transformer=test_transformer)

test_loader = torch.utils.data.DataLoader(dataset=test_datasets,
                                          batch_size=configs['batch_size'],
                                          shuffle=True)

# model
model = VGG('VGG16').to(device)
model.load_state_dict(torch.load(f"./{configs['model']}_{configs['dataset']}.pth"))
model.eval()

# cost
criterion = nn.CrossEntropyLoss().to(device)

test_iter = len(test_loader)
n_test_correct = 0
test_loss = 0

for i, (images, labels) in tqdm(enumerate(test_loader), total=test_iter):
    images, labels = images.to(device), labels.to(device)
    # forward
    pred = model(images)
    # acc
    _, predicted = torch.max(pred, 1)
    n_test_correct += (predicted == labels).double().sum().item()
    # loss
    loss = criterion(pred, labels)
    test_loss += loss.item()

test_acc = n_test_correct / (test_iter * configs['batch_size'])
test_loss = test_loss / test_iter

logger.info("TEST [Acc / Loss] : [%f / %f]" % (test_acc, test_loss))
