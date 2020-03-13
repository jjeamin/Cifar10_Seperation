import torch
import matplotlib.pyplot as plt
from src.utils import *
from src.vgg import VGG, get_index
from src.search import Search, get_filter_idx
from src.prune import *


configs = {
    'task': 'classify',
    'model': 'VGG16',
    'dataset': 'CIFAR10',
    'classes': 10,
    'batch_size': 32,
    'lr': 0.001,
    'root_path': './datasets/cifar-10-img'
}

logger = get_logger('./log.log')
class_names = ['airplane', 'dog', 'horse']

# model
model = VGG('VGG16').to('cuda')
model.load_state_dict(torch.load(f"./{configs['model']}_{configs['dataset']}.pth"))
model.eval()

idx = get_index('VGG16')

for _ in range(0, 1):
    # filter search
    search = Search(model,
                    configs['root_path'],
                    class_names,
                    data_size=2,
                    dtype='train')

    filters = get_filter_idx(search.get_diffs(0))

    # pruning
    model = prune(model,
                  filters,
                  last_prune=False)

    # train / test
    for _ in range(0, 1):
        model = train(model, configs, logger=logger)
        test(model, configs, logger=logger)

# # class 0 : airplane
# cls = 0
#
# # multi to binary
# model = to_binary(model, cls)
#
# for _ in range(0, 1):
#     # binary search
#     search = Search(model,
#                     configs['root_path'],
#                     class_names,
#                     data_size=1000,
#                     dtype='train')
#
#     filters = get_filter_idx(search.get_binary_diffs(0))
#
#     # pruning
#     model = prune(model,
#                   filters,
#                   last_prune=False)
#     print(model)
#     # binary train / test
#     for _ in range(0, 5):
#         model = binary_train(model, configs, logger=logger, cls=cls)
#         binary_test(model, configs, logger=logger, cls=cls)
