import os
import logging
import pickle
import torchvision.transforms as transforms
from logging import handlers


def get_logger(file_name='log.log'):
    # create logger
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    # formatter handler
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    logger.addHandler(stream_hander)

    # file handler
    log_max_size = 10 * 1024 * 1024
    log_file_count = 20

    file_handler = handlers.RotatingFileHandler(filename=file_name, maxBytes=log_max_size, backupCount=log_file_count)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_class_path(path, dtype, cls):
    result = []

    type_path = os.path.join(path, dtype)
    class_path = os.path.join(type_path, cls)

    for img_name in os.listdir(class_path):
        result.append(os.path.join(class_path, img_name))

    return result


def pil_to_tensor(img, size=(32, 32), device='cuda'):
    transformer = transforms.Compose([transforms.Resize(size),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    tensor_img = transformer(img)
    tensor_img = tensor_img.unsqueeze(dim=0).to(device)

    return tensor_img


def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)  # 단 한줄씩 읽어옴

    return data
