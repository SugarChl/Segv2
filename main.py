import numpy as np
import torch

from PIL import Image
from argparse import ArgumentParser
from torch import nn
from torch.optim import SGD, Adam
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
from torchvision.transforms import ToTensor, ToPILImage
import pickle
from dataset import Potsdam, normalise_for_val, normalise_params, Normalise_for_val, ToTensor_for_val
from data import predict_a_image, compute_, computeAll_, ThreeDtoOneD, OneDtoThreeD
from model import *
from tqdm import tqdm
import time

import os

NUM_CHANNELS = 3
NUM_CLASSES = 5


def train(args, model):
    model.train()
    learning_rate = args.learning_rate

    train_loader = DataLoader(Potsdam(args.traindata_dir, iterations=args.iterations),
                              num_workers=args.num_workers,
                              batch_size=args.batch_size,
                              shuffle=True)

    criterion = nn.NLLLoss2d().cuda()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    epoch_loss = []
    right_pixel = []
    all_pixel = []
    for step, sample in enumerate(train_loader):

        images = sample['image'].cuda()
        labels = sample['label'].cuda()
        dsm_images = sample['dsm_img'].cuda()

        input_images = Variable(images)
        dsm_images = Variable(dsm_images)
        targets = Variable(labels)

        outputs = model(input_images, dsm_images)

        soft_output = nn.LogSoftmax()(outputs)

        loss = criterion(soft_output, targets.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        out = soft_output.cpu().detach().numpy().argmax(axis=1)
        tar = targets.cpu().detach().numpy()
        right_pixel.append(sum(sum(sum(out == tar))))
        all_pixel.append(out.size)
        train_acc = sum(right_pixel) / sum(all_pixel)
        if args.steps_loss > 0 and step % args.steps_loss == 0:
            train_loss = sum(epoch_loss) / len(epoch_loss)
            print('loss: {:.4f} Acc: {:.4f} ( [{}/{}])'.format(train_loss,
                                                               train_acc,
                                                               step,
                                                               args.iterations))
            epoch_loss = []
            right_pixel = []
            all_pixel = []

        # 每1000次 保存一下参数。（不测试，因为如果现在就测试，测试结果是有问题的，当时嫌麻烦就没找bug。
        # 目前，是训练结束后，单独进行测试。依次读取所有保存的模型，测试。

        # 这里有两种策略，一个是每次加载当前效果最好的模型，再进行训练
        # 另一个是 从始自终一直训练。
        # 效果好像差不多，我选择第一种
        if step > 0 and step % 1000 == 0:
            filename = 'checkpoints/{}-{}.pth'.format(args.model, step)
            torch.save(model.state_dict(), filename)
            print('save: {} '.format(filename))

        if step > 0 and step % 10000 == 0:
            learning_rate = learning_rate * 0.2  # 每10000次 学习率减小1/5
            optimizer = Adam(model.parameters(), lr=learning_rate)


def val(args, model, data_loader):
    model.eval()

    criterion = nn.NLLLoss2d().cuda()
    epoch_loss = []
    right_pixel = []
    all_pixel = []
    for step, sample in tqdm(enumerate(data_loader)):
        if args.cuda:
            images = sample['image'].cuda()
            labels = sample['label'].cuda()
            dsm_images = sample['dsm_img'].cuda()

        inputs = Variable(images)
        targets = Variable(labels)
        dsm_images = Variable(dsm_images)
        outputs = model(inputs, dsm_images)

        soft_output = nn.LogSoftmax()(outputs)
        loss = criterion(soft_output, targets.long())
        epoch_loss.append(loss.item())

        out = soft_output.cpu().detach().numpy().argmax(axis=1)
        tar = targets.cpu().detach().numpy()

        right_pixel.append(sum(sum(sum(out == tar))))
        all_pixel.append(out.size)

    print('loss: {:.4f} Avg_Acc: {:.4f}'.format(sum(epoch_loss) / len(epoch_loss),
                                                sum(right_pixel) / sum(all_pixel)))

    return sum(epoch_loss) / len(epoch_loss), sum(right_pixel) / sum(all_pixel)


def get_final_predict(args, model):
    date = time.strftime("%Y-%m-%d-%H", time.localtime())
    new_dir = "result/" + args.model + "_" + str(date)
    if os.path.exists(new_dir) is False:
        os.mkdir(new_dir)

    val_input_transform = Compose([
        Normalise_for_val(*normalise_params),
        ToTensor_for_val(),
    ])

    if args.dataset == "Vaihingen":
        test_imgs = ["11", "15", "28", "30", "34"]
    elif args.dataset == "Potsdam":
        test_imgs = ["02_11", "02_12", "04_10", "05_11", "06_7", "07_8", "07_10"]
    else:
        return

    for img in test_imgs:
        if args.dataset == "Vaihingen":
            dsm_img_path = os.path.join(args.testdata_dir, "dsm", "dsm_09cm_matching_area" + img + "_hsv.png")
            img_path = os.path.join(args.testdata_dir, "test", "top_mosaic_09cm_area" + img + ".tif")
        elif args.dataset == "Potsdam":
            dsm_img_path = os.path.join(args.testdata_dir, "dsm", "dsm_potsdam_" + img + ".png")
            img_path = os.path.join(args.testdata_dir, "test", "top_potsdam_" + img + ".tif")

        result = predict_a_image(args, model, img_path, val_input_transform, dsm_img_path)
        Image.fromarray(result.astype('uint8')).convert('RGB').save(new_dir + "/" + str(img) + ".png")

    return computeAll_(new_dir)


def main(args):
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    import random
    RandomSeed = 917
    torch.manual_seed(RandomSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RandomSeed)
    np.random.seed(RandomSeed)
    random.seed(RandomSeed)

    if args.model == "msdfm":
        model = MSDFM(args)
        model.cuda()
    else:
        return

    if args.run_mode == "train":
        train(args, model)
    elif args.run_mode == "test":
        if args.pretrained_model is not None:  # 指定模型
            model.load_state_dict(torch.load(args.pretrained_model), strict=True)
            get_final_predict(args, model)
        else:  # 测试所有模型
            models = os.listdir("checkpoints")
            models.sort()
            for i in models:
                print(i)
                model.load_state_dict(torch.load("checkpoints/" + i), strict=True)
                get_final_predict(args, model)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="msdfm")

    parser.add_argument('--run_mode', default="train")

    parser.add_argument('--image', default="D:\\Seg\\data\\top_mosaic_09cm_area1.tif")
    parser.add_argument('--dataset', default="Potsdam")  # Vaihingen 或者 Potsdam
    parser.add_argument('--pretrained_model', default=None)

    parser.add_argument('--traindata_dir', default="/home/sugarchl1/Segv2/data/Potsdam")

    parser.add_argument('--num-epochs', type=int, default=50)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps-loss', type=int, default=100)
    parser.add_argument('--steps-plot', type=int, default=1)
    parser.add_argument('--steps-save', type=int, default=500)
    parser.add_argument('--min_lr', type=int, default=1e-6)
    parser.add_argument('--iterations', type=int, default=500000)

    main(parser.parse_args())
