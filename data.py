import numpy as np
from PIL import Image
import os
from torch.autograd import Variable
from torch import nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report, precision_recall_fscore_support
import cv2
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt

top_filename = "top_mosaic_09cm_area1.tif"
dsm_filename = "dsm_09cm_matching_area1.tif"
gt_filename = "top_mosaic_09cm_area1.tif"

from sklearn.metrics import confusion_matrix


def trans():
    data = os.listdir("D:\ISPRS\data\data\gt")
    for i in data:
        img = Image.open("D:\ISPRS\data\data\gt\\" + i)
        img.save("D:\ISPRS\data\data\gt\\" + i.replace("tif", "png"))


def crop_save(type, t, im, left, up, right, bottom, q, n):
    w = right - left
    h = bottom - up
    if h != w:
        print("****")
    cropped = im.crop((left, up, right, bottom))
    if t == "top":
        cropped.save("D:\ISPRS\data\data\\top\\" + str(type) + "\\top_" + str(q) + "_" + str(n) + ".jpg")
    elif t == "gt":
        cropped.save("D:\ISPRS\data\data\\gt\\gt_" + str(q) + "_" + str(n) + ".png")
    elif t == "dsm":
        cropped.save("D:\ISPRS\data\data\dsm\\gray_val\\dsm_" + str(q) + "_" + str(n) + ".png")


def data_aug(train_path, gt_path=None):
    imgs = os.listdir(train_path)
    for img_name in tqdm(imgs):
        img = Image.open(os.path.join(train_path, img_name))
        img.rotate(90).save(os.path.join(train_path, img_name.replace(".png", "_1.png")))
        img.rotate(180).save(os.path.join(train_path, img_name.replace(".png", "_2.png")))
        img.rotate(270).save(os.path.join(train_path, img_name.replace(".png", "_3.png")))
        img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(train_path, img_name.replace(".png", "_4.png")))
        img.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(train_path, img_name.replace(".png", "_5.png")))

        # img = Image.open(os.path.join(gt_path, img_name.replace(".jpg", ".png").replace("top", "gt")))

        # img.rotate(90).save(os.path.join(gt_path, img_name.replace(".jpg", "_1.png").replace("top", "gt")))
        # img.rotate(180).save(os.path.join(gt_path, img_name.replace(".jpg", "_2.png").replace("top", "gt")))
        # img.rotate(270).save(os.path.join(gt_path, img_name.replace(".jpg", "_3.png").replace("top", "gt")))
        # img.transpose(Image.FLIP_LEFT_RIGHT).save(os.path.join(gt_path, img_name.replace(".jpg", "_4.png").replace("top", "gt")))
        # img.transpose(Image.FLIP_TOP_BOTTOM).save(os.path.join(gt_path, img_name.replace(".jpg", "_5.png").replace("top", "gt")))


def dsm2hotpic(img):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(img, cmap=plt.cm.hsv)
    c = im.make_image(renderer="colormap", unsampled=True)

    hotpic = Image.fromarray(c[0]).convert("RGB")
    return hotpic


def dsm_seg(files, size=512, overlap=0.75):
    for q in files:
        dsm_im = Image.open("D:\ISPRS\data\ISPRS_data\dsm\\dsm_09cm_matching_area" + str(q) + ".tif")
        print(dsm_im)
        dsm_im = np.array(dsm_im)

        im = dsm2hotpic(dsm_im)
        t = "dsm"
        left = 0
        up = 0
        right = 0
        bottom = 0
        n = 1
        for up in [i * size * (1 - overlap) for i in range(50)]:
            if up + size >= im.size[1]:
                bottom = im.size[1] - 1
                up = bottom - size
                for left in [i * size * (1 - overlap) for i in range(50)]:
                    if left + size >= im.size[0]:
                        right = im.size[0] - 1
                        left = right - size
                        crop_save(type, t, im, int(left), int(up), int(right), int(bottom), q, n)
                        n += 1
                        break
                    else:
                        right = left + size
                        crop_save(type, t, im, int(left), int(up), int(right), int(bottom), q, n)
                        n += 1
                break
            else:
                bottom = up + size
                for left in [i * size * (1 - overlap) for i in range(50)]:
                    if left + size >= im.size[0]:
                        right = im.size[0] - 1
                        left = right - size
                        crop_save(type, t, im, int(left), int(up), int(right), int(bottom), q, n)
                        n += 1
                        break
                    else:
                        right = left + size
                        crop_save(type, t, im, int(left), int(up), int(right), int(bottom), q, n)
                        n += 1


def data_seg(files, type, size=512, overlap=0.75):
    for q in files:
        top_im = Image.open("D:\ISPRS\data\ISPRS_data/top/top_mosaic_09cm_area" + str(q) + ".tif")
        gt_im = Image.open("D:\ISPRS\data\ISPRS_data/gts_for_participants/top_mosaic_09cm_area" + str(q) + ".tif")
        for im, t in [(top_im, "top"), (gt_im, "gt")]:
            left = 0
            up = 0
            right = 0
            bottom = 0
            n = 1
            for up in [i * size * (1 - overlap) for i in range(50)]:
                if up + size >= im.size[1]:
                    bottom = im.size[1] - 1
                    up = bottom - size
                    for left in [i * size * (1 - overlap) for i in range(50)]:
                        if left + size >= im.size[0]:
                            right = im.size[0] - 1
                            left = right - size
                            # print(int(left), int(up), int(right), int(bottom))
                            crop_save(type, t, im, int(left), int(up), int(right), int(bottom), q, n)
                            n += 1
                            break
                        else:
                            right = left + size
                            # print(int(left), int(up), int(right), int(bottom))
                            crop_save(type, t, im, int(left), int(up), int(right), int(bottom), q, n)
                            n += 1
                    break
                else:
                    bottom = up + size
                    for left in [i * size * (1 - overlap) for i in range(50)]:
                        if left + size >= im.size[0]:
                            right = im.size[0] - 1
                            left = right - size
                            # print(int(left), int(up), int(right), int(bottom))
                            crop_save(type, t, im, int(left), int(up), int(right), int(bottom), q, n)
                            n += 1
                            break
                        else:
                            right = left + size
                            # print(int(left), int(up), int(right), int(bottom))
                            crop_save(type, t, im, int(left), int(up), int(right), int(bottom), q, n)
                            n += 1


def ThreeDtoOneD(img, shape):
    def t(gt, a, b, c, t):
        w1 = gt[:, :, 0] == a
        w2 = gt[:, :, 1] == b
        w3 = gt[:, :, 2] == c
        x = w1 * w2 * w3
        f = np.where(x > 0, t, 0)
        return f

    gt = np.zeros(shape)
    gt += t(img, 0, 0, 255, 0)
    gt += t(img, 255, 255, 255, 1)
    gt += t(img, 0, 255, 0, 2)
    gt += t(img, 0, 255, 255, 3)
    gt += t(img, 255, 255, 0, 4)
    #gt += t(img, 255, 0, 0, 5)
    gt += t(img, 0, 0, 0, 5)
    return gt



def OneDtoThreeD(img):
    # img = np.select([img < 10], [img-1]).astype(int)

    R = np.choose(img, [0, 255, 0, 0, 255, 255], mode="wrap")
    G = np.choose(img, [0, 255, 255, 255, 255, 0], mode="wrap")
    B = np.choose(img, [255, 255, 0, 255, 0, 0], mode="wrap")

    img = np.concatenate([R[:, :, np.newaxis],
                          G[:, :, np.newaxis],
                          B[:, :, np.newaxis]], axis=2)
    return img


def Crop(img, i, j, h, w):
    # print(i,j)
    i, j = int(i), int(j)
    # print(img.shape)
    return img[j: j + w, i:i + h, :]


def predict_a_image(args, model, img_path, input_transform, dsm=None, imgsize=256, overlap=0.75):
    if args.dataset == "Potsdam":
        N_classes = 6
    else:
        N_classes = 5

    # 可选操作 不重要
    # def trans(tile):
    #     # print(tile.shape)
    #     for i in range(48):
    #         tile[:, :, i, :] *= 0.1
    #         tile[:, :, -1 - i, :] *= 0.1
    #         tile[:, :, :, i] *= 0.1
    #         tile[:, :, :, -1 - i] *= 0.1
    #     return tile

    def predict_a_tile(model, img, input_transform, dsm=None):
        if dsm is not None:
            dsm = input_transform(dsm).unsqueeze(0)
            dsm_input = Variable(dsm.cuda())
        img = input_transform(img).unsqueeze(0)
        input = Variable(img.cuda())
        if dsm is not None:
            output = model(input, dsm_input)
        else:
            output = model(input)

        return output.cpu().detach().numpy()

    model = model.cuda()
    model.eval()
    img = Image.open(img_path)
    assert img.size[0] >= imgsize and img.size[1] >= imgsize

    if dsm is not None:
        # dsm_img = Image.fromarray(np.load(dsm))
        # dsm_img = np.load(dsm)
        dsm_img = np.array(Image.open(dsm))
        # print(dsm_img.shape)
        # dsm_img = dsm2hotpic(dsm_img)
        # assert dsm_img.size[0] == img.size[0] and dsm_img.size[1] == img.size[1]

    final_result = np.zeros([N_classes, img.size[1], img.size[0]])

    # 把大的图片分割成小的图片以供模型使用
    # overlap = 0.75 意思是   相邻的两个图片 有75%是重叠的。  为了解决图片的边缘问题
    for up in [i * imgsize * (1 - overlap) for i in range(500)]:
        if up + imgsize >= img.size[1]:
            bottom = img.size[1] - 1
            up = bottom - imgsize
            for left in [i * imgsize * (1 - overlap) for i in range(500)]:
                if left + imgsize >= img.size[0]:
                    right = img.size[0] - 1
                    left = right - imgsize

                    img_tile = img.crop((left, up, right, bottom))
                    if dsm is not None:
                        dsm_tile = Crop(dsm_img, left, up, 256, 256)
                        result_tile = predict_a_tile(model, img_tile, input_transform, dsm_tile)
                    else:
                        result_tile = predict_a_tile(model, img_tile, input_transform)

                    assert result_tile.shape[3] == bottom - up and result_tile.shape[2] == right - left
                    final_result[:, int(up):int(bottom), int(left):int(right)] += trans(result_tile).squeeze(0)
                    break
                else:
                    right = left + imgsize
                    img_tile = img.crop((left, up, right, bottom))
                    if dsm is not None:
                        dsm_tile = Crop(dsm_img, left, up, 256, 256)
                        result_tile = predict_a_tile(model, img_tile, input_transform, dsm_tile)
                    else:
                        result_tile = predict_a_tile(model, img_tile, input_transform)

                    assert result_tile.shape[3] == bottom - up and result_tile.shape[2] == right - left
                    final_result[:, int(up):int(bottom), int(left):int(right)] += trans(result_tile).squeeze(0)
            break
        else:
            bottom = up + imgsize
            for left in [i * imgsize * (1 - overlap) for i in range(500)]:
                if left + imgsize >= img.size[0]:
                    right = img.size[0] - 1
                    left = right - imgsize
                    img_tile = img.crop((left, up, right, bottom))
                    if dsm is not None:
                        dsm_tile = Crop(dsm_img, left, up, 256, 256)
                        result_tile = predict_a_tile(model, img_tile, input_transform, dsm_tile)
                    else:
                        result_tile = predict_a_tile(model, img_tile, input_transform)

                    assert result_tile.shape[3] == bottom - up and result_tile.shape[2] == right - left
                    final_result[:, int(up):int(bottom), int(left):int(right)] += trans(result_tile).squeeze(0)
                    break
                else:
                    right = left + imgsize
                    img_tile = img.crop((left, up, right, bottom))
                    if dsm is not None:
                        dsm_tile = Crop(dsm_img, left, up, 256, 256)
                        result_tile = predict_a_tile(model, img_tile, input_transform, dsm_tile)
                    else:
                        result_tile = predict_a_tile(model, img_tile, input_transform)

                    assert result_tile.shape[3] == bottom - up and result_tile.shape[2] == right - left
                    final_result[:, int(up):int(bottom), int(left):int(right)] += trans(result_tile).squeeze(0)

    final_result = final_result.argmax(axis=0)  # 生成的final_result 每个像素点有N个数据，分别代表N个类别的概率。（取最大
    return OneDtoThreeD(final_result)


def compute_(predict_img, gt_img, for_all=False):
    if for_all is False:
        predict_img = np.array(Image.open(predict_img))
        gt_img = np.array(Image.open(gt_img))
        trans_predict_img = ThreeDtoOneD(predict_img, shape=[predict_img.shape[0], predict_img.shape[1]]).flatten()
        trans_gt_img = ThreeDtoOneD(gt_img, shape=[gt_img.shape[0], gt_img.shape[1]]).flatten()
    else:
        trans_predict_img = predict_img
        trans_gt_img = gt_img

    assert len(trans_gt_img) == len(trans_predict_img)

    # 以下的操作是去除带边界的gt的黑色像素部分
    x3 = (trans_gt_img == 5)
    x4 = (trans_gt_img != 5)
    x3 = x3 * 5
    trans_predict_img = x4 * trans_predict_img + x3
    index = np.argwhere(trans_gt_img == 5)
    trans_gt_img = np.delete(trans_gt_img, index)
    trans_predict_img = np.delete(trans_predict_img, index)

    # OverallAcc = sum(trans_predict_img == trans_gt_img) / trans_predict_img.size
    # f1 = f1_score(trans_gt_img, trans_predict_img, average='macro')
    x = precision_recall_fscore_support(trans_gt_img, trans_predict_img)[0]
    ## print(classification_report(trans_gt_img, trans_predict_img, labels=[1.0, 2.0, 3.0, 4.0, 5.0], target_names=["Build", "Road", "Tree", "Shurb", "Car"]))
    ## print("F1: {:.4f}  Acc: {:.2f}%".format(f1, OverallAcc*100))
    # print("   Build     Road     Tree     Shrub    Car      F1     OverallAcc")
    # print("    {:.2f}%    {:.2f}%   {:.2f}%   {:.2f}%   {:.2f}%   {:.2f}  {:.2f}%"
    #      .format(x[0] * 100,
    #              x[1] * 100,
    #              x[2] * 100,
    #              x[3] * 100,
    #              x[4] * 100,
    #              f1 * 100,
    #              OverallAcc * 100))
    # return OverallAcc


def compute_acc(trans_predict_img, trans_gt_img):
    # x3 = (trans_gt_img == 5)
    # x4 = (trans_gt_img != 5)
    # x3 = x3 * 5
    # trans_predict_img = x4 * trans_predict_img + x3
    index = np.argwhere(trans_gt_img == 5)
    trans_gt_img = np.delete(trans_gt_img, index)
    trans_predict_img = np.delete(trans_predict_img, index)
    true = sum(trans_predict_img == trans_gt_img)
    false = trans_predict_img.size - true
    return true, false


def compute_confusion_matrix(y_pred, y_true):
    labels = ["Build", "Imp. Surf.", "Low Veg.", "Tree", "Car", "Clutter"]

    tick_marks = np.array(range(len(labels))) + 0.5

    def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        # plt.title(title)
        # plt.colorbar()
        xlocations = np.array(range(len(labels)))
        plt.xticks(xlocations, labels, rotation=90)
        plt.yticks(xlocations, labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    cm = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 8), dpi=120)

    ind_array = np.arange(len(labels))
    x, y = np.meshgrid(ind_array, ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c > 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c*100,), color='red', fontsize=10, va='center', ha='center')
    # offset the tick
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    plot_confusion_matrix(cm_normalized, title='')
    # show confusion matrix
    plt.savefig('potsdam_confusion_matrix.png', format='png')
    plt.show()


def computeAll_(dir):
    gt_dir = "D:\Seg\data\Vaihingen\egt"
    # gt_dir = "D:\Seg\data\Potsdam\gt"
    test_img = [11, 15, 28, 30, 34]
    # test_img = ["7_10", "7_8", "6_7", "5_11", "4_10", "2_12", "2_11"]
    # test_img = [2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 27, 29, 31, 33, 35, 38]
    predict_img = np.zeros([0])
    gt = np.zeros([0])

    for img in test_img:
        print(img)
        # predict_img = np.zeros([0])
        # gt = np.zeros([0])
        gt_img = np.array(Image.open(os.path.join(gt_dir, "top_mosaic_09cm_area{}.tif".format(img))))
        oneDgtimg = ThreeDtoOneD(gt_img, shape=[gt_img.shape[0], gt_img.shape[1]]).flatten()
        index = np.argwhere(oneDgtimg == 5)
        trans_gt_img = np.delete(oneDgtimg, index)
        gt = np.concatenate([gt, trans_gt_img], axis=0)

        pre_img = np.array(Image.open(os.path.join(dir, str(img) + ".png")))
        oneDpreimg = ThreeDtoOneD(pre_img, shape=[pre_img.shape[0], pre_img.shape[1]]).flatten()
        trans_predict_img = np.delete(oneDpreimg, index)
        predict_img = np.concatenate([predict_img, trans_predict_img], axis=0)

        # true, false = compute_acc(predict_img, gt)
        # t += true
        # f += false
        # print(float(t) / float(t + f))

    # return compute_(predict_img, gt,for_all=True)


if __name__ == "__main__":
    # dsm_seg([11])
    # dsm_im = Image.open("D:\ISPRS\data\ISPRS_data\dsm\\dsm_09cm_matching_area1.tif")
    predict_img = np.load("potsdam_pred.npy")
    gt = np.load("potsdam_gt.npy")
    compute_confusion_matrix(y_pred=predict_img, y_true=gt)
    #computeAll_("C:\\Users\\73876\go\msdfm_v")
    # print(dsm_im)
    # dsm_im.show()
    # dsm_im = np.array(dsm_im)
    # print(dsm_im)
    # im = dsm2hotpic(dsm_im)
