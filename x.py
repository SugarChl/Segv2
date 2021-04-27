import gdal
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def dsm2rgb(img):
    fig = plt.figure()

    ax = fig.add_subplot(111)
    im = ax.imshow(img, cmap=plt.cm.hsv)
    c = im.make_image(renderer="colormap", unsampled=True)
    hotpic = Image.fromarray(c[0]).convert("RGB")
    return hotpic


if __name__ == "__main__":
    img = gdal.Open("D:\\dsm_potsdam_02_10.tif")
    cols = img.RasterXSize  # 图像长度
    rows = (img.RasterYSize)  # 图像宽度
    b0 = np.array(img.GetRasterBand(1).ReadAsArray(0, 0, cols, rows))[:, :, np.newaxis]

    print(b0.shape)  # (6000, 6000, 1)
    b0 = b0.squeeze()  # (6000, 6000)
    x = dsm2rgb(b0)  # (6000, 6000, 3)
    print(x.show())