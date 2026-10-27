import numpy as np
import matplotlib.pyplot as plt

def dm01_image_show():
    """
    展示图片
    :return:
    """
    # 全黑图片：全是0
    data = np.zeros((200, 200, 3))
    # 全白图片：全是255
    data = np.full((200, 200, 3), 255)
    # 灰色图片
    data = np.full((200, 200, 3), 150)
    print("data--->", data)
    plt.imshow(data)
    plt.show()

def dm02_color_image_show():
    """
    加载彩色图片
    :return:
    """
    # 加载彩色图片
    data = plt.imread("../data/img.jpg")
    print("data--->", data)
    print("data.shape--->", data.shape)

    # 绘图
    plt.imshow(data)
    plt.show()

if __name__ == '__main__':
    dm01_image_show()
    dm02_color_image_show()
    print("End!")
