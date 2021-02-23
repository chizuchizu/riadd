import cv2
import matplotlib.pyplot as plt


def load_img(path, h, w):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 元画像による
    img = cv2.resize(img, (h, w))
    return img


def threshold(img):
    """
    人力なのでチューニングが必要

    現在
    ====
    元々の画像　→　明るいところを抽出　→　2値化
    """

    # ヒストグラム平坦化
    equ = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    th_img = cv2.threshold(equ, 254, 255, cv2.THRESH_BINARY)[1]
    return th_img


def max_countour(countours):
    """
    輪郭の中で最も座標の数が多いものを返す（最も大きくて複雑な図形　≒　視神経乳頭）
    """
    count_max = 0
    max_idx = -1
    for i, cnt in enumerate(countours):
        # count_max = max(cnt.shape[0], count_max)

        if cnt.shape[0] > count_max:
            count_max = cnt.shape[0]
            max_idx = i
    return countours[max_idx]


def get_moment(th_img):
    """
    輪郭情報から重心を取得
    """
    countours, hierachy = cv2.findContours(th_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    mu = cv2.moments(max_countour(countours), False)
    x, y = int(mu["m10"] / mu["m00"]), int(mu["m01"] / mu["m00"])

    return x, y


def main(path, s_path, img_size_div_2=60, debug=False):
    h = w = 500
    img = load_img(path, h, w)

    # 視神経乳頭らしきものを抽出（ノイズもある）
    th_img = threshold(img)

    # 重心
    x, y = get_moment(th_img)

    if x - img_size_div_2 < 0:
        # 左にはみ出た場合
        x1 = 0
        x2 = img_size_div_2 * 2
    elif x + img_size_div_2 > w:
        # 右にはみ出た場合
        x1 = w - img_size_div_2 * 2
        x2 = w
    else:
        x1 = x - img_size_div_2
        x2 = x + img_size_div_2

    if y - img_size_div_2 < 0:
        # 上にはみ出た場合
        y1 = 0
        y2 = img_size_div_2 * 2
    elif y + img_size_div_2 > h:
        # 下にはみ出た場合
        y1 = h - img_size_div_2 * 2
        y2 = h
    else:
        y1 = y - img_size_div_2
        y2 = y + img_size_div_2

    new_img = img[y1:y2, x1:x2, :]

    if debug:
        plt.imshow(img[y1:y2, x1:x2, :])
        plt.show()
    else:
        cv2.imwrite(s_path, new_img)

# path = "../data/train_p_1/10.png"
# # path = "../data/Training_Set/Training (1)/10.png"
# main(path)
