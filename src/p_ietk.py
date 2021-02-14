from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
from ietk import methods
from joblib import Parallel, delayed
from ietk import util
import matplotlib.pyplot as plt


# from multiprocessing import Pool
def train_preprocess(img_path):
    """
    Create circular crop around image centre
    """
    img = cv2.imread(img_path).astype(float)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)

    img /= 255

    I, fg = util.center_crop_and_get_foreground_mask(img)

    enhanced_img = methods.brighten_darken((I.astype(float) / 255), 'A+B+X', focus_region=fg)
    enhanced_img2 = methods.sharpen(enhanced_img, bg=~fg)
    enhanced_img2 = (np.clip(enhanced_img2, 0, 1) * 255).astype(np.uint8)
    """
    plt.imshow(enhanced_img2)
    plt.show()
    """
    cv2.imwrite(img_path.replace("Training_Set/Training", "train_p_ietk"), enhanced_img2)
    return
    # return (enhanced_img2 * 255).astype(np.uint8)


def eval_preprocess(img_path):
    img = cv2.imread(img_path).astype(float)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(float)

    img /= 255

    I, fg = util.center_crop_and_get_foreground_mask(img)

    enhanced_img = methods.brighten_darken((I.astype(float) / 255), 'A+B+X', focus_region=fg)
    enhanced_img2 = methods.sharpen(enhanced_img, bg=~fg)
    enhanced_img2 = (np.clip(enhanced_img2, 0, 1) * 255).astype(np.uint8)
    """
    plt.imshow(enhanced_img2)
    plt.show()
    """
    cv2.imwrite(img_path.replace("Evaluation_Set", "eval_p_ietk"), enhanced_img2)


Parallel(n_jobs=10)(delayed(train_preprocess)(x) for x in glob("../data/Training_Set/Training/*.png"))
Parallel(n_jobs=10)(delayed(eval_preprocess)(x) for x in glob("../data/Evaluation_Set/*.png"))

# with Pool(processes=10) as p:
#     p.map(func=train_preprocess, iterable=glob("../data/Training_Set/Training/*.png"))
#     # result = list(tqdm(imap, total=1920))
#
# with Pool(processes=10) as p:
#     p.map(func=eval_preprocess, iterable=glob("../data/Evaluation_Set/*.png"))
# for x in tqdm(glob("../data/Training_Set/Training/*.png")):
#     train_preprocess(x)
# #     # cv2.imwrite(x.replace("Training_Set/Training", "train_p_ietk"), preprocess(x))
# #
# for x in tqdm(glob("../data/Evaluation_Set/*.png")):
#     eval_preprocess(x)
# cv2.imwrite(x.replace("Evaluation_Set", "eval_p_ietk"), preprocess(x))
