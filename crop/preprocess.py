from joblib import Parallel, delayed
from glob import glob

from crop.main import main

Parallel(n_jobs=10)(delayed(main)(x, x.replace("train_p_1", "train_crop_120")) for x in glob("../data/train_p_1/*.png"))
Parallel(n_jobs=10)(delayed(main)(x, x.replace("eval_p_1", "eval_crop_120")) for x in glob("../data/eval_p_1/*.png"))
