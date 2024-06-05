import os
import sys

from pathlib import Path

FILE = Path(__file__).resolve() #该detect.py文件的绝对路径
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path: #模块的查询路径的列表
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relativel


#from utils.general import check_requirements
from extraplusKalman2 import parse_opt,Initial,inference

opt = parse_opt()
#check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
model,dataset,names,pt,bs = Initial(**vars(opt))
inference(model,dataset,names,pt,bs,**vars(opt))