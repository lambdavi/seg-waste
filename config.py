import os
from easydict import EasyDict as edict
from utils.args import get_parser
import time
import torch

parser = get_parser()
args = parser.parse_args()


# init
__C = edict()

cfg = __C
__C.DATA = edict()
__C.NET = edict()
__C.TRAIN = edict()
__C.VAL = edict()
__C.TEST = edict()
__C.VIS = edict()
__C.MODEL = args.model
__C.TASK = args.task
__C.SAVE = args.save
__C.LOAD = args.load
__C.PRED_PATH = args.pred_path

#------------------------------DATA------------------------

__C.DATA.DATASET = 'city' # dataset
__C.DATA.DATA_PATH = 'dataset/'
__C.DATA.NUM_CLASSES = 4 if args.task == "multi" else 1
__C.DATA.IGNORE_LABEL = 255
__C.DATA.IGNORE_LABEL_TO_TRAIN_ID = 19 # 255->19
__C.DATA.NUM_WORKERS = 2                                    

__C.DATA.MEAN_STD = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

#------------------------------TRAIN------------------------

# stage
__C.TRAIN.STAGE = 'encoder' # encoder or all
__C.TRAIN.PRETRAINED_ENCODER = 'ENEt' # Path of the pretrained encoder

# input setting

__C.TRAIN.BATCH_SIZE = args.bs #imgs
__C.TRAIN.IMG_SIZE = (224,448)

__C.TRAIN.GPU_ID = [0]


__C.TRAIN.RESUME = '' #model path

# learning rate settings
__C.TRAIN.LR = args.lr
__C.TRAIN.LR_DECAY = 0.995
__C.TRAIN.NUM_EPOCH_LR_DECAY = 1 #epoches

__C.TRAIN.WEIGHT_DECAY = args.wd

__C.TRAIN.MAX_EPOCH = args.n_epochs

# output 
__C.TRAIN.PRINT_FREQ = 10

now = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())

__C.TRAIN.EXP_NAME =  now \
                    + '_' + __C.TRAIN.STAGE + '_ENet'  \
                    + '_' + __C.DATA.DATASET \
                    + '_' + str(__C.TRAIN.IMG_SIZE) \
                    + '_lr_' + str(__C.TRAIN.LR)


__C.TRAIN.LABEL_WEIGHT = torch.FloatTensor([1,1])

__C.TRAIN.CKPT_PATH = './ckpt'
__C.TRAIN.EXP_LOG_PATH = './logs'
__C.TRAIN.EXP_PATH = './exp'

#------------------------------VAL------------------------
__C.VAL.BATCH_SIZE = args.bs # imgs
__C.VAL.SAMPLE_RATE = 1

#------------------------------TEST------------------------
__C.TEST.GPU_ID = 0

#------------------------------VIS------------------------

__C.VIS.SAMPLE_RATE = 0

__C.VIS.PALETTE_LABEL_COLORS = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]



#------------------------------MISC------------------------
if not os.path.exists(__C.TRAIN.CKPT_PATH):
    os.mkdir(__C.TRAIN.CKPT_PATH)
if not os.path.exists(os.path.join(__C.TRAIN.CKPT_PATH, __C.TRAIN.EXP_NAME)):
    os.mkdir(os.path.join(__C.TRAIN.CKPT_PATH, __C.TRAIN.EXP_NAME))

if not os.path.exists(__C.TRAIN.EXP_LOG_PATH):
    os.mkdir(__C.TRAIN.EXP_LOG_PATH)
if not os.path.exists(__C.TRAIN.EXP_PATH):
    os.mkdir(__C.TRAIN.EXP_PATH)

#================================================================================
#================================================================================
#================================================================================  
