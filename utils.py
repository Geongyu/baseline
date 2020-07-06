
import os
import glob

import numpy as np
from skimage import io

try:
    import torch
except:
    pass

import shutil
from collections import Iterable
import matplotlib.pyplot as plt
from slacker import Slacker
import argparse

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.sum_2 = 0 # sum of squares
        self.count = 0
        self.std = 0

    def update(self, val=None, n=1):
        if val!=None: # update if val is not None
            self.val = val
            self.sum += val * n
            self.sum_2 += val**2 * n
            self.count += n
            self.avg = self.sum / self.count
            self.std = np.sqrt(self.sum_2/self.count - self.avg**2)
        else:
            pass

class Logger(object):
    def __init__(self, path, int_form=':03d', float_form=':.4f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.',v)
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')

    def read(self):
        with open(self.path, 'r') as f:
            log = []
            for line in f:
                values = []
                for v in line.split(' '):
                    try:
                        v = float(v)
                    except:
                        pass
                    values.append(v)
                log.append(values)
        return log

def save_checkpoint(state, is_best, work_dir, filename='checkpoint.pth'):
    checkpoint_path = os.path.join(work_dir, filename)
    if is_best:
        torch.save(state, checkpoint_path)
        shutil.copyfile(checkpoint_path,
                        os.path.join(work_dir, 'model_best.pth'))

def load_exam(exam_dir, ftype='png'):

    file_extension = '.'.join(['*', ftype])
    data_paths = glob.glob(os.path.join(exam_dir, file_extension))
    data_paths = sorted(data_paths, key=lambda x: x.split('/')[-1]) # sort by filename

    slices = []
    for data_path in data_paths:
        arr = io.imread(data_path)
        slices.append(arr)

    data_3d = np.stack(slices)

    return data_3d

def pad_3d(data_3d, target_length, padding_value=0):

    d, h, w = data_3d.shape # assume single channel
    margin = target_length - d # assume that h and w are sufficiently larger than target_length
    padding_size = margin // 2
    upper_padding_size = padding_size
    lower_padding_size = margin - upper_padding_size

    padded = np.pad(data_3d, ((upper_padding_size, lower_padding_size),
                              (0,0), (0,0)),
                    'constant', constant_values=(padding_value,padding_value))

    return padded, (upper_padding_size, lower_padding_size)

def calc_stats(data_root):

    data_ids = os.listdir(os.path.join(data_root, 'images'))

    mean_meter = AverageMeter()
    std_meter = AverageMeter()

    for data_id in data_ids:
        image_dir = os.path.join(data_root, 'images', data_id)
        image_3d = load_exam(image_dir, ftype='png')
        pixel_mean = image_3d.mean()
        pixel_std = image_3d.std()

        mean_meter.update(pixel_mean, image_3d.size)
        std_meter.update(pixel_std, image_3d.size)

    total_mean = mean_meter.avg
    total_std = np.sqrt(std_meter.sum_2/std_meter.count)

    return {'mean': total_mean, 'std': total_std}

def draw_self_supervision_curve(work_dir, trn_logger, val_logger, test_logger):

    trn_logger = trn_logger.read()
    val_logger = val_logger.read()
    test_logger = test_logger.read()

    epoch, trn_total_loss, trn_supervision_loss, trn_segmentation_loss, trn_iou, trn_dice, trn_accuracy  = zip(*trn_logger)
    epoch, val_total_loss, val_supervision_loss, val_segmentation_loss, val_iou, val_dice, val_accuracy = zip(*val_logger)
    epoch, tst_total_loss, tst_supervision_loss, tst_segmentation_loss, tst_iou, tst_dice, tst_accuracy = zip(*test_logger)

    plt.figure(1)
    plt.plot(epoch, trn_total_loss, 'navy', label='Train Total Loss')
    plt.plot(epoch, val_total_loss, 'darkorange', label='Validation Total Loss')
    plt.plot(epoch, tst_total_loss, 'yellowgreen', label='Test Total Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Total Loss')
    plt.ylim(0.0, 0.8)
    plt.savefig(os.path.join(work_dir, 'Total_Loss.png'))

    plt.figure(2)
    plt.plot(epoch, trn_segmentation_loss, 'navy', label='Segmentation Train Loss')
    plt.plot(epoch, val_segmentation_loss, 'darkorange', label='Segmentation Validation Loss')
    plt.plot(epoch, tst_segmentation_loss, 'yellowgreen', label='Segmentation Test Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Segmentation Loss')
    plt.ylim(0.0, 0.5)
    plt.savefig(os.path.join(work_dir, 'Segmentation.png'))

    plt.figure(3)
    plt.plot(epoch, trn_supervision_loss, 'navy', label='Supervision Train Loss')
    plt.plot(epoch, val_supervision_loss, 'darkorange', label='Supervision Validation Loss')
    plt.plot(epoch, tst_supervision_loss, 'yellowgreen', label='Supervision Test Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Supervision Loss')
    plt.ylim(0.0, 0.1)
    plt.savefig(os.path.join(work_dir, 'Supervision_Loss.png'))

    plt.figure(4)
    plt.plot(epoch, trn_iou, 'navy', label='Train IoU')
    plt.plot(epoch, val_iou, 'darkorange', label='Validation IoU')
    plt.plot(epoch, tst_iou, 'yellowgreen', label='Test IoU')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare IoU')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'IoU.png'))

    plt.figure(5)
    plt.plot(epoch, trn_dice, 'navy', label='Train DICE Score')
    plt.plot(epoch, val_dice, 'darkorange', label='Validation DICE Score')
    plt.plot(epoch, tst_dice, 'yellowgreen', label='Test DICE Score')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare DICE Score')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'DICE_Score.png'))

    plt.figure(6)
    plt.plot(epoch, trn_accuracy, 'navy', label='Train Accuracy')
    plt.plot(epoch, val_accuracy, 'darkorange', label='Validation Accuracy')
    plt.plot(epoch, tst_accuracy, 'yellowgreen', label='Test Accuracy')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Accuracy')
    plt.ylim(0.5, 1.0)
    plt.savefig(os.path.join(work_dir, 'Accuracy.png'))


def check_correct_forgget(output, target, ephoch,
size, correct, before_correct) :
    if ephoch == 0 :
        correct = ((output-target) == 0).float() # 맞추면 1 틀리면 0
        forget = None
        added = None
    else :
        added = ((output-target) == 0).float()
        forget = np.zeros(size) - (((before_correct - added) == 1).float())
        correct += added
    return correct, forget, added

def send_slack_message(token,channel,messge):
    token = token
    slack = Slacker(token)
    slack.chat.post_message(channel, messge)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class History(object):
    def __init__(self, n_data, train_mode=False, total_epoch=None):
        self.n_data = n_data
        self.counter = np.zeros(n_data)
        if train_mode:
            self.correctness = np.zeros((n_data, total_epoch))
        else:
            self.correctness = np.zeros((n_data, 1, 512, 512))

    def update(self, data_idx, correctness, epoch=None):
        if type(data_idx) is list :
            for i in data_idx :
                self.correctness[i] += correctness.cpu().numpy()
        else :
            self.correctness[data_idx] += correctness.cpu().numpy()

    def get_sum_correctness(self):
        sum_correctness = self.correctness[:, :].sum(axis=1)
        return sum_correctness

    def return_correct(self) :
        re_cor = self.correctness
        tmp = []
        for i in range(len(re_cor)) :
            print(i, " / ", len(re_cor))
            tmp.append(np.where(re_cor[i].flatten() == 0))
        return tmp

    def multi_io(self) :
        tmp = TMP()
        tmp.sum_correct(self.correctness)

def draw_seg_curve(work_dir,logger1,logger2):

    logger1 = logger1.read()
    logger2 = logger2.read()

    epoch, segmentation_loss1, iou1, dice1, cacc1  = zip(*logger1)
    epoch, total_loss2, segmentation_loss2, iou2, dice2, cacc2 = zip(*logger2)

    plt.figure(1)
    plt.plot(epoch, segmentation_loss1, 'navy', label='Segmentation Train Loss')
    plt.plot(epoch, segmentation_loss2, 'darkorange', label='Segmentation Validation Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Segmentation Loss')
    plt.ylim(0.0, 1.1)
    plt.savefig(os.path.join(work_dir, 'Segmentation.png'))

    plt.figure(2)
    plt.plot(epoch, iou1, 'navy', label='Train IoU')
    plt.plot(epoch, iou2, 'darkorange', label='Validation IoU')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare IoU')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'IoU.png'))

    plt.figure(3)
    plt.plot(epoch, dice1, 'navy', label='Train DICE Score')
    plt.plot(epoch, dice2, 'darkorange', label='Validation DICE Score')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare DICE Score')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'DICE_Score.png'))

    plt.figure(4)
    plt.plot(epoch, cacc1, 'navy', label='Train Accuracy')
    plt.plot(epoch, cacc2, 'darkorange', label='Validation Accuracy')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Accuracy')
    plt.ylim(0.5, 1.0)
    plt.savefig(os.path.join(work_dir, 'Accuracy.png'))

def draw_mtl_curve(work_dir,logger1,logger2):

    logger1 = logger1.read()
    logger2 = logger2.read()

    epoch, total_loss1, segmentation_loss1, classification_loss1, iou1, dice1, sacc1, cacc1  = zip(*logger1)
    epoch, total_loss2, segmentation_loss2, classification_loss2, iou2, dice2, sacc2, cacc2  = zip(*logger2)

    plt.figure(1)
    plt.plot(epoch, total_loss1, 'navy', label='Total Train Loss')
    plt.plot(epoch, total_loss2, 'darkorange', label='Total Validation Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Total Loss')
    plt.ylim(0.0, 1.8)
    plt.savefig(os.path.join(work_dir, 'Total_Loss.png'))

    plt.figure(2)
    plt.plot(epoch, segmentation_loss1, 'navy', label='Segmentation Train Loss')
    plt.plot(epoch, segmentation_loss2, 'darkorange', label='Segmentation Validation Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Segmentation Loss')
    plt.ylim(0.0, 1.1)
    plt.savefig(os.path.join(work_dir, 'Segmentation_Loss.png'))

    plt.figure(3)
    plt.plot(epoch, classification_loss1, 'navy', label='Classification Train Loss')
    plt.plot(epoch, classification_loss2, 'darkorange', label='Classification Validation Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Classification Loss')
    plt.ylim(0.0, 0.6)
    plt.savefig(os.path.join(work_dir, 'Classification_Loss.png'))

    plt.figure(4)
    plt.plot(epoch, iou1, 'navy', label='Train IoU')
    plt.plot(epoch, iou2, 'darkorange', label='Validation IoU')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare IoU')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'IoU.png'))

    plt.figure(5)
    plt.plot(epoch, dice1, 'navy', label='Train DICE Score')
    plt.plot(epoch, dice2, 'darkorange', label='Validation DICE Score')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare DICE Score')
    plt.ylim(0.0, 1.0)
    plt.savefig(os.path.join(work_dir, 'DICE_Score.png'))

    plt.figure(6)
    plt.plot(epoch, sacc1, 'navy', label='Train Segmentation Accuracy')
    plt.plot(epoch, sacc2, 'darkorange', label='Validation Segmentation Accuracy')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Segmentation Accuracy')
    plt.ylim(0.5, 1.0)
    plt.savefig(os.path.join(work_dir, 'Seg_Accuracy.png'))

    plt.figure(7)
    plt.plot(epoch, cacc1, 'navy', label='Train Classifcation Accuracy')
    plt.plot(epoch, cacc2, 'darkorange', label='Validation Classifcation Accuracy')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Classifcation Accuracy')
    plt.ylim(0.5, 1.0)
    plt.savefig(os.path.join(work_dir, 'Cls_Accuracy.png'))

    plt.figure(8)
    plt.plot(epoch, sacc1, 'navy', label='Train Segmentation Accuracy')
    plt.plot(epoch, cacc1, 'darkorange', label='Train Classifcation Accuracy')
    plt.plot(epoch, iou1, 'dodgerblue', label='Train IoU')
    plt.plot(epoch, dice1, 'forestgreen', label='Train DICE Score')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Training Curves')
    plt.ylim(0.3, 1.0)
    plt.savefig(os.path.join(work_dir, 'Training_Curves.png'))

    plt.figure(9)
    plt.plot(epoch, sacc2, 'navy', label='Validation Segmentation Accuracy')
    plt.plot(epoch, cacc2, 'darkorange', label='Validation Classifcation Accuracy')
    plt.plot(epoch, iou2, 'dodgerblue', label='Validation IoU')
    plt.plot(epoch, dice2, 'forestgreen', label='Validation DICE Score')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Validation Curves')
    plt.ylim(0.3, 1.0)
    plt.savefig(os.path.join(work_dir, 'Validation_Curves.png'))

'''
class History(object):
    def __init__(self, n_data, size, train_mode=False):
        self.n_data = n_data
        self.pixel = size
        self.counter = np.zeros(n_data)
        #self.n_dimension = n_dimension
        if train_mode:
            #import ipdb; ipdb.set_trace()
            self.correctness = np.zeros((n_data, 1, size[0], size[1]))
           # import ipdb; ipdb.set_trace()
          #  self.confidence = np.zeros((n_data, size[0], size[1], total_epoch))
            self.forgettable = np.zeros((n_data, size[0], size[1]))
        else:
            self.correctness = np.zeros(n_data)
           # self.confidence = np.zeros(n_data)
            self.forgettable = np.zeros(n_data)

    def update(self, data_idx, pixel, correctness, output, epoch=None):
       # probs = torch.nn.functional.softmax(output, dim=1)
       # confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()
        self.counter[data_idx] += 1
        if epoch is not None:
            #import ipdb; ipdb.set_trace()
            self.correctness[data_idx, pixel[0], pixel[1], epoch] += correctness.cpu().numpy()
            #self.confidence[data_idx, pixel, epoch] += confidence.cpu().detach().numpy()
            # forgettable
            if epoch != 0:
                forget_idx = []
                for i in data_idx:
                    if self.correctness[data_idx, pixel[0], pixel[1], epoch-1] == 1 and self.correctness[data_idx, pixel[0], pixel[1], epoch] == 0:
                        forget_idx.append(i)
                self.forgettable[forget_idx, pixel[0], pixel[1], epoch] += 1
        # valid
        else:
            self.correctness[data_idx] += correctness.cpu().numpy()
            #self.confidence[data_idx] += confidence.cpu().detach().numpy()
            # if epoch != 0 and self.correctness[data_idx,epoch - 1] == 1 and self.correctness[data_idx,epoch] == 0:
            #     print('forgettable')
            #     self.forgettable[data_idx,epoch] += correctness.cpu().numpy()

    def get_sum_correctness(self):
        sum_correctness = self.correctness[:, :].sum(axis=1)
        return sum_correctness
'''
'''
class Histroy(object) :
    def __init__(self, size, chennel, n_datas) :
        self.n_datas = n_datas
        self.size = size
        self.channel = chennel

        if type(size) == "tuple" :
            size = size[0]
'''

'''
class History(object):
    def __init__(self, n_data, train_mode=False, total_epoch=None):
        self.n_data = n_data
        self.counter = np.zeros(n_data)
        if train_mode:
            self.correctness = np.zeros((n_data, total_epoch))
            self.confidence = np.zeros((n_data, total_epoch))
            self.rank_confidence = np.zeros((n_data, total_epoch))
        else:
            self.correctness = np.zeros(n_data)
            self.confidence = np.zeros(n_data)
            self.rank_confidence = np.zeros(n_data)
    def update(self, data_idx, correctness, output, rank_target, epoch=None):
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, _ = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()
        self.counter[data_idx] += 1
        if epoch is not None:
            self.correctness[data_idx,epoch] += correctness.cpu().numpy()
            self.confidence[data_idx,epoch] += confidence.cpu().detach().numpy()
            self.rank_confidence[data_idx,epoch] += rank_target.cpu().detach().numpy()
        else:
            self.correctness[data_idx,epoch] += correctness.cpu().numpy()
            self.confidence[data_idx,epoch] += confidence.cpu().detach().numpy()
            self.rank_confidence[data_idx,epoch] += rank_target.cpu().detach().numpy()
    def get_sum_correctness(self):
        sum_correctness = self.correctness[:, :].sum(axis=1)
        return sum_correctness
    def correctness_normalize(self, data, epoch):
        data_min = self.correctness.sum(axis=1).min()
        data_max = float(epoch + 1)
        return (data - data_min) / (data_max - data_min)
    def calc_ranking_target(self, data_idx, epoch=None):
        data_idx = data_idx.cpu().numpy()
        cum_correctness = self.correctness[data_idx,:].sum(axis=1)
        # rescaling correctness values
        cum_correctness = self.correctness_normalize(cum_correctness,epoch)
        n_pair = len(data_idx) // 2
        target1 = cum_correctness[0:n_pair]
        target2 = cum_correctness[n_pair:]
        target = np.zeros(n_pair)
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)
        target = greater + less
        target = torch.from_numpy(target).float().cuda()
        #target = (target1 - target2) / (epoch+1)
        margin = abs(target1 - target2)
        margin = torch.from_numpy(margin).float().cuda()
        return target, margin
    def EAURC(self,epoch):
        conf_correct = sorted(zip(self.confidence[:,epoch], self.correctness[:,epoch]), key=lambda x:x[0], reverse=True)
        sorted_conf, sorted_correct = zip(*conf_correct)
        risk_li, coverage_li = uncertainty_measure.coverage_risk(sorted_conf, sorted_correct)
        eaurc, aurc = uncertainty_measure.cal_EAURC(risk_li)
        return eaurc, aurc
    def rank_EAURC(self,epoch):
        conf_correct = sorted(zip(self.rank_confidence[:,epoch], self.correctness[:,epoch]), key=lambda x: x[0], reverse=True)
        sorted_conf, sorted_correct = zip(*conf_correct)
        risk_li, coverage_li = uncertainty_measure.coverage_risk(sorted_conf, sorted_correct)
        eaurc, aurc = uncertainty_measure.cal_EAURC(risk_li)
        return eaurc, aurc
'''
def draw_curve(work_dir,logger1,logger2):

    logger1 = logger1.read()
    logger2 = logger2.read()

    epoch, loss, iou, dice, accuracy, specificities, sensitivities, precisions = zip(*logger1)
    epoch, loss2, iou2, dice2, accuracy2, specificities2, sensitivities2, precisions2 = zip(*logger2)

    plt.figure(1)
    plt.plot(epoch, loss, 'navy', label='Train Total Loss')
    plt.plot(epoch, loss2, 'darkorange', label='Validation Total Loss')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Loss')
    plt.ylim(0.0, 1.3)
    plt.savefig(os.path.join(work_dir, 'Losses.png'))

    plt.figure(2)
    plt.plot(epoch, iou, 'navy', label='Train IoU')
    plt.plot(epoch, iou2, 'darkorange', label='Validation IoU')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare IoU')
    plt.ylim(0.7, 1.0)
    plt.savefig(os.path.join(work_dir, 'IoU.png'))

    plt.figure(3)
    plt.plot(epoch, dice, 'navy', label='Train Dice Score')
    plt.plot(epoch, dice2, 'darkorange', label='Validation Dice Score')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Dice Score')
    plt.ylim(0.7, 1.0)
    plt.savefig(os.path.join(work_dir, 'Dice_score.png'))

    plt.figure(4)
    plt.plot(epoch, accuracy, 'navy', label='Train Accuracy')
    plt.plot(epoch, accuracy2, 'darkorange', label='Validation Accuracy')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Accuracy')
    plt.ylim(0.7, 1.0)
    plt.savefig(os.path.join(work_dir, 'Accuracy.png'))

    plt.figure(5)
    plt.plot(epoch, specificities, 'navy', label='Train Specificites')
    plt.plot(epoch, specificities2, 'darkorange', label='Validation Specificites')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Specificites')
    plt.ylim(0.7, 1.0)
    plt.savefig(os.path.join(work_dir, 'Specificites.png'))

    plt.figure(6)
    plt.plot(epoch, sensitivities, 'navy', label='Train Sensitivities')
    plt.plot(epoch, sensitivities2, 'darkorange', label='Validation Sensitivities')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Sensitivities')
    plt.ylim(0.7, 1.0)
    plt.savefig(os.path.join(work_dir, 'Sensitivities.png'))

    plt.figure(7)
    plt.plot(epoch, precisions, 'navy', label='Train precisions')
    plt.plot(epoch, precisions2, 'darkorange', label='Validation precisions')
    plt.grid()

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Compare Precisions')
    plt.ylim(0.7, 1.0)
    plt.savefig(os.path.join(work_dir, 'precisions.png'))

class Performance(object) :
    def __init__(self) :
        self.confusionmatrix = [0, 0, 0, 0]
        # TP, TN, FP, FN

    def cal_confusion(self, pred, real) :
        if real == 1 and pred == 1 :
            self.confusionmatrix[0] += 1
        elif real == 1 and pred == 0 :
            self.confusionmatrix[3] += 1
        elif real == 0 and pred == 0 :
            self.confusionmatrix[1] += 1
        else :
            self.confusionmatrix[2] += 1

    def return_matrix(self) :
        return self.confusionmatrix

if __name__ == "__main__" :
    import os
    work_dir = "/daintlab/home/geongyu/MTL/MTL/adam/Self_Supervision/add_factor/Inpainting_half_training_less_decay1"

    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))
    tst_logger = Logger(os.path.join(work_dir, 'test.log'))

    draw_self_supervision_curve(work_dir, trn_logger, val_logger, tst_logger)

    ''' work_dir = "/daintlab/home/geongyu/MTL/MTL/Self_Supervision/Inpainting_half_training2"

    trn_logger = Logger(os.path.join(work_dir, 'train.log'))
    trn_raw_logger = Logger(os.path.join(work_dir, 'train_raw.log'))
    val_logger = Logger(os.path.join(work_dir, 'validation.log'))
    draw_self_supervision_curve(work_dir, trn_logger, val_logger)
    '''
