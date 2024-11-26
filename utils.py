import os
import os.path
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.utils.data as data
import pandas as pd 
import torchvision
from torchvision import transforms
from enum import Enum
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from  sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
from simplejson import OrderedDict
import h5py
import pdb


"""
@Author: Bidur Khanal
contains some important util functions,
some util functions adopted from https://github.com/pytorch/examples/blob/main/imagenet/main.py

"""


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)



def get_acc(true_labels, predicted_labels):
    """Args: 
        true_labels: given groundtruth labels
        predicted_labels: predicted labels from the model

    Returns:
        per-class-accuracies(list): a list of per-class accuracies
        average-per-class accuracy: average of per-class accuracies
    """
    
    cm = confusion_matrix(true_labels, predicted_labels)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_acc = cm.diagonal()
    avg_acc =  np.mean(class_acc)
    overall_acc = accuracy_score(true_labels, predicted_labels)
    return class_acc, avg_acc, overall_acc



# compute the AUC per class 
# implementation adapted from 
# https://stackoverflow.com/questions/39685740/calculate-sklearn-roc-auc-score-for-multi-class
def class_report(y_true, y_pred, y_score=None, average='micro', sklearn_cls_report = True):

    if isinstance(y_true, list):
        y_true = np.array(y_true)

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    if y_true.shape != y_pred.shape:
        print("Error! y_true %s is not the same shape as y_pred %s" % (
              y_true.shape,
              y_pred.shape)
        )
        return

    if sklearn_cls_report:
       
        class_report_df = pd.DataFrame(classification_report(y_true= y_true, y_pred = y_pred, output_dict=True)).transpose()
    else:

        lb = LabelBinarizer()

        if len(y_true.shape) == 1:
            lb.fit(y_true)

        #Value counts of predictions
        labels, cnt = np.unique(
            y_pred,
            return_counts=True)
        n_classes = len(labels)
        pred_cnt = pd.Series(cnt, index=labels)

        metrics_summary = precision_recall_fscore_support(
                y_true=y_true,
                y_pred=y_pred,
                labels=labels)

        avg = list(precision_recall_fscore_support(
                y_true=y_true, 
                y_pred=y_pred,
                average='weighted'))

        metrics_sum_index = ['precision', 'recall', 'f1-score', 'support']
        class_report_df = pd.DataFrame(
            list(metrics_summary),
            index=metrics_sum_index,
            columns=labels)

        support = class_report_df.loc['support']
        total = support.sum() 
        class_report_df['avg / total'] = avg[:-1] + [total]

        class_report_df = class_report_df.T
        class_report_df['pred'] = pred_cnt
        class_report_df['pred'].iloc[-1] = total

        if not (y_score is None):

            try:
                fpr = dict()
                tpr = dict()
                roc_auc = dict()
                for label_it, label in enumerate(labels):
                    fpr[label], tpr[label], _ = roc_curve(
                        (y_true == label).astype(int), 
                        y_score[:, label_it])

                    roc_auc[label] = auc(fpr[label], tpr[label])

                if average == 'micro':
                    if n_classes <= 2:
                        fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                            lb.transform(y_true).ravel(), 
                            y_score[:, 1].ravel())
                    else:
                        fpr["avg / total"], tpr["avg / total"], _ = roc_curve(
                                lb.transform(y_true).ravel(), 
                                y_score.ravel())

                    roc_auc["avg / total"] = auc(
                        fpr["avg / total"], 
                        tpr["avg / total"])

                elif average == 'macro':
                    # First aggregate all false positive rates
                    all_fpr = np.unique(np.concatenate([
                        fpr[i] for i in labels]
                    ))

                    # Then interpolate all ROC curves at this points
                    mean_tpr = np.zeros_like(all_fpr)
                    for i in labels:
                        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

                    # Finally average it and compute AUC
                    mean_tpr /= n_classes

                    fpr["macro"] = all_fpr
                    tpr["macro"] = mean_tpr

                    roc_auc["avg / total"] = auc(fpr["macro"], tpr["macro"])

                class_report_df['AUC'] = pd.Series(roc_auc)

            except:
                return class_report_df

    return class_report_df



#### function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    #transpose the matrix to make x-axis True Class and Y-axis Predicted Class
    cm= np.transpose(cm)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

   
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[0]),
           yticks=np.arange(cm.shape[1]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           
           #here we are not printing the title
           #title=title,
           xlabel='True label',
           ylabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig,ax



class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        
    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res




class custom_ISIC_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", split_type = "train", transform=None, target_transform=None, num_classes= 9, seed=1):

        self.root = root
        self.as_rgb = True
        self.split_type = split_type
        
        with h5py.File(os.path.join(root,"ISIC_dataset/", split_type+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.split_type == "train":

            return image, target, index
        else:
            return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    

class custom_hyper_kvasir_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", split_type = "train", transform=None, target_transform=None, num_classes= 9, seed=1):

        self.root = root
        self.as_rgb = True
        self.split_type = split_type
        
        with h5py.File(os.path.join(root,"hyper_kvasir/", split_type+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.split_type == "train":
            return image, target, index
        else:
            return image, target
        
    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class custom_DRD_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", split_type = "train", transform=None, target_transform=None, num_classes= 9, seed=1):

        self.root = root
        self.as_rgb = True
        self.split_type = split_type
        
        with h5py.File(os.path.join(root,"DRD_dataset/", split_type+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.split_type == "train":
            return image, target, index
        else:
            return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str



class custom_Imbalanced_Histopathology_faster(data.Dataset):
    """
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset

    """

    def __init__(self, root = "data/", split_type = "train", transform=None, target_transform=None, num_classes= 9, seed=1):

        self.root = root
        self.as_rgb = True
        self.split_type = split_type

        with h5py.File(os.path.join(root,"Imbalanced_Histopathology_dataset/", split_type+".hdf5"), 'r') as hf:
            self.targets = hf["dataset"]["targets"][:]
            self.images = hf["dataset"]["images"][:]

        self.transform = transform
        self.target_transform = target_transform
        
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """

        index = int(index)
        image, target = self.images[index], self.targets[index]
        image = Image.fromarray(image)

        if self.as_rgb:
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
    
        if self.split_type == "train":
            return image, target, index
        else:
            return image, target

    def __len__(self):
        return len(self.targets)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


#### adopted from https://github.com/bhanML/Co-teaching/blob/master/loss.py ####
def loss_coteaching(y_1, y_2, t, criterion, forget_rate, ind, device, corrupt_indices = None, uncorrupt_indices=None):
   
    loss_1 = criterion(y_1, t)
    ind_1_sorted = np.argsort(loss_1.data.cpu()).to(device)
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = criterion(y_2, t)
    ind_2_sorted = np.argsort(loss_2.data.cpu()).to(device)
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))


    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]

  
    # exchange
    loss_1_update = criterion(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = criterion(y_2[ind_1_update], t[ind_1_update])

    # list of indexes of examples guessed as uncorrupted
    guessed_uncorrupted = ind[ind_1_update]

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, guessed_uncorrupted, loss_1




def loss_coteaching_select_with_VOG_mixed(y_1, y_2, t, criterion,vogs1, vogs2, forget_rate, ind, device, args, epoch,corrupt_indices = None, uncorrupt_indices = None):

    if epoch<= args.num_gradual:

        loss_1 = criterion(y_1, t)
        ind_1_sorted = np.argsort(loss_1.data.cpu()).to(device)
        loss_1_sorted = loss_1[ind_1_sorted]

        loss_2 = criterion(y_2, t)
        ind_2_sorted = np.argsort(loss_2.data.cpu()).to(device)
        loss_2_sorted = loss_2[ind_2_sorted]

        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(loss_1_sorted))

        ind_1_update=ind_1_sorted[:num_remember]
        ind_2_update=ind_2_sorted[:num_remember]

        # exchange
        loss_1_update = criterion(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = criterion(y_2[ind_1_update], t[ind_1_update])

        # list of indexes of examples guessed as uncorrupted
        guessed_uncorrupted = ind[ind_1_update]

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, guessed_uncorrupted, loss_1
        
    else:

        vogs1 = vogs1[ind]
        vogs2 = vogs2[ind]

        ind_1_sorted = np.argsort(vogs1)
        ind_2_sorted = np.argsort(vogs2)

        loss_1 = criterion(y_1, t)
        ind_1_sorted_from_loss = np.argsort(loss_1.data.cpu())
        
        loss_2 = criterion(y_2, t)
        ind_2_sorted_from_loss = np.argsort(loss_2.data.cpu())

        vogs_1_sorted = vogs1[ind_1_sorted]
        remember_rate = 1 - forget_rate
        num_remember = int(remember_rate * len(vogs_1_sorted))

        ######### mixing strategy to select from both loss and VOG ##########################
        ind_1_update_from_loss=ind_1_sorted_from_loss[:int(num_remember*(1-args.mix_ratio))]
        ind_2_update_from_loss=ind_2_sorted_from_loss[:int(num_remember*(1-args.mix_ratio))]


        ind_1_update_from_VOG = [x for x in ind_1_sorted if x not in ind_1_update_from_loss][:int(num_remember*args.mix_ratio)]
        ind_2_update_from_VOG = [x for x in ind_2_sorted if x not in ind_2_update_from_loss][:int(num_remember*args.mix_ratio)]

        ind_1_update = ind_1_update_from_loss.numpy().tolist() + ind_1_update_from_VOG
        ind_2_update = ind_2_update_from_loss.numpy().tolist() + ind_2_update_from_VOG
        ######### mixing strategy to select from both loss and VOG ##########################

        # exchange
        loss_1_update = criterion(y_1[ind_2_update], t[ind_2_update])
        loss_2_update = criterion(y_2[ind_1_update], t[ind_1_update])

        # list of indexes of examples guessed as uncorrupted
        guessed_uncorrupted = ind[ind_1_update]

        return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember,guessed_uncorrupted, loss_1


def safe_load_dict(model, new_model_state):
    """
    Safe loading of previous ckpt file.
    """
    old_model_state = model.state_dict()
    c = 0
    for name, param in new_model_state.items():
        n = name.split('.')
        beg = n[0]
        end = n[1:]
        if beg == 'model':
            name = '.'.join(end)
        if name not in old_model_state:
            print('%s not found in old model.' % name)
            continue
        c += 1
        if old_model_state[name].shape != param.shape:
            print('Shape mismatch...ignoring %s' % name)
            continue
        else:
            old_model_state[name].copy_(param)
    if c == 0:
        raise AssertionError('No previous ckpt names matched and the ckpt was not loaded properly.')


def load_model (net, device, ckpt_file = None, model_name = None):
    if ckpt_file is not None:
            resumed = torch.load(ckpt_file,map_location=device)

            if model_name == "moco":
                state_dict = resumed["state_dict"]
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith("module.encoder_q") and not k.startswith(
                        "module.encoder_q.fc"
                    ):
                        # remove prefix
                        state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]
                print("Resuming from {}".format(ckpt_file))
                safe_load_dict(net,state_dict)

            elif model_name == "bigbigan":
                    #state_dict = resumed["state_dict"]
                    state_dict = resumed
                    for k in list(state_dict.keys()):
                        # retain only encoder_q up to before the embedding layer
                        if k.startswith("encoder.model") and not k.startswith("encoder.model.fc"):
                            print (k)
                            # remove prefix
                            state_dict[k[len("encoder.model.") :]] = state_dict[k]
                        # delete renamed or unused k
                        del state_dict[k]
                    print("Resuming from {}".format(ckpt_file))
                    safe_load_dict(net,state_dict)

            elif model_name == "vae":
                new_keys = list(k for k,v in net.state_dict().items())[:-2]

                #Function to replace keys in the OrderedDict
                def replace_keys(d, new_keys_list):
                    if isinstance(d, dict):
                        new_dict = OrderedDict()
                        for old_key, value in d.items():
                            if new_keys_list:
                                new_key = new_keys_list.pop(0)
                                new_dict[new_key] = value
                            else:
                                new_dict[old_key] = value
                        return new_dict
                    else:
                        return d

                resumed = replace_keys(resumed["state_dict"], new_keys)
                print("Resuming from {}".format(ckpt_file))
                safe_load_dict(net, resumed)

            else:
        
                if 'state_dict' in resumed:
                    state_dict_key = 'state_dict'
                    print("Resuming from {}".format(ckpt_file))
                    safe_load_dict(net, resumed[state_dict_key])
                else:
                    print("Resuming from {}".format(ckpt_file))
                    safe_load_dict(net, resumed)
    return net




