import torch, warnings
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False, CE=True, OPL=False, gamma=0.5, opl_weight=1.0):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.all_out = all_outputs
        self.CE = CE
        self.OPL = OPL
        self.gamma = gamma
        self.opl_weight = opl_weight
        self.means = []
        self.covs = []
        self.class_labels = []

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--all-outputs', action='store_true', required=False,
                            help='Allow all weights related to all outputs to be modified (default=%(default)s)')
        parser.add_argument('--CE', action='store_false', required=False,
                            help='CE loss (default=%(default)s)')
        parser.add_argument('--OPL', action='store_true', required=False,
                            help='OPL loss (default=%(default)s)')
        parser.add_argument('--gamma', default=0.5, type=float, required=False,
                        help='Gamma for neg pair in OPL (default=%(default)s)')
        parser.add_argument('--opl_weight', default=1, type=float, required=False,
                        help='Weight for OPL loss (default=%(default)s)')
    
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1 and not self.all_out:
            # if there are no exemplars, previous heads are not modified
            params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        else:
            params = self.model.parameters()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def save_protype(self, trained_model, loader):
        trained_model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, targets in loader:
                output, feature = trained_model(images.to(self.device), return_features=True)
                labels.append(targets.numpy())
                features.append(feature.cpu().numpy())
        labels = np.hstack(labels)
        labels_set = np.unique(labels)
        features = np.concatenate(features, 0)
        feature_dim = features.shape[1]

        for item in labels_set:
            index = np.where(item==labels)[0]
            feature_classwise = features[index]
            self.class_labels.append(item)
            self.means.append(torch.from_numpy(np.mean(feature_classwise, axis=0)))
            self.covs.append(torch.from_numpy(np.cov(feature_classwise.T)))

    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if t == 0:
            # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
            # negative values"
            if self.model.model.__class__.__name__ == 'ResNet':
                old_block = self.model.model.layer3[-1]
                self.model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                               old_block.conv2, old_block.bn2, old_block.downsample)
            elif self.mode.model.__class__.name__ == 'SmallCNN':
                self.model.model.last_relu = False
            else:
                warnings.warn("Warning: ReLU not removed from last block.")
        super().pre_train_process(t, trn_loader)

    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward current model
            if not self.OPL:
                features = None
                outputs = self.model(images.to(self.device))
            else:
                outputs, features = self.model(images.to(self.device), return_features=True)
            loss = self.criterion(t, outputs, targets.to(self.device), features)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()

    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # FINETUNING TRAINING -- contains the epochs loop
        super().train_loop(t, trn_loader, val_loader)

        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

        self.save_protype(self.model, trn_loader)
    
    def classify(self, task, features, targets):
        # expand means to all batch images                   # bs*256*num_classes
        means = torch.stack(self.means)
        means = torch.stack([means]*features.shape[0])
        means = means.transpose(1,2)    
        # expand all features to all classes
        features = features.unsqueeze(2)
        features = features.expand_as(means)
        # get cosine-similarities for all images to all prototypes
        # note: features and means do not need normalize 
        cos_sim = torch.nn.functional.cosine_similarity(features, means.to(self.device), dim=1, eps=1e-08)   # bs*num_classes
        pred = cos_sim.argmax(1)
        hits_tag = (pred == targets.to(self.device)).float()
        return hits_tag, hits_tag 

    def eval_ncm(self, t, val_loader):
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward old model
                old_features = None
                if t > 0:
                    old_outputs, old_features = self.model_old(images.to(self.device), return_features=True)
                # Forward current model
                outputs, feats = self.model(images.to(self.device), return_features=True)
                loss = self.criterion(t, outputs, targets.to(self.device), feats)
                # during training, the usual accuracy is not computed
                if t > len(self.means)-1:
                    print('No means created yet!')
                    hits_taw, hits_tag = torch.zeros(targets.shape[0]).float(), torch.zeros(targets.shape[0]).float()
                else:
                    hits_taw, hits_tag = self.classify(t, feats, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def eval(self, t, val_loader):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, targets in val_loader:
                # Forward current model
                if self.OPL:
                    outputs, features = self.model(images.to(self.device), return_features=True)
                else:
                    outputs = self.model(images.to(self.device))
                    features = None
                loss = self.criterion(t, outputs, targets.to(self.device), features)
                hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def criterion(self, t, outputs, targets, features=None):
        """Returns the loss value"""
        if self.all_out or len(self.exemplars_dataset) > 0:
            if self.CE and not self.OPL:
                return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
            if self.CE and self.OPl:
                return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets) + self.opl_weight*OrthogonalProjectionLoss(self.gamma)(features, targets, normalize=True)
            if not self.CE and self.OPL:
                return OrthogonalProjectionLoss(self.gamma)(features, targets, normalize=True)
        else:
            if self.CE and not self.OPL:
                return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
            if self.CE and self.OPL:
                return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t]) + self.opl_weight*OrthogonalProjectionLoss(self.gamma)(features, targets - self.model.task_offset[t], normalize=True)
            if not self.CE and self.OPL:
                return OrthogonalProjectionLoss(self.gamma)(features, targets, normalize=True)

class OrthogonalProjectionLoss(nn.Module):
    def __init__(self, gamma=0.5):
        super(OrthogonalProjectionLoss, self).__init__()
        self.gamma = gamma

    def forward(self, features, labels=None, normalize=True):
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        #  features are normalized
        if normalize:
            features = F.normalize(features, p=2, dim=1)

        labels = labels[:, None]  # extend dim

        mask = torch.eq(labels, labels.t()).bool().to(device)
        eye = torch.eye(mask.shape[0], mask.shape[1]).bool().to(device)

        mask_pos = mask.masked_fill(eye, 0).float()
        mask_neg = (~mask).float()
        dot_prod = torch.matmul(features, features.t())

        pos_pairs_mean = (mask_pos * dot_prod).sum() / (mask_pos.sum() + 1e-6)
        neg_pairs_mean = torch.abs(mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

        loss = (1.0 - pos_pairs_mean) + self.gamma * neg_pairs_mean

        return loss

# This class implements a ResNet Basic Block without the final ReLu in the forward
class BasicBlockNoRelu(nn.Module):
    expansion = 1

    def __init__(self, conv1, bn1, relu, conv2, bn2, downsample):
        super(BasicBlockNoRelu, self).__init__()
        self.conv1 = conv1
        self.bn1 = bn1
        self.relu = relu
        self.conv2 = conv2
        self.bn2 = bn2
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        # Removed final ReLU
        return out