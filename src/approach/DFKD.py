from copy import deepcopy
import torch, warnings
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from argparse import ArgumentParser

from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset

from torch.autograd import Variable
import torchvision.utils as vutils


class Appr(Inc_Learning_Appr):
    """Class implementing the finetuning baseline"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False, eval_on_train=False,
                 logger=None, exemplars_dataset=None, all_outputs=False, CE=True, OPL=True, gamma=0.5, opl_weight=1.0, last_relu=True):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, logger,
                                   exemplars_dataset)
        self.means = []
        self.covs = []
        self.class_labels = []
        self.all_out = all_outputs
        self.CE = CE     # use CE loss or NCM loss
        self.OPL = OPL
        self.gamma = gamma
        self.opl_weight = opl_weight
        self.last_relu = last_relu

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
        parser.add_argument('--opl_weight', default=1., type=float, required=False,
                        help='Weight for OPL loss (default=%(default)s)')
        parser.add_argument('--last_relu', action='store_false', required=False,
                        help='Turn on relu on feature layer? (default=%(default)s)')
        return parser.parse_known_args(args)

    def _get_optimizer(self):
        """Returns the optimizer"""
        # if len(self.exemplars_dataset) == 0 and len(self.model.heads) > 1:
        #     # if there are no exemplars, previous heads are not modified
        #     params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
        # else:
        if not self.CE:
            params = list(self.model.model.parameters())
        else:
            if self.all_out:
                params = list(self.model.parameters())
            else:
                params = list(self.model.model.parameters()) + list(self.model.heads[-1].parameters())
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
        # labels = np.reshape(labels, labels.shape[0]*labels.shape[1])
        features = np.concatenate(features, 0)
        # features = np.reshape((feature, features.shape[0]*features.shape[1], features.shape[2]))
        feature_dim = features.shape[1]

        for item in labels_set:
            index = np.where(item==labels)[0]
            feature_classwise = features[index]
            self.class_labels.append(item)
            self.means.append(torch.from_numpy(np.mean(feature_classwise, axis=0)))
            self.covs.append(torch.from_numpy(np.cov(feature_classwise.T)))

    # not get each class yet
    def generate_fake_data(self, previous_model, input_size, num_iters=1000, bs=100, lr=1e-3, train_iters=1000, temp=20.0):
        print("Start generating synthetic data for old classes")
        save_path = './data/generated_data/'
        kl = torch.nn.KLDivLoss()
        for param in previous_model.paramaters():
            param.requires_grad = False
        for task_id in range(len(self.means)):
            save_path += 'class'+str(task_id)+'/'
            cnt = 0
            penultimate_dist = torch.distributions.MultivariateNormal(self.means[task_id], self.covs[task_id])
            penultimate_samples = penultimate_dist.sample((bs*num_iters,))
            generated_softmax = Variable(torch.softmax(penultimate_samples, dim=1), requires_grad=False)
            for i in range(num_iters):
                noise = torch.randn((bs, input_size[0], input_size[1], input_size[2])).cuda()
                noise = Variable(noise ,requires_grad=True)
                optimizer = torch.optim.Adam([noise], lr)
                for iter in range(train_iters):
                    optimizer.zero_grad()
                    logits = previous_model(noise)/temp
                    outputs = torch.log_softmax(logits, dim=1)
                    loss = kl(outputs, generated_softmax)   
                    loss.backward()
                    optimizer.step()
                    if iter%10==0:
                        print(iter, '/', train_iters, 'loss:', loss)
                for m in range(bs):
                    cnt += 1
                    vutils.save_image(noise[m, :, :, :].data.clone(), save_path + str(cnt) + '.jpg', normalize=True)

        for param in previous_model.paramaters():
            param.requires_grad = True
        
    def pre_train_process(self, t, trn_loader):
        """Runs before training all epochs of the task (before the train session)"""
        if not self.last_relu:
            if t == 0:
                # Sec. 4.1: "the ReLU in the penultimate layer is removed to allow the features to take both positive and
                # negative values"
                if self.model.model.__class__.__name__ == 'ResNet':
                    old_block = self.model.model.layer3[-1]
                    self.model.model.layer3[-1] = BasicBlockNoRelu(old_block.conv1, old_block.bn1, old_block.relu,
                                                                old_block.conv2, old_block.bn2, old_block.downsample)
                elif self.model.model.__class__.__name__ == 'SmallCNN':
                    self.model.model.last_relu = False
                else:
                    warnings.warn("Warning: ReLU not removed from last block.")
        super().pre_train_process(t, trn_loader)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""

        # Save old model to extract features later. This is different from the original approach, since they propose to
        #  extract the features and store them for future usage. However, when using data augmentation, it is easier to
        #  keep the model frozen and extract the features when needed.
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        self.model_old.freeze_all()


    def train_loop(self, t, trn_loader, val_loader):
        """Contains the epochs loop"""

        # during training, compute acc by normal procedure
        self.eval_type = 'normal'

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

        # GET NEW CLASS PROTOTYPES  (issue: what to do with OLD PROTOTYPES?)
        # note: rewrite Save_Prototype to allow any BS -> done
        # note: class_labels not correct -> check target from trn_loader -> done
        self.save_protype(self.model, trn_loader)
        self.eval_type = 'ncm'        


    def train_epoch(self, t, trn_loader):
        """Runs a single epoch"""
        self.model.train()
        if self.fix_bn and t > 0:
            self.model.freeze_bn()
        for images, targets in trn_loader:
            # Forward old model
            old_features = None
            if t > 0:
                old_outputs, old_features = self.model_old(images.to(self.device), return_features=True)
            # Forward current model
            outputs, feats = self.model(images.to(self.device), return_features=True)
            loss = self.criterion(t, outputs, targets.to(self.device), feats, old_features)
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()


    # argmax
    # need abs?
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
    
    def eval(self, t, val_loader):
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
                loss = self.criterion(t, outputs, targets.to(self.device), feats, old_features)
                # during training, the usual accuracy is not computed
                if self.eval_type=='normal':
                    hits_taw, hits_tag = self.calculate_metrics(outputs, targets)
                else:
                    print('eval using ncm')
                    hits_taw, hits_tag = self.classify(t, feats, targets)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num
    
    def criterion(self, t, outputs, targets, features, old_features=None):
        """Returns the loss value"""
        # if self.all_out or len(self.exemplars_dataset) > 0:
        #     return torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
        # return torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])

        if self.CE: 
            if self.all_out:
                loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), targets)
            else:
                loss = torch.nn.functional.cross_entropy(outputs[t], targets - self.model.task_offset[t])
        else: loss = 0.0
        
        features = F.normalize(features, p=2, dim=1)
        if old_features is not None:
            old_features = F.normalize(old_features, p=2, dim=1)
        
        # compute OPL loss for current classes
        # note: OPL requires number of features == number of targets
        loss += self.opl_weight*OrthogonalProjectionLoss(self.gamma)(features, targets - self.model.task_offset[t], normalize=False)
        
        # constraint OPL loss between current classes and old prototypes
        for mean in self.means:
            mean = mean.expand_as(features).detach().to(self.device)
            loss += (torch.abs(features*mean).sum(dim=1)).mean() / len(self.means)
            # loss += nn.CosineEmbeddingLoss()(features, mean, -1*torch.ones(features.shape[0]).to(self.device))
        
        # constraint old prototypes to be parallel
        if old_features is not None:
            loss += nn.CosineEmbeddingLoss()(features, old_features.detach(),
                                            torch.ones(features.shape[0]).to(self.device))
        return loss

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
        neg_pairs_mean = (mask_neg * dot_prod).sum() / (mask_neg.sum() + 1e-6)  # TODO: removed abs

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