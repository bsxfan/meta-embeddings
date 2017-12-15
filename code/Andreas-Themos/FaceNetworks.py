import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import copy
import math
import numpy
import os
import torchvision.datasets as dset
import logging
from torch.utils.data import DataLoader
from torchvision import transforms
from MetaEmbeddings import GaussianMetaEmbedding
from utils_test import test_model
from Config import ConfigNetwork, ConfigFaceDatasets
import random
import pyro
import pyro.distributions as dist


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Nike Brümmer, Adrian Bulat"]


"""
# TODO: enforce a clearly separated interface class, ideally sth like
import abc

class BaseNetwork(metaclass=abc.ABCMeta,nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self.__init_layers__()
    
    @abc.abstractmethod
    sef __init_layers__()
    
class SiameseNetwork(BaseNetwork)
class ResNetBase(BaseNetwork) # abstract/interface
class ResNetSiameseNetwork(ResNetBase)
"""


class SiameseNetwork(nn.Module):
    def __init__(self, embedding_size=ConfigNetwork.embedding_size, block=ConfigNetwork.basic_block, normalize=ConfigNetwork.normalize_embedding):
        super(SiameseNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.normalize = normalize
        self.inplanes = 64
        layers = [2, 2, 2, 2]
        """
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        """
        init_res_net = ConfigNetwork.init_resnet
        self.conv1 = copy.deepcopy(init_res_net.conv1)
        self.bn1 = copy.deepcopy(init_res_net.bn1)
        self.relu = copy.deepcopy(init_res_net.relu)
        self.maxpool = copy.deepcopy(init_res_net.maxpool)
        self.layer1 = copy.deepcopy(init_res_net.layer1)
        self.layer2 = copy.deepcopy(init_res_net.layer2)
        self.layer3 = copy.deepcopy(init_res_net.layer3)
        self.layer4 = copy.deepcopy(init_res_net.layer4)
        # dropping the layers: avgpool & fc, having an own fc layer instead
        self.num_fc_input = init_res_net.fc._parameters['weight'].data.cpu().numpy().shape[1]
        if ConfigNetwork.dropouts:
            self.dropout = nn.Dropout()
        self.fc = nn.Linear(self.num_fc_input * 4 * 4, self.embedding_size)
        self.fc.bias.data.zero_()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.register_buffer('embeddings_mean', torch.zeros(self.embedding_size))
        if ConfigNetwork.embeddings_mean_file is not None:
            if os.path.exists(ConfigNetwork.embeddings_mean_file):
                self.register_buffer('embeddings_mean', torch.from_numpy(numpy.load(ConfigNetwork.embeddings_mean_file)).type(torch.FloatTensor))
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    def set_ResNet_requires_grad(self, requires_grad):
        self.conv1.requires_grad = requires_grad
        self.bn1.requires_grad = requires_grad
        # self.relu.parameters().requires_grad = requires_grad
        self.maxpool.requires_grad = requires_grad
        self.layer1.requires_grad = requires_grad
        if ConfigNetwork.freere_ResNet_layer_depth > 1:
            self.layer2.requires_grad = requires_grad
        if ConfigNetwork.freere_ResNet_layer_depth > 2:
            self.layer3.requires_grad = requires_grad
        if ConfigNetwork.freere_ResNet_layer_depth > 3:
            self.layer4.requires_grad = requires_grad
    def forward_once_ResNet(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        if ConfigNetwork.dropouts:
            x = self.dropout(x)
        return x
    def forward_once(self, x):
        x = self.forward_once_ResNet(x)
        x = self.fc(x)
        if self.normalize:
            x = x - Variable(self.embeddings_mean)
            # logging.debug('x size: {}'.format(x.size()))
            xn = torch.norm(x, p=2, dim=1).detach().view(-1,1).expand_as(x)
            # logging.debug('xn size: {}'.format(xn.size()))
            x = x / xn
        # if self.scale_up:
        #     x = x.mul(self.c.expand_as(x))
        return x
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
    @staticmethod
    def distance_pairwise_euclidean(output1, output2):
        return F.pairwise_distance(output1, output2)
    @staticmethod
    def net_distance(data, net, epoch):
        output1, output2, label = data
        distance = SiameseNetwork.distance_pairwise_euclidean(output1, output2)
        return distance, label
    @staticmethod
    def train_epoch(train_dataloader, net, optimizer, epoch, iteration_number):
        criterion = ContrastiveLoss()
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
            output1,output2 = net(img0,img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1,output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i % ConfigNetwork.iteration_step == 0 :
                logging.info("Epoch number {},\t  iteration {},\t Current loss {}".format(epoch, i, loss_contrastive.data[0]))
                iteration_number += ConfigNetwork.iteration_step
                test_model(database_dir=(output1, output2, label),
                           net=net,
                           net_distance=SiameseNetwork.net_distance,
                           epoch=epoch)


class SoftMaxNetwork(SiameseNetwork):
    def __init__(self, num_train_classes, embedding_size=ConfigNetwork.embedding_size, block=ConfigNetwork.basic_block):
        super(SoftMaxNetwork, self).__init__(embedding_size=embedding_size, block=block)
        self.fc_softmax = nn.Linear(self.embedding_size, num_train_classes)
    def forward_once(self, x):
        x = super(SoftMaxNetwork, self).forward_once(x)
        if self.training:
            x = self.fc_softmax(x)
        return x
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        return output1
    def to_siameseNetwork(self):
        net = SiameseNetwork(embedding_size=self.embedding_size)
        net.conv1 = copy.deepcopy(self.conv1)
        net.bn1 = copy.deepcopy(self.bn1)
        net.relu = copy.deepcopy(self.relu)
        net.maxpool = copy.deepcopy(self.maxpool)
        net.layer1 = copy.deepcopy(self.layer1)
        net.layer2 = copy.deepcopy(self.layer2)
        net.layer3 = copy.deepcopy(self.layer3)
        net.layer4 = copy.deepcopy(self.layer4)
        # dropping the layers: avgpool & fc, having an own fc layer instead
        if ConfigNetwork.dropouts:
            net.dropout = copy.deepcopy(self.dropout)
        net.fc = copy.deepcopy(self.fc)
        return net
    @staticmethod
    def train_epoch(train_dataloader, net, optimizer, epoch, iteration_number):
        criterion = CrossEntropyLoss()
        output2 = None
        last_label = None
        for i, data in enumerate(train_dataloader,0):
            img0, label = data
            lidx = copy.deepcopy(label.numpy())
            img0, label = Variable(img0).cuda(), Variable(label).cuda()
            output1 = net(img0, img0)
            optimizer.zero_grad()
            loss_ce = criterion(output1,label)
            loss_ce.backward()
            optimizer.step()
            if i % ConfigNetwork.iteration_step == 0 :
                logging.info("Epoch number {},\t  iteration: {},\t Current loss {}".format(epoch, i, loss_ce.data[0]))
                iteration_number += ConfigNetwork.iteration_step
                # track euclidean distance


class GME_SiameseNetwork(SiameseNetwork):
    def __init__(self,
                 embedding_size=ConfigNetwork.embedding_size,
                 expected_mu=torch.zeros(ConfigNetwork.embedding_size),
                 expected_B=torch.eye(ConfigNetwork.embedding_size).diag(),
                 Bsize=ConfigNetwork.precision_size,
                 pretrained_siamese_net=ConfigNetwork.pretrained_siamese_net,
                 normalize=ConfigNetwork.normalize_embedding,
                 block=ConfigNetwork.basic_block,
                 ntc=None):
        super(GME_SiameseNetwork, self).__init__(embedding_size=embedding_size, normalize=normalize, block=block)
        if pretrained_siamese_net is not None:
            if os.path.exists(pretrained_siamese_net):
                if ConfigNetwork.train_with_softmax & ntc is not None: # ntc is misused as a 'hidden parameter'
                    init_net = SoftMaxNetwork(num_train_classes=ntc, embedding_size=embedding_size, block=block)
                else:
                    init_net = SiameseNetwork(embedding_size=embedding_size, normalize=normalize, block=block)
                init_net.load_state_dict(torch.load(pretrained_siamese_net))
                self.conv1 = copy.deepcopy(init_net.conv1)
                self.bn1 = copy.deepcopy(init_net.bn1)
                self.relu = copy.deepcopy(init_net.relu)
                self.maxpool = copy.deepcopy(init_net.maxpool)
                self.layer1 = copy.deepcopy(init_net.layer1)
                self.layer2 = copy.deepcopy(init_net.layer2)
                self.layer3 = copy.deepcopy(init_net.layer3)
                self.layer4 = copy.deepcopy(init_net.layer4)
                if ConfigNetwork.dropouts:
                    self.dropout = copy.deepcopy(init_net.dropout)
                self.fc = copy.deepcopy(init_net.fc)
                logging.debug('loaded pre-trained: {}'.format(pretrained_siamese_net))
            else:
                logging.warning('file not exists: {}'.format(pretrained_siamese_net))
        # self.fc_mu = nn.Linear(embedding_size, embedding_size)
        self.register_buffer('I',torch.FloatTensor(self.embedding_size).fill_(1.0))
        self.xscale = nn.Parameter(torch.FloatTensor([self.embedding_size]))
        self.bscale = nn.Parameter(torch.FloatTensor([1 / self.embedding_size]))
        self.Bsize = Bsize
        self.e_mu = nn.Parameter(expected_mu)
        self.e_B = nn.Parameter(expected_B)
        # self.e_B = nn.Parameter(torch.eye(self.embedding_size))
        self.sigmoid = nn.Sigmoid().cuda()
        self.softplus = nn.Softplus().cuda()
        """
        if self.Bsize > 0:
            # stdv = 1. / math.sqrt(self.Bsize)
            # self.Bscale = Variable(torch.ones(1)).cuda()
            # stdv = .1 / math.sqrt(self.Bsize)
            # self.fc_phi = nn.Linear(self.num_fc_input * 4 * 4, self.Bsize)
            self.fc_phi = nn.Linear(embedding_size, self.Bsize)
            # self.fc_phi.bias.data.fill_(0) # init B estimate should be 1, i.e. the original softmax
            # self.fc_phi.weight.data.uniform_(-stdv, stdv)
        else:
            # PLDA case: constant embedding precision
            self.fc_phi = Variable(self.I)
        """
        self.fc_phi = nn.Linear(embedding_size, embedding_size)
        self.fc_phi.bias.data.fill_(0)
        self.fc_phi.weight.data /= self.embedding_size ** 0.5
        self.last_mu = torch.zeros(self.embedding_size)
        self.last_std = torch.ones(self.embedding_size)
    def set_ResNet_requires_grad(self, requires_grad):
        super(GME_SiameseNetwork, self).set_ResNet_requires_grad(requires_grad)
        if ConfigNetwork.freere_ResNet_layer_depth > 4:
            self.fc.requires_grad = requires_grad
            self.e_mu.requires_grad = requires_grad
            self.e_B.requires_grad = requires_grad
    def forward_embeddings(self, x):
        x = super(GME_SiameseNetwork, self).forward_once(x)
        mu = x - self.e_mu.expand_as(x) # * self.xscale.expand_as(x)

        phi = self.fc_phi(x)
        # B = self.bscale.expand_as(x) * self.e_B
        B = self.sigmoid(phi.mean(1)).expand(x.size(1),x.size(0)).t() * self.e_B * 2 # e_B is vector
        # B = self.sigmoid(phi.mean(1)).expand(x.size(1),x.size(0)).t() @ self.e_B * 2 # e_B is matrix

        # B = self.e_B.expand_as(x)
        # mu = mu * self.sigmoid(phi.mean(1)).expand(x.size(1),x.size(0)).t() * 2

        if self.training:
            order = 4
            random_rescale = Variable(torch.exp(- torch.randn(x.size(0)) * order)).cuda()
            B = random_rescale.unsqueeze(1) * B

        return mu, B
    def forward_once(self, x):
        mu, B = self.forward_embeddings(x)
        a = B * mu
        return a, B
    def forward(self, input1, input2):
        output1_a, output1_B = self.forward_once(input1)
        output2_a, output2_B = self.forward_once(input2)
        return output1_a, output1_B, output2_a, output2_B


class GME_SoftmaxNetwork(GME_SiameseNetwork):
    def __init__(self, num_train_classes,
                 embedding_size=ConfigNetwork.embedding_size,
                 expected_mu=torch.zeros(ConfigNetwork.embedding_size),
                 expected_B=torch.eye(ConfigNetwork.embedding_size).diag(),
                 Bsize=ConfigNetwork.precision_size,
                 pretrained_siamese_net=ConfigNetwork.pretrained_siamese_net,
                 normalize=ConfigNetwork.normalize_embedding,
                 block=ConfigNetwork.basic_block):
        super(GME_SoftmaxNetwork, self).__init__(embedding_size=ConfigNetwork.embedding_size,
                                                 Bsize=ConfigNetwork.precision_size,
                                                 expected_mu=expected_mu,
                                                 expected_B=expected_B,
                                                 pretrained_siamese_net=pretrained_siamese_net,
                                                 normalize=ConfigNetwork.normalize_embedding,
                                                 block=ConfigNetwork.basic_block,
                                                 ntc=num_train_classes)
        # self.fc_softmax =  nn.Linear(self.embedding_size, num_train_classes)
        self.num_train_classes = num_train_classes
        self.pretrained_net = pretrained_siamese_net
        if os.path.exists(self.pretrained_net):
            logging.debug('using softmax weights')
            init_net = SoftMaxNetwork(num_train_classes=num_train_classes, embedding_size=embedding_size, block=block)
            # self.fc_softmax = copy.deepcopy(init_net.fc_softmax.weight)
            self.register_buffer('fc_softmax', copy.deepcopy(init_net.fc_softmax.weight.data))
        else:
            self.register_buffer('fc_softmax', torch.randn(self.num_train_classes, self.embedding_size))
            logging.debug('softmax: {}'.format(self.fc_softmax.cpu().numpy()))
            fcsn = torch.norm(self.fc_softmax, p=2, dim=1).view(-1, 1).expand_as(self.fc_softmax)
            self.fc_softmax = (self.fc_softmax / fcsn) * (self.embedding_size ** 0.5)
            logging.debug('softmax normed: {}'.format(self.fc_softmax.cpu().numpy()))
    def to_softmaxNetwork(self):
        net = SoftMaxNetwork(num_train_classes=self.num_train_classes, embedding_size=self.embedding_size)
        net.conv1 = copy.deepcopy(self.conv1)
        net.bn1 = copy.deepcopy(self.bn1)
        net.relu = copy.deepcopy(self.relu)
        net.maxpool = copy.deepcopy(self.maxpool)
        net.layer1 = copy.deepcopy(self.layer1)
        net.layer2 = copy.deepcopy(self.layer2)
        net.layer3 = copy.deepcopy(self.layer3)
        net.layer4 = copy.deepcopy(self.layer4)
        # dropping the layers: avgpool & fc, having an own fc layer instead
        if ConfigNetwork.dropouts:
            net.dropout = copy.deepcopy(self.dropout)
        net.fc = copy.deepcopy(self.fc)
        if self.pretrained_net is not None:
            net.fc_softmax = nn.Linear(self.num_train_classes, self.embedding_size)
            state_dict = torch.load(self.pretrained_net, map_location=lambda storage, loc: storage)
            net.fc_softmax.bias = nn.Parameter(state_dict['fc_softmax.bias'])
            net.fc_softmax.weight = nn.Parameter(state_dict['fc_softmax.weight'])
        else:
            net.fc_softmax = copy.deepcopy(self.fc_softmax)
        return net
    def forward_once(self, x):
        mu, B = self.forward_embeddings(x)
        a = B * mu
        if self.training:
            """
            Mahalanobis distance: sqrt( (x-m)' S^-1 (x-m) )
            => as distance score
            """
            # B is diagonal
            """
            gamma = Variable(self.I).expand_as(B) + B
            latent = mu # (1 / gamma) * a
    
            softmax_expanded = Variable(self.fc_softmax).expand(gamma.size(0), *self.fc_softmax.size()).transpose(0,1)
            # softmax = self.fc_softmax(x)
            # softmax_expanded = softmax.expand(gamma.size(0), *softmax.size()).transpose(0,1)
            chol_gamma = torch.sqrt(gamma).expand_as(softmax_expanded)
            w = softmax_expanded - latent.expand(self.num_train_classes, *latent.size())
            z = chol_gamma * w
            x = torch.sqrt((z ** 2).sum(2))
            # logdet_gamma = torch.sum(torch.log(chol_gamma), dim=2) * 2
            # x = (x - logdet_gamma) / 2
            """

            """
            neg_llr = Variable(torch.zeros(a.size(0), self.num_train_classes))
            a2 = Variable(self.fc_softmax)
            B2 = Variable(self.e_B.data).expand(self.num_train_classes, self.embedding_size) * 5
            for batch_idx in range(a.size(0)):
                a1 = a[batch_idx].expand(self.num_train_classes, self.embedding_size)
                B1 = B[batch_idx].expand(self.num_train_classes, self.embedding_size)
                neg_llr[batch_idx] = GaussianMetaEmbedding.distance_neg_llr(a1, B1, a2, B2)
            return neg_llr
            """

            """
            gme_batch = GaussianMetaEmbedding(a, B).cuda()
            gme_softmax = GaussianMetaEmbedding(Variable(self.fc_softmax), Variable(self.e_B.data).expand(self.num_train_classes, self.embedding_size) * 5).cuda()
            logden_batch = gme_batch.log_expectation().expand(self.num_train_classes, a.size(0))
            logden_softmax = gme_softmax.log_expectation().expand(self.num_train_classes, a.size(0)).expand(a.size(0),self.fc_softmax.size(0)).t()

            # pooled log expectation
            apooled = gme_batch.a.expand(self.fc_softmax.size(0),*a.size()) + gme_softmax.a.expand(a.size(0),*self.fc_softmax.size()).transpose(1,0)
            Bpooled = gme_batch.B.expand(self.fc_softmax.size(0),*a.size()) + gme_softmax.B.expand(a.size(0),*self.fc_softmax.size()).transpose(1,0)
            chol_BI = torch.sqrt(Variable(gme_softmax.prior_precision).expand(self.fc_softmax.size(0),a.size(0)).cuda() + Bpooled)
            logdet_BI = torch.sum(torch.log(chol_BI), dim=2) * 2
            z = torch.reciprocal(chol_BI) * apooled
            lognum = ((z * z).sum(dim=2) - logdet_BI) / 2

            return -(lognum - logden_batch - logden_softmax)
            """

            return GaussianMetaEmbedding.llr_identification(a, B, Variable(self.fc_softmax), Variable(self.e_B.data).expand(self.num_train_classes, self.embedding_size))

            # return GaussianMetaEmbedding(a,B).log_expectation()
        else:
            #"""
            logging.debug('batch info')
            if self.Bsize > 0 or ConfigNetwork.freere_ResNet_layer_depth > 4:
                logging.debug('phi_bias: {} to {}'.format(
                    self.fc_phi.bias.data.cpu().numpy().min(),
                    self.fc_phi.bias.data.cpu().numpy().max()))
                logging.debug('phi_weights: {} to {}'.format(
                    self.fc_phi.weight.data.cpu().numpy().min(),
                    self.fc_phi.weight.data.cpu().numpy().max()))
            else:
                logging.debug('phi = 1')
            logging.debug('mu: {} to {}, means (0) std: {}, means (1) ex. {} with std: {}'.format(
                    mu.data.cpu().numpy().min(),
                    mu.data.cpu().numpy().max(),
                    mu.data.cpu().numpy().mean(axis=0).std(),
                    mu.data.cpu().numpy().mean(axis=1)[:3],
                    mu.data.cpu().numpy().mean(axis=1).std()))
            logging.debug('a: {} to {}, means (0) std: {}, means (1) ex. {} with std: {}'.format(
                    a.data.cpu().numpy().min(),
                    a.data.cpu().numpy().max(),
                    a.data.cpu().numpy().mean(axis=0).std(),
                    a.data.cpu().numpy().mean(axis=1)[:3],
                    a.data.cpu().numpy().mean(axis=1).std()))
            logging.debug('E[mu]: {} to {}, mean : {}, ex. {} with std: {}'.format(
                    self.e_mu.data.cpu().numpy().min(),
                    self.e_mu.data.cpu().numpy().max(),
                    self.e_mu.data.cpu().numpy().mean(),
                    self.e_mu.data.cpu().numpy()[:3],
                    self.e_mu.data.cpu().numpy().std()))
            logging.debug('E[B]: {} to {}, mean : {}, ex. {} with std: {}'.format(
                    self.e_B.data.cpu().numpy().min(),
                    self.e_B.data.cpu().numpy().max(),
                    self.e_B.data.cpu().numpy().mean(),
                    self.e_B.data.cpu().numpy()[:3],
                    self.e_B.data.cpu().numpy().std()))
            if self.Bsize > 0 or ConfigNetwork.freere_ResNet_layer_depth > 4:
                logging.debug('B: {} to {}, means (0) std: {}, means (1) ex. {} with std: {}'.format(
                        B.data.cpu().numpy().min(),
                        B.data.cpu().numpy().max(),
                        B.data.cpu().numpy().mean(axis=0).std(),
                        B.data.cpu().numpy().mean(axis=1)[:3],
                        B.data.cpu().numpy().mean(axis=1).std()))
            else:
                logging.debug('B: flat: {}'.format(B.data.cpu().numpy()[0,0]))
            #"""
            return a, B
    def forward(self, input1, input2):
        output1 = self.forward_once(input1)

        # requires input2 to be labels
        if self.training:
            classes = torch.from_numpy(numpy.unique(input2.data.cpu().numpy())).type(torch.LongTensor).cuda()
            sm = Variable(torch.zeros(output1.size(0), self.num_train_classes)).cuda()
            for c in classes:
                idx = (c == input2)
                nsamples = int(idx.sum())
                if nsamples:
                    # sm[idx, c] = - output1[idx].sum().expand(nsamples)
                    # sm[idx, c] = -output1[idx]
                    sm[idx, c] = output1[idx, c]
            output1 = sm #.expand(output1.size(0), self.fc_softmax.size(0))
        return output1
    @staticmethod
    def train_epoch(train_dataloader, net, optimizer, epoch, iteration_number):
        criterion = CrossEntropyLoss()
        output2 = None
        last_label = None
        for i, data in enumerate(train_dataloader,0):
            img0, label = data
            lidx = copy.deepcopy(label.numpy())
            img0, label = Variable(img0).cuda(), Variable(label).cuda()
            output1 = net(img0, label)
            optimizer.zero_grad()
            loss_ce = criterion(output1,label)
            loss_ce.backward()
            optimizer.step()
            if i % ConfigNetwork.iteration_step == 0 :
                logging.info("Epoch number {},\t  iteration: {},\t Current loss {}".format(epoch, i, loss_ce.data[0]))
                iteration_number += ConfigNetwork.iteration_step
                # track euclidean distance


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0 * ConfigNetwork.embedding_size**0.5 if ConfigNetwork.normalize_embedding else 25.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = SiameseNetwork.distance_pairwise_euclidean(output1, output2)
        loss_contrastive = torch.mean(
            (1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

""" roadwork ahead 
class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, input_size=100**2, fudge=ConfigNetwork.vae_fudge):
        super(Decoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, input_size)
        # setup the non-linearity
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.fudge = fudge

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        # fixing numerical instabilities of sigmoid with a fudge
        mu_img = (self.sigmoid(self.fc21(hidden))+self.fudge) * (1-2*self.fudge)
        return mu_img


class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        # setup the three linear transformations used
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearity
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.view(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_mu = self.fc21(hidden)
        z_sigma = torch.exp(self.fc22(hidden))
        return z_mu, z_sigma


class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, hidden_dim=400, use_cuda=False):
        super(VAE, self).__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim, hidden_dim)
        self.decoder = Decoder(z_dim, hidden_dim)

        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim

    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)

        # setup hyperparameters for prior p(z)
        # the type_as ensures we get CUDA Tensors if x is on gpu
        z_mu = ng_zeros([x.size(0), self.z_dim], type_as=x.data)
        z_sigma = ng_ones([x.size(0), self.z_dim], type_as=x.data)
        # sample from prior
        # (value will be sampled by guide when computing the ELBO)
        z = pyro.sample("latent", dist.normal, z_mu, z_sigma)

        # decode the latent code z
        mu_img = self.decoder(z)
        # score against actual images
        pyro.observe("obs", dist.bernoulli, x.view(-1, 784), mu_img)

    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        # use the encoder to get the parameters used to define q(z|x)
        z_mu, z_sigma = self.encoder(x)
        # sample the latent code z
        pyro.sample("latent", dist.normal, z_mu, z_sigma)

    def set_ResNet_requires_grad(self, requires_grad):
        pass

    def P(self):
        pass

    def forward_once(self, x):
        # TODO x = self.forward_once_ResNet(x)
        x = self.fc(x)
        if self.normalize:
            x = x - Variable(self.embeddings_mean)
            xn = torch.norm(x, p=2, dim=1).detach().view(-1,1).expand_as(x)
            x = x / xn
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

    # network specific functions for the framework
    @staticmethod
    def net_distance(data, net, epoch):
        output1, output2, label = data
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        return distance, label

    @staticmethod
    def train_epoch(train_dataloader, net, optimizer, epoch, iteration_number):
        criterion = torch.nn.CrossEntropyLoss()
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label = data
            output1, output2 = net(img0, img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()

            for p in params:
                if p.grad is not None:
                    data = p.grad.data
                    p.grad = Variable(data.new().resize_as_(data).zero_())
            if i % ConfigNetwork.iteration_step == 0 :
                logging.info("Epoch number {},\t  iteration {},\t Current loss {}".format(epoch, i, loss_contrastive.data[0]))
                iteration_number += ConfigNetwork.iteration_step
"""
