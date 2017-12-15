import os
import torch
from torch import nn
from torch.autograd import Variable
import numpy
import copy
import logging
from utils_test import test_model

import torch.nn.functional as F

from Config import ConfigNetwork


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Nike Brümmer, Adrian Bulat"]


class GaussianMetaEmbedding(nn.Module):
    """
    z subject identity variable (in d)
    r record
    e embedding (in d)
    f meta-embedding, infinite-dimensional
    a (in d)
    b (in D, D >= 0, b_ij >= 0)
    f-representation in(d+D)
        r_j |--> e_j = (a_j, b_j)
        f_j(z) = exp(a_j'z - .5 z'B_jz)
        with B_j (in dxd), composed as: B_j = E_0 + sum_i=1^D b_ij E_i
        with {E_i}_i=0^D are fixed (in dxd), trainable (full, low-rank or diag etc.)

    for full covariance, E_i acts as an expansion with sample depending weights
    in the diagonal approximation, E_i become one-hot vectors in D=d space,
    where each one-hot element represents a basis element on the diagonal of the precision

    for R = {r_1...n}: {e_j}_j=1^n represents {f_j}_j=1^n
        f_R = prod_j=1^n f_j
        e_R = sum_j=1^n e_j
        ... e_j essentially is logarithmic

    identity element is Gaussian with zero mean, i.e.: 1(z) = 1

    prior: p(z) = N(z | 0, I) = exp(-.5 z'z) / (2pi)^(d/2)

    expectation (L1norm): // dropping '_j'
        E(a,B) = <f>
        with f(z) = exp(a'z - .5 z'Bz), see above
        defining: mu = (I + B)^-1 a
            E(a,B) = exp(.5 a' mu) / |I + B|^.5
            log E(a,B) = .5 a' mu - .5 log(|I + B|)



    :param a: in R^d
    :param B: a positve semi-definite matrix, i.e. the precision
    :param logscale: aka gConst

    meta embedding: f(z) = exp(a'z -(1/2)z'Bz)
    with z as the identity variable
    """
    def __init__(self, a, B, logscale=Variable(torch.FloatTensor([0])), diag=True):
        super(GaussianMetaEmbedding, self).__init__()
        self.a = a
        self.B = B
        self.logscale = logscale
        self.diag = diag

        if self.diag:
            self.dim = (self.B.size()[0], ConfigNetwork.embedding_size)
            self.register_buffer('prior_precision', torch.ones(ConfigNetwork.embedding_size))
        else:
            self.dim = (self.B.size()[0], ConfigNetwork.embedding_size, ConfigNetwork.embedding_size)
            self.register_buffer('prior_precision', torch.eye(ConfigNetwork.embedding_size))

    def expectation(self):
        return torch.exp(self.log_expectation())

    def llr(self, g):
        assert(isinstance(g, GaussianMetaEmbedding))
        fg = self.pool(g)
        lognum = fg.log_expectation()
        logden1 = self.log_expectation()
        logden2 = g.log_expectation()
        return lognum - logden1 - logden2

    def log_expectation(self, cuda=True):
        ## mu = cholBI\(cholBI'\a);
        ## y = (mu'*a - log_det)/2;
        if self.diag:
            prior = copy.deepcopy(self.prior_precision)
            piCon = copy.deepcopy(self.logscale)
            if cuda:
                prior = prior.cuda()
                piCon = piCon.cuda()
            chol_BI = torch.sqrt(Variable(prior).expand_as(self.B) + self.B)
            logdet_BI = torch.sum(torch.log(chol_BI), dim=1) * 2
            z = torch.reciprocal(chol_BI) * self.a
            log_e = piCon.expand_as(logdet_BI) + ((z * z).sum(dim=1) - logdet_BI) / 2
        else:
            # TODO fix before trying
            chol_BI = torch.potrf(Variable(self.prior_precision).expand_as(self.B) + self.B, upper=False)
            logdet_BI = 2 * torch.sum(torch.log(torch.diag(chol_BI)), dim=(1,2))
            z = torch.inverse(chol_BI) @ self.a
            log_e = self.logscale.expand_as(logdet_BI) + (torch.mul(z, z).sum(dim=(1,2)) - logdet_BI) / 2
        return log_e

    def scale(self, s):
        if not isinstance(s, torch.FloatTensor):
            s = torch.FloatTensor([s])
        return self.shiftlogscale(torch.log(s))

    def L1normalize(self):
        return self.shiftlogscale(-self.log_expectation())

    def shiftlogscale(self, shift):
        return GaussianMetaEmbedding(self.a, self.B, self.logscale+shift)

    def norm_square(self):
        return self.inner_product(self)

    def inner_product(self, g):
        assert(isinstance(g, GaussianMetaEmbedding))
        return self.pool(g).expectation()

    def pool(self, AE):
        assert(isinstance(AE, GaussianMetaEmbedding))
        return GaussianMetaEmbedding(self.a + AE.a, self.B + AE.B, self.logscale + AE.logscale)

    def convolve(self, AE):
        assert(isinstance(AE, GaussianMetaEmbedding))
        a1,B1,s1 = self.a, self.B, self.logscale
        a2,B2,s2 = AE.a, AE.B, AE.logscale

        # TODO this is a hot fix for now...
        # a1 = B1 * a1
        # a2 = B2 * a2

        if self.diag:
            chol12 = torch.sqrt(B1+B2)
            chol12_inv = torch.reciprocal(chol12)
            chol12_inv_mul = torch.mul(chol12_inv, chol12_inv)
        else:
            chol12 = torch.potrf(B1+B2, upper=False)
            chol12_inv = torch.inverse(chol12)

        def solve(rhs):
            if self.diag:
                return torch.mul(chol12_inv_mul, rhs)
            else:
                return torch.dot(chol12_inv.view(ConfigNetwork.embedding_size, 1).transpose(0, 1), torch.dot(chol12_inv, rhs))

        if self.diag:
            newB = torch.mul(B1, solve(B2)) # this is inv(inv(B1)+inv(B2))
            newa = torch.mul(B2, solve(a1)) + torch.mul(B1, solve(a2)) # newB * (B1\a1 + B2\a2 )
        else:
            newB = torch.dot(B1, solve(B2)) # this is inv(inv(B1)+inv(B2))
            newa = torch.dot(B2, solve(a1)) + torch.dot(B1, solve(a2)) # newB * (B1\a1 + B2\a2 )
        return GaussianMetaEmbedding(newa, newB, s1+s2)

    def distance_square(self, g):
        assert(isinstance(g, GaussianMetaEmbedding))
        return self.norm_square() + g.norm_square() - 2 * self.inner_product(g)

    def norm_square_of_sum(self, g):
        assert(isinstance(g, GaussianMetaEmbedding))
        return self.norm_square() + g.norm_square() + 2 * self.inner_product(g)

    def raise_to_themos_factor(self, e):
        return GaussianMetaEmbedding(torch.mul(self.B * self.a, e), torch.mul(self.B, e), torch.mul(self.logscale, e))

    def get_mu_cov(self):
        # mapping from natural to the (centeralized) expectational parameterization
        """
        if self.diag:
            C = torch.reciprocal(self.B)
            return torch.mul(C, self.a), C
        else:
            C = torch.inverse(self.B)
            return torch.dot(C, self.a), C
        """
        # return self.a, torch.reciprocal(self.B)
        C = torch.reciprocal(self.B)
        return torch.mul(C, self.a), C

    @staticmethod
    def llr_verification(a1, B1, a2, B2, cuda=True):
        # diagonal precision -> det(covariance) = exp(-B.sum()) ~> as 1/det: add term instead of subtra
        log_scale_pi = Variable(torch.FloatTensor([-ConfigNetwork.embedding_size / 2 * numpy.log(2 * numpy.pi)]))
        if cuda:
            """
            output1_a = output1_a.cuda()
            output2_a = output2_a.cuda()
            output1_B = output1_B.cuda()
            output2_B = output2_B.cuda()
            """
            log_scale_pi = log_scale_pi.cuda()
        gme1 = GaussianMetaEmbedding(a=a1,
                                     B=B1,
                                     logscale=log_scale_pi
                                     )
        gme2 = GaussianMetaEmbedding(a=a2,
                                     B=B2,
                                     logscale=log_scale_pi
                                     )
        if cuda:
            gme1 = gme1.cuda()
            gme2 = gme2.cuda()
        llr = gme1.llr(gme2)
        return llr

    @staticmethod
    def neg_llr_verification(a1, B1, a2, B2, cuda=True):
        return -GaussianMetaEmbedding.llr_verification(a1, B1, a2, B2, cuda)

    @staticmethod
    def llr_identification(a1, B1, a2, B2, cuda=True):
        gme1 = GaussianMetaEmbedding(a1, B1)
        gme2 = GaussianMetaEmbedding(a2, B2)
        if cuda:
            gme1 = gme1.cuda()
            gme2 = gme2.cuda()
        logden1 = gme1.log_expectation(cuda=cuda).expand(a2.size(0), a1.size(0)).t()
        logden2 = gme2.log_expectation(cuda=cuda).expand(a1.size(0), a2.size(0))

        # pooled log expectation
        apooled = gme1.a.expand(a2.size(0),*a1.size()).transpose(1,0) + gme2.a.expand(a1.size(0),*a2.size())
        Bpooled = gme1.B.expand(a2.size(0),*a1.size()).transpose(1,0) + gme2.B.expand(a1.size(0),*a2.size())

        prior = copy.deepcopy(gme1.prior_precision)
        piCon = copy.deepcopy(gme1.logscale) + copy.deepcopy(gme2.logscale)
        if cuda:
            prior = prior.cuda()
            piCon = piCon.cuda()
        chol_BI = torch.sqrt(Variable(prior).expand_as(Bpooled) + Bpooled)
        logdet_BI = torch.sum(torch.log(chol_BI), dim=2) * 2
        z = torch.reciprocal(chol_BI) * apooled
        lognum = piCon.expand_as(logdet_BI) + ((z**2).sum(dim=2) - logdet_BI) / 2

        return lognum - logden1 - logden2

    @staticmethod
    def neg_llr_identification(a1, B1, a2, B2, cuda=True):
        return -GaussianMetaEmbedding.llr_identification(a1, B1, a2, B2, cuda)

    """
    cuda=False
    a1=Variable(torch.randn(5,200))
    B1=Variable(torch.randn(5,200))
    B1=1/B1**2
    a2=Variable(torch.randn(3,200))
    B2=Variable(torch.randn(3,200))
    B2=1/B2**2

    """
    """

    @staticmethod
    def distance_posterior_llr(output1_a, output1_B, output2_a, output2_B, cuda=True):
        torch_sigmoid = nn.Sigmoid()
        if cuda:
            torch_sigmoid = torch_sigmoid.cuda()
        neg_llr = GaussianMetaEmbedding.distance_neg_llr(output1_a, output1_B, output2_a, output2_B)
        posterior_llr_dist = torch_sigmoid(neg_llr)
        return posterior_llr_dist

    @staticmethod
    def distance_euclidean(output1_a, output1_B, output2_a, output2_B):
        dist = F.pairwise_distance(output1_a, output2_a)
        return dist

    @staticmethod
    def net_distance(data, net, epoch):
        output1_a, output1_B, output2_a, output2_B, label = data
        dist = GaussianMetaEmbedding.distance_neg_llr(output1_a, output1_B, output2_a, output2_B)

        if not os.path.exists(ConfigNetwork.debug_file_basename + '_label.npy'):
            label.data.cpu().numpy().tofile(ConfigNetwork.debug_file_basename + '_label.npy')
            output1_a.data.cpu().numpy().tofile(ConfigNetwork.debug_file_basename + '_1a.npy')
            output2_a.data.cpu().numpy().tofile(ConfigNetwork.debug_file_basename + '_2a.npy')
            logging.critical('wrote debug embeddings file')
        return dist, label
    """

    """
        a1 = Variable(torch.from_numpy(numpy.fromfile('debug_embeddings_1a.npy',dtype='float32').reshape(250,200)))
        a2 = Variable(torch.from_numpy(numpy.fromfile('debug_embeddings_2a.npy',dtype='float32').reshape(250,200)))
        lbl = Variable(torch.from_numpy(numpy.fromfile('debug_embeddings_label.npy',dtype='int32'))) > 0
    """

    """
        if epoch == 0:
            # dist = GaussianMetaEmbedding.distance_neg_llr(output1_a, output1_B, output2_a, output2_B)
            dist = GaussianMetaEmbedding.distance_euclidean(output1_a, output1_B, output2_a, output2_B)
        else:
            dist = GaussianMetaEmbedding.distance_posterior_llr(output1_a, output1_B, output2_a, output2_B)
        return dist, label
    """

"""
    @staticmethod
    def train_epoch(train_dataloader, net, optimizer, epoch, iteration_number, cuda=True):
        criterion = ContrastiveLossGME()
        for i, data in enumerate(train_dataloader,0):
            sample0, sample1 , label = data
            sample0, sample1 , label = Variable(sample0), Variable(sample1), Variable(label)
            if cuda:
                sample0, sample1 , label = sample0.cuda(), sample1.cuda(), label.cuda()
            output1_a, output1_B, output2_a, output2_B = net(sample0,sample1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1_a, output1_B, output2_a, output2_B, label)
            if i % ConfigNetwork.iteration_step == 0 :
                # save data for debugging
                numpy.save(os.path.join(ConfigNetwork.storage_dir, ConfigNetwork.modelname + '_a1.npy'), output1_a.data.cpu().numpy())
                numpy.save(os.path.join(ConfigNetwork.storage_dir, ConfigNetwork.modelname + '_B1.npy'), output1_B.data.cpu().numpy())
                numpy.save(os.path.join(ConfigNetwork.storage_dir, ConfigNetwork.modelname + '_a2.npy'), output2_a.data.cpu().numpy())
                numpy.save(os.path.join(ConfigNetwork.storage_dir, ConfigNetwork.modelname + '_B2.npy'), output2_B.data.cpu().numpy())
            loss_contrastive.backward()
            optimizer.step()
            if i % ConfigNetwork.iteration_step == 0 :
                logging.info("Epoch number {},\t  iteration {},\t Current loss {}".format(epoch, i, loss_contrastive.data[0]))
                iteration_number += ConfigNetwork.iteration_step
                test_model((output1_a, output1_B, output2_a, output2_B, label), net, net_distance=GaussianMetaEmbedding.net_distance, epoch=epoch)
                # curiousity logging
                logging.debug('output1_B avg std per dims: {}'.format(
                    output1_B.data.cpu().numpy().std(axis=0).mean()))


class ContrastiveLossGME_euclidean(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLossGME_euclidean, self).__init__()
        self.register_buffer('margin', torch.FloatTensor([margin]))
    def forward(self, output1_a, output1_B, output2_a, output2_B, label):
        euclidean_distance = GaussianMetaEmbedding.distance_euclidean(output1_a, output1_B, output2_a, output2_B)
        loss_contrastive = torch.mean(
            (1-label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(Variable(self.margin) - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive


class ContrastiveLossGME(nn.Module):
    def __init__(self, margin=20): # margin=numpy.log(3)): # 1 is turning point in LR domain
        super(ContrastiveLossGME, self).__init__()
        self.register_buffer('margin', torch.FloatTensor([margin]))
    def forward(self, output1_a, output1_B, output2_a, output2_B, label):
        neg_llr = GaussianMetaEmbedding.distance_neg_llr(output1_a, output1_B, output2_a, output2_B)
        loss_contrastive = torch.mean(
            (1 - label) * torch.clamp(Variable(self.margin).expand_as(neg_llr) + neg_llr, min=0.0) +
            (label) * torch.clamp(Variable(self.margin).expand_as(neg_llr) - neg_llr, min=0.0)
        )
        return loss_contrastive


class ContrastiveLossGME_posterior(ContrastiveLossGME):
    def __init__(self, margin=0.666): # 1 is turning point in LR domain
        super(ContrastiveLossGME_posterior, self).__init__(margin=margin)
    def forward(self, output1_a, output1_B, output2_a, output2_B, label):
        posterior_llr_dist = GaussianMetaEmbedding.distance_posterior_llr(output1_a, output1_B, output2_a, output2_B)
        loss_contrastive = torch.mean(
            (1 - label) * posterior_llr_dist +
            (label) * torch.clamp(Variable(self.margin).expand_as(posterior_llr_dist) - posterior_llr_dist, min=0.0)
        )
        return loss_contrastive
"""