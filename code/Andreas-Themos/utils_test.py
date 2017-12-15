import numpy
import logging
from torch.utils.data import DataLoader

from utils_data import data_loader


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Nike Brümmer, Adrian Bulat"]


def neglogsigmoid(log_odds):
    neg_log_p = -log_odds
    exp_log_o = numpy.exp(-log_odds)
    exp_idx = exp_log_o < exp_log_o + 1
    neg_log_p[exp_idx] = numpy.log(1 + exp_log_o[exp_idx])
    return neg_log_p


def cllr(tar, non):
    c1 = neglogsigmoid(tar).mean()
    c2 = neglogsigmoid(-non).mean()
    return (c1+c2) / numpy.log(2) / 2


def eer_interpolation(tar, non):
    eer_linspace = numpy.linspace(non.mean(), tar.mean(), num=200)
    eer_matrix = numpy.zeros((200, 2))
    for eer_i in range(200):
        fmr = (non >= eer_linspace[eer_i]).sum() / non.shape[0]
        fnmr = (tar < eer_linspace[eer_i]).sum() / tar.shape[0]
        eer_matrix[eer_i, 0] = numpy.abs(fmr - fnmr)
        eer_matrix[eer_i, 1] = (fmr + fnmr) / 2
    eer_idx = numpy.argmin(eer_matrix[:, 0])
    eer = eer_matrix[eer_idx, 1]
    return eer


def test_model(database_dir, net, net_distance, epoch=None):
    # net_distance returns: (distance, labels)
    # test
    dataloader = data_loader(database_dir)
    logging.debug('test data loaded')
    tar = []
    non = []
    net.eval()
    if isinstance(dataloader, DataLoader):
        for i, data in enumerate(dataloader, 0):
            distance, label = net_distance(data=data, net=net, epoch=epoch)
            label = label.view(-1)
            tar_dist = distance[label == 0].data.cpu().numpy()
            non_dist = distance[label == 1].data.cpu().numpy()
            tar.append(list(tar_dist))
            non.append(list(non_dist))
            """
            tar_mu = tar_dist.mean()
            tar_std = tar_dist.std()
            non_mu = non_dist.mean()
            non_std = non_dist.std()
            logging.debug("tar mean: {}, std: {}".format(tar_mu, tar_std))
            logging.debug("non mean: {}, std: {}".format(non_mu, non_std))
            eer = eer_interpolation(-tar_dist, -non_dist)
            logging.debug('batch eer: {}'.format(eer))
            """
        tar = numpy.concatenate(tar)
        non = numpy.concatenate(non)
    else:
        # examine a data batch
        distance, label = net_distance(data=dataloader, net=net, epoch=epoch)
        label = label.view(-1)
        tar = distance[label == 0].data.cpu().numpy()
        non = distance[label == 1].data.cpu().numpy()
    tar_mu = tar.mean()
    tar_std = tar.std()
    non_mu = non.mean()
    non_std = non.std()
    logging.debug("tar mean: {}, std: {}".format(tar_mu, tar_std))
    logging.debug("non mean: {}, std: {}".format(non_mu, non_std))
    eer = eer_interpolation(-tar, -non)
    if epoch is None:
        logging.critical('EER: {}'.format(eer))
        # cllr_ = cllr(tar, non)
        # logging.critical('EER: {}, Cllr: {}'.format(eer, cllr_))
    else:
        logging.debug('batch eer: {}'.format(eer))


"""
test_siamese_dataset = test_dataloader.dataset
assert (isinstance(test_siamese_dataset, SiameseNetworkDataset))
test_imageFolderDataset = test_siamese_dataset.imageFolderDataset
assert (isinstance(test_imageFolderDataset, Config.dataset_class))
tar, non = comparison_singles(embedding0_batch=numpy.asarray(test_imageFolderDataset.imgs)[:, 0],
                              class0_batch=numpy.asarray(test_imageFolderDataset.imgs)[:, 1],
                              embedding_stats_file=embedding_stats_file)
eer = eer_interpolation(tar, non)
"""