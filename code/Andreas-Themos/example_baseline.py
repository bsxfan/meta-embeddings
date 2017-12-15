import os
import torch
import logging

from Config import ConfigNetwork, ConfigFaceDatasets, init_logging
from FaceDatasets import GMESoftMaxDatabaseTrain, LazyApproxGibbsSampler
from torch.utils.data import DataLoader
from FaceNetworks import SoftMaxNetwork, SiameseNetwork, GME_SiameseNetwork, GME_SoftmaxNetwork #, VAE
from utils_data import data_loader
from utils_train import train_net, net_distance
from utils_test import test_model
import pickle


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Nike Brümmer, Adrian Bulat"]


init_logging()
logging.info('initialize')
logging.debug('gpu: {}'.format(os.environ.get('CUDA_VISIBLE_DEVICES')))
A = torch.Tensor(2,2).normal_().cuda() # just checking if CUDA is there
logging.info('softmax: {}'.format(ConfigNetwork.train_with_softmax))
logging.info('VAE: {}'.format(ConfigNetwork.train_vae))
logging.info('Gaussian meta embedding: {}'.format(ConfigNetwork.train_with_meta_embeddings))
logging.info('Normalized embedding: {}'.format(ConfigNetwork.normalize_embedding))
logging.info('dev: {}'.format(ConfigFaceDatasets.training_dir))
logging.info('val: {}'.format(ConfigFaceDatasets.validation_dir))
logging.info('evl: {}'.format(ConfigFaceDatasets.testing_dir))

if not os.path.exists(ConfigNetwork.train_dataloader):
    train_dataloader = data_loader(ConfigFaceDatasets.training_dir, train_add_noise=True)
    pickle.dump( train_dataloader, open( ConfigNetwork.train_dataloader, "wb" ) )
else:
    train_dataloader = pickle.load(open(ConfigNetwork.train_dataloader, "rb"))

if ConfigNetwork.train_with_meta_embeddings:
    """
    train_dataloader = DataLoader(GMESoftMaxDatabaseTrain(train_dataloader.dataset),
                            shuffle=False,
                            num_workers=ConfigNetwork.num_workers,
                            batch_size=ConfigNetwork.batch_size_train)
    """
    train_dataloader = DataLoader(LazyApproxGibbsSampler(train_dataloader.dataset),
                                  shuffle=False,
                                  num_workers=ConfigNetwork.num_workers,
                                  batch_size=ConfigNetwork.batch_size_train)

logging.debug('train data loaded')
if not os.path.exists(ConfigNetwork.test_dataloader):
    test_dataloader = data_loader(ConfigFaceDatasets.validation_dir, shuffle=False)
    pickle.dump( test_dataloader, open( ConfigNetwork.test_dataloader, "wb" ) )
else:
    test_dataloader = pickle.load(open(ConfigNetwork.test_dataloader, "rb"))
logging.debug('test data loaded')

# initializing plain network, existing nets are loaded later
if ConfigNetwork.train_with_softmax:
    logging.info('train softmax network')
    num_train_classes = len(train_dataloader.dataset.image_dict.keys())
    if ConfigNetwork.train_with_meta_embeddings:
        logging.info('embedding: {} with B: {}'.format(ConfigNetwork.embedding_size, ConfigNetwork.precision_size))
        net = GME_SoftmaxNetwork(num_train_classes=num_train_classes).cuda()
    else:
        net = SoftMaxNetwork(num_train_classes=num_train_classes).cuda()
#elif ConfigNetwork.train_vae:
#    logging.info('train VAE network')
#    net = VAE(use_cuda=True)
else:
    logging.info('train siamese network')
    if ConfigNetwork.train_with_meta_embeddings:
        logging.info('embedding: {} with B: {}'.format(ConfigNetwork.embedding_size, ConfigNetwork.precision_size))
        net = GME_SiameseNetwork().cuda()
    else:
        net = SiameseNetwork().cuda()

# training and validation
train_net(net=net, train_dataloader=train_dataloader, test_dataloader=test_dataloader)

# load trained models
net.load_state_dict(torch.load(ConfigNetwork.modelname))
logging.info('model loaded.')


# testing
evl_dataloader = data_loader(ConfigFaceDatasets.testing_dir, shuffle=False)
logging.debug('eval data loaded')
logging.debug('eval mated comparisons')
base_file_pattern = os.path.join(ConfigNetwork.storage_dir, '{}_eval'.format(ConfigNetwork.modelname))
test_model(database_dir=evl_dataloader,
           net=net,
           net_distance=net_distance
           )
