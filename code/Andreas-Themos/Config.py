import torchvision.datasets as dset
from torchvision.datasets.folder import is_image_file
from torchvision.models.resnet import BasicBlock, resnet18, resnet34, resnet50, resnet101, resnet152
import logging
import numpy


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Nike Brümmer, Adrian Bulat"]


def init_logging():
    log_filename = 'log-baseline-face'
    level = logging.DEBUG

    numpy.set_printoptions(linewidth=250, precision=4)
    frm = '%(asctime)s - %(levelname)s - %(message)s'
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)

    logging.basicConfig(format=frm, level=level)
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(logging.Formatter(frm))
    fh.setLevel(level)
    root.addHandler(fh)


class ConfigNetwork():
    # network definition
    embedding_size = 200 # 200 # 600 # should be between 100 and 200, eg as PLDA, note facenet: 120
    precision_size = 0 # 200 # 600 # a prime for testing
    train_number_epochs = 200

    debug_file_basename = '/export/b16/tstafylakis/debug_embeddings'

    init_resnet = resnet18(pretrained=True)
    basic_block = BasicBlock

    normalize_embedding = True
    train_vae = False
    train_with_softmax = True if not train_vae else False
    dropouts = True
    train_with_meta_embeddings = True # True # False


    # training, validation, testing set-up
    num_workers = 8
    batch_size_train = 100 if train_with_meta_embeddings else 50
    batch_size_test = 250
    iteration_step = 100 # logging

    num_train_files = 162770
    freeze_ResNet_epochs = 2 if train_with_meta_embeddings else 1
    # freere_ResNet_layer_depth = 0 if train_with_meta_embeddings else 4
    freere_ResNet_layer_depth = 5 if train_with_meta_embeddings else 4

    # learning_rate_scheduler = False
    learning_rate = 0.0001 if train_with_meta_embeddings else 0.001 # 0.0005
    if train_vae:
        learning_rate = 0.01
    learning_rate_defactor = 0.8 # 0.9 # 0.8
    learning_rate_defactor_after_epoch = 10 if train_with_meta_embeddings else 15 # w/o: 5 - w/: # 2 # 10

    vae_fudge = 2.718

    select_difficult_pairs_epoch = None  # 20 # None
    select_difficult_pairs_topN_per_subject = 5

    # output
    storage_dir = '/export/b16/tstafylakis/'
    modelname = storage_dir + 'trained_{siamese}_network_celeba_{gme}'.format(
        siamese='softmax' if train_with_softmax else 'siamese',
        gme='gaussian_meta_embedding' if train_with_meta_embeddings else 'embedding'
    )
    pretrained_siamese_net = '/export/b16/tstafylakis/171206/trained_softmax_network_celeba_embedding_epoch_118_model' # None
    embeddings_file = storage_dir + 'train_embeddings.h5'
    embeddings_file_plda = storage_dir + 'train_embeddings_plda.h5'

    embeddings_mean_file = storage_dir + 'train_embeddings_mean.npy' if train_with_meta_embeddings else None

    select_difficult_pairs_idx = storage_dir + 'difficult_pairs_idx'

    train_dataloader = storage_dir + 'train_loader.p'
    test_dataloader = storage_dir + 'test_loader.p'


class ConfigFaceDatasets():
    # data sets
    dataset_class = dset.ImageFolder # root -> identity folders -> samples, for now, images, see dset.ImageFolder.IMG_EXTENSIONS
    dataset_checkup_fn = is_image_file
    training_dir = ConfigNetwork.storage_dir + "celeba_trn"
    validation_dir = ConfigNetwork.storage_dir + "celeba_val"
    testing_dir = ConfigNetwork.storage_dir + "lfw-deepfunneled"
    LFW_files = ['~/pairs_mated.txt', '~/pairs_nonmated.txt']
    # sample normalization
    img_convert_mode ="RGB" # "L"
    landmark_net_file = ConfigNetwork.storage_dir + 'mmod_human_face_detector.dat'
