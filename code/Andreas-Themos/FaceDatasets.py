import os
import numpy
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
from PIL import Image
import PIL.ImageOps
import random
import logging
from Config import ConfigFaceDatasets, ConfigNetwork
import torch
import copy
import h5py
import logging


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brümmer, Adrian Bulat"]


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True, train_add_noise=False):
        self.transform = transform
        self.should_invert = should_invert
        if imageFolderDataset is not None:
            self.imageFolderDataset = imageFolderDataset
            self.subject_to_sample_dict = {}
            assert(isinstance(self.imageFolderDataset, ConfigFaceDatasets.dataset_class))
            self.length = len(self.imageFolderDataset.imgs)
            img_paths = numpy.asarray(self.imageFolderDataset.imgs, dtype=str)[:, 0].astype(str)
            img_class = numpy.asarray(self.imageFolderDataset.imgs, dtype=str)[:, 1].astype(str)
            for subject in numpy.unique(img_class):
                idx_selection = img_class == subject
                self.subject_to_sample_dict[subject] = list(img_paths[idx_selection])
            self.landmarks = None
            landmark_file = os.path.join(ConfigNetwork.storage_dir, self.imageFolderDataset.root.replace('/', '_') + '.h5')
            if os.path.exists(landmark_file):
                def _recurse(hdfobject, datadict, root):
                    # https://github.com/SiggiGue/hdfdict/blob/master/hdfdict/hdfdict.py
                    for key, value in hdfobject.items():
                        if type(value) == h5py.Group:
                            _recurse(value, datadict, root + '/' + key)
                        elif isinstance(value, h5py.Dataset):
                            datadict[root + '/' + key] = value.value
                    return datadict

                self.landmarks = {}
                with h5py.File(landmark_file, 'r') as h5dict:
                    self.landmarks = _recurse(h5dict, self.landmarks, '')

    def prepare_img(self, img_tuple):
        img = Image.open(img_tuple[0])
        if self.landmarks is not None:
            lm = self.landmarks[img_tuple[0]]
            if numpy.abs(lm[2]-lm[0]) > 10 and numpy.abs(lm[3]-lm[1]) > 10:
                img = img.crop(list(lm))
        img = img.convert(mode=ConfigFaceDatasets.img_convert_mode)
        if self.should_invert:
            img = PIL.ImageOps.invert(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __getitem__(self, index):
        assert(isinstance(self.imageFolderDataset, ConfigFaceDatasets.dataset_class))
        img0_tuple = self.imageFolderDataset.imgs[index] # random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        img1_tuple = None
        if should_get_same_class:
            while len(self.subject_to_sample_dict[str(img0_tuple[1])]) < 2:
                img0_tuple = random.choice(self.imageFolderDataset.imgs)
            same_imgs = list(set(self.subject_to_sample_dict[str(img0_tuple[1])]) - {img0_tuple[0]})
            img1_path = random.choice(same_imgs)
            img1_tuple = (img1_path, img0_tuple[1])
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)
            while img0_tuple[1] == img1_tuple[1]:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
        if img1_tuple is None:
            logging.error('no pair could be selected for {}'.format(img0_tuple))
            assert(False)
        img0 = self.prepare_img(img0_tuple)
        img1 = self.prepare_img(img1_tuple)
        # non-mated as 0, mated as 1:
        return img0, img1, torch.from_numpy(numpy.array([1 - should_get_same_class], dtype=numpy.float32))
    def __len__(self):
        return self.length


class SiameseNetworkDatasetDifficult(SiameseNetworkDataset):
    def __init__(self, imageFolderDataset, indice_tuples_genuine, indice_tuples_impostor, transform=None, should_invert=True, train_add_noise=False):
        super(SiameseNetworkDatasetDifficult, self).__init__(imageFolderDataset=imageFolderDataset,
                                                             transform=transform,
                                                             should_invert=should_invert,
                                                             train_add_noise=train_add_noise)
        # self.indice_tuples = indice_tuples
        self.indice_tuples_genuine=indice_tuples_genuine
        self.indice_tuples_impostor=indice_tuples_impostor
    def __getitem__(self, index):
        # this assumes #genuine == #impostor !!!
        defactor = int(4) # int(8)
        should_get_same_class = index % defactor #4 # alternating selection: 0,1 - difficult, 2,3 - random
        if should_get_same_class == 0:
            idx = (index / defactor) % self.indice_tuples_genuine.shape[0]
            idx0, idx1 = self.indice_tuples_genuine[int(idx),:2]
        elif should_get_same_class == 1:
            idx = (index / defactor + 1) % self.indice_tuples_impostor.shape[0]
            idx0, idx1 = self.indice_tuples_impostor[int(idx),:2]
        else:
            fake_idx = numpy.random.choice(super(SiameseNetworkDatasetDifficult, self).__len__(), 1)[0]
            img0, img1, label = super(SiameseNetworkDatasetDifficult, self).__getitem__(fake_idx)
        if should_get_same_class < 2:
            img0_tuple = self.imageFolderDataset.imgs[int(idx0)]
            img1_tuple = self.imageFolderDataset.imgs[int(idx1)]
            img0 = self.prepare_img(img0_tuple)
            img1 = self.prepare_img(img1_tuple)
            label = torch.from_numpy(numpy.array([should_get_same_class], dtype=numpy.float32))
        return img0, img1, label
    def __len__(self):
        return (self.indice_tuples_genuine.shape[0] + self.indice_tuples_impostor.shape[0]) * 2 # * 4 # * 2


"""
class SiameseNetworkDatasetDifficult(SiameseNetworkDataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True, train_add_noise=False, net=None, net_distance=None):
        super(SiameseNetworkDatasetDifficult, self).__init__(imageFolderDataset=imageFolderDataset,
                                                             transform=transform,
                                                             should_invert=should_invert,
                                                             train_add_noise=train_add_noise)
        self.net = net
        self.net_distance = net_distance
    def __getitem__(self, index):
        assert(isinstance(self.imageFolderDataset, ConfigFaceDatasets.dataset_class))
        img0_tuple = self.imageFolderDataset.imgs[index] # random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        img1_tuple = None
        if should_get_same_class:
            while len(self.image_dict[str(img0_tuple[1])]) < 2:
                img0_tuple = random.choice(self.imageFolderDataset.imgs)
            same_imgs = list(set(self.image_dict[str(img0_tuple[1])]) - {img0_tuple[0]})
            # img1_path = random.choice(same_imgs)
            comparison_idx = numpy.arange(min(len(same_imgs), ConfigNetwork.select_difficult_pairs_out_of_N))
            random.shuffle(comparison_idx)
            img1_path = same_imgs[0]
            top_one_score = torch.from_numpy(numpy.finfo('float').min)
            img0 = self.prepare_img(img0_tuple[0])
            for i in comparison_idx:
                img1 = self.prepare_img(same_imgs[i])
                label = numpy.zeros(1)
                img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
                output1, output2 = self.net(img0, img1)
                dist = self.net_distance(data=(output1, output2, label), net=self.net, epoch=None)[0]
                if dist > top_one_score:
                    top_one_score = dist
                    img1_path = same_imgs[i]
            img1_tuple = (img1_path, img0_tuple[1])
        else:
            img1_tuple_list = []
            while len(img1_tuple_list) <= ConfigNetwork.select_difficult_pairs_epoch:
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                while img0_tuple[1] == img1_tuple[1]:
                    img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img1_tuple[0] not in img1_tuple_list[:,0]:
                    img1_tuple_list.append(img1_tuple)
            img1_idx = 0
            top_one_score = torch.from_numpy(numpy.finfo('float').max)
            img0 = self.prepare_img(img0_tuple[0][0])
            for i in numpy.arange(ConfigNetwork.select_difficult_pairs_epoch):
                img1 = self.prepare_img(img1_tuple_list[i][0])
                label = numpy.ones(1)
                img0, img1 , label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda()
                output1, output2 = self.net(img0, img1)
                dist = self.net_distance(data=(output1, output2, label), net=self.net, epoch=None)[0]
                if dist > top_one_score:
                    top_one_score = dist
                    img1_idx = i
            #img1_tuple = (img1_path, img0_tuple[1])
            img1_tuple = img1_tuple_list[img1_idx]
        if img1_tuple is None:
            logging.error('no pair could be selected for {}'.format(img0_tuple))
            assert(False)
        img0 = self.prepare_img(img0_tuple)
        img1 = self.prepare_img(img1_tuple)
        # non-mated as 0, mated as 1:
        return img0, img1, torch.from_numpy(numpy.array([1 - should_get_same_class], dtype=numpy.float32))
"""


class SoftMaxDatabase(SiameseNetworkDataset):
    def __getitem__(self, index):
        assert(isinstance(self.imageFolderDataset, ConfigFaceDatasets.dataset_class))
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        img0 = self.prepare_img(img0_tuple)
        return img0, img0_tuple[1]


class GMESoftMaxDatabaseTrain(SiameseNetworkDataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True, train_add_noise=False):
        if isinstance(imageFolderDataset, SiameseNetworkDataset):
            super(GMESoftMaxDatabaseTrain, self).__init__(imageFolderDataset=None,
                                                          transform=imageFolderDataset.transform,
                                                          should_invert=imageFolderDataset.should_invert,
                                                          train_add_noise=train_add_noise)
            self.imageFolderDataset = copy.deepcopy(imageFolderDataset.imageFolderDataset)
            self.image_dict = copy.deepcopy(imageFolderDataset.subject_to_sample_dict)
            self.length = copy.deepcopy(imageFolderDataset.length)
            self.landmarks = copy.deepcopy(imageFolderDataset.landmarks)
        else:
            super(GMESoftMaxDatabaseTrain, self).__init__(imageFolderDataset=imageFolderDataset,
                                                          transform=transform,
                                                          should_invert=should_invert,
                                                          train_add_noise=train_add_noise)
    def __getitem__(self, index):
        assert(isinstance(self.imageFolderDataset, ConfigFaceDatasets.dataset_class))
        img0_tuple = self.imageFolderDataset.imgs[index]
        img0 = self.prepare_img(img0_tuple)
        return img0, img0_tuple[1]


class LazyApproxGibbsSampler(SiameseNetworkDataset):
    def __init__(self, train_dataset):
        assert(isinstance(train_dataset, SiameseNetworkDataset))
        super(LazyApproxGibbsSampler, self).__init__(imageFolderDataset=None,
                                                     transform=train_dataset.transform,
                                                     should_invert=train_dataset.should_invert)
        self.imageFolderDataset = copy.deepcopy(train_dataset.imageFolderDataset)
        self.image_dict = copy.deepcopy(train_dataset.image_dict)
        self.length = copy.deepcopy(train_dataset.length)
        self.landmarks = copy.deepcopy(train_dataset.landmarks)

        # prepare approx. Gibbs sampling
        # 1. select m idx out of 1..n as I @ random or @ round-robin
        # 2. obtain L\I, remove I from L, #classes: k' -> in LI
        #   3. sample m class labels, not in L\I, providing 0 to m additional classes -> #add. classes: k''
        #   4. sample k'' recordings by some flavor of Markov chain Monte Carlo
        # 5. reassamble L = LI + L\I
        # 6. Metropolis-Hastings acceptance criterion, see Bishop, Ch. 11
        # => simplification: of the k'' depending instances, we remember what we have already seen, that's it

        # difficult pairs by LLRs, later

        num_samples_per_subject = [len(v) for v in self.image_dict.values()]
        self.type = numpy.tile([True, False], int(self.length / 2))
        while self.type.shape[0] < self.length:
            self.type = numpy.append(self.type, False)
        self.type_pos = numpy.argwhere(~self.type).flatten()
        self.openset_map = numpy.random.permutation(self.type_pos)
    def __getitem__(self, index):
        assert(isinstance(self.imageFolderDataset, ConfigFaceDatasets.dataset_class))
        type = self.type[index] # mated (or False: open-set)
        if type:
            img0_tuple = self.imageFolderDataset.imgs[index]
        else:
            pos = numpy.argwhere(index == self.type_pos).flatten()[0]
            idx = self.openset_map[pos]
            #if isinstance(idx, numpy.ndarray):
            #    idx = idx[0]
            img0_tuple = self.imageFolderDataset.imgs[idx]
        img0 = self.prepare_img(img0_tuple)
        return img0, img0_tuple[1]


"""
class DifficultPairsApproxGibbsSampler(SiameseNetworkDataset):
    def __init__(self, train_dataset, softmax_vs_softmax_llrs):
        assert(isinstance(train_dataset, SiameseNetworkDataset))
        super(DifficultPairsApproxGibbsSampler, self).__init__(imageFolderDataset=None,
                                                      transform=train_dataset.transform,
                                                      should_invert=train_dataset.should_invert)
        self.imageFolderDataset = copy.deepcopy(train_dataset.imageFolderDataset)
        self.image_dict = copy.deepcopy(train_dataset.image_dict)
        self.length = copy.deepcopy(train_dataset.length)
        self.landmarks = copy.deepcopy(train_dataset.landmarks)

        # prepare approx. Gibbs sampling
        # 1. select m idx out of 1..n as I @ random or @ round-robin
        # 2. obtain L\I, remove I from L, #classes: k' -> in LI
        #   3. sample m class labels, not in L\I, providing 0 to m additional classes -> #add. classes: k''
        #   4. sample k'' recordings by some flavor of Markov chain Monte Carlo
        # 5. reassamble L = LI + L\I
        # 6. Metropolis-Hastings acceptance criterion, see Bishop, Ch. 11
        # => simplification: of the k'' depending instances, we remember what we have already seen, that's it

        # difficult pairs by LLRs, later

        num_samples_per_subject = [len(v) for v in self.image_dict.values()]
        self.type = numpy.tile([True, False], int(self.length / 2))
        self.type_pos = numpy.argwhere(~self.type).flatten()
        while self.type.shape[0] < self.length:
            self.type = numpy.append(self.type, True)
        # softmax_vs_softmax_llrs
        self.openset_map = numpy.random.permutation(self.type_pos)
    def __getitem__(self, index):
        assert(isinstance(self.imageFolderDataset, ConfigFaceDatasets.dataset_class))
        type = self.type[index] # mated (or False: open-set)
        if type:
            img0_tuple = self.imageFolderDataset.imgs[index]
        else:
            pos = numpy.argwhere(index == self.type_pos)[0]
            img0_tuple = self.imageFolderDataset.imgs[self.openset_map[pos]]
        img0 = self.prepare_img(img0_tuple)
        return img0, img0_tuple[1]
"""


class LFW_SiameseNetworkDataset(SiameseNetworkDataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        super(LFW_SiameseNetworkDataset, self).__init__(imageFolderDataset, transform, should_invert)
        # load comparison files
        def to_utf8(arr):
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    arr[i][j] = arr[i][j].decode('UTF-8')
            return arr
        self.comparisons = []
        # mated
        evl_mated = to_utf8(numpy.loadtxt(ConfigFaceDatasets.LFW_files[0], dtype=str))
        for idx, (ref_label, ref_idx, prb_idx) in evl_mated:
            ref_img = os.path.join(ConfigFaceDatasets.testing_dir, ref_label, ref_label + '_' + '{:04d}'.format(ref_idx))
            prb_img = os.path.join(ConfigFaceDatasets.testing_dir, ref_label, ref_label + '_' + '{:04d}'.format(prb_idx))
            label = torch.from_numpy(numpy.array([0], dtype=numpy.float32))
            self.comparisons.append([ref_img, prb_img, label])
        # non-mated
        evl_nonmated = to_utf8(numpy.loadtxt(ConfigFaceDatasets.LFW_files[1], dtype=str))
        for idx, (ref_label, ref_idx, prb_label, prb_idx) in evl_nonmated:
            ref_img = os.path.join(ConfigFaceDatasets.testing_dir, ref_label, ref_label + '_' + '{:04d}'.format(ref_idx))
            prb_img = os.path.join(ConfigFaceDatasets.testing_dir, prb_label, prb_label + '_' + '{:04d}'.format(prb_idx))
            label = torch.from_numpy(numpy.array([1], dtype=numpy.float32))
            self.comparisons.append([ref_img, prb_img, label])
    def __getitem__(self, index):
        img0, img1, label = self.comparisons[index]
        img0 = self.prepare_img((img0, 'dummy'))
        img1 = self.prepare_img((img1, 'dummy'))
        return img0, img1, label
    def __len__(self):
        return len(self.comparisons)


def get_data_loader(folder, shuffle=True, train_add_noise=False):
    folder_dataset = dset.ImageFolder(root=folder)
    # from https://github.com/pytorch/examples/blob/master/imagenet/main.py
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if train_add_noise:
        transform = transforms.Compose([transforms.Resize((100,100)),
                                        transforms.Grayscale(),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomResizedCrop(100),
                                        #transforms.RandomVerticalFlip(),
                                        #transforms.TenCrop(100)
                                        #transforms.ColorJitter(),
                                        #transforms.RandomRotation(),
                                        #transforms.RandomGrayscale(),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])
        """
        if ConfigNetwork.train_with_meta_embeddings:
            transform = transforms.Compose([transforms.Resize((100,100)),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomResizedCrop(100),
                                            transforms.RandomVerticalFlip(),
                                            # transforms.TenCrop(100),
                                            transforms.ColorJitter(),
                                            transforms.RandomRotation(degrees=60),
                                            transforms.RandomGrayscale(),
                                            transforms.ToTensor(),
                                            normalize,
                                            ])
        """
    else:
        transform = transforms.Compose([transforms.Resize((100,100)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])
    if (ConfigNetwork.train_with_softmax or ConfigNetwork.train_vae) and folder is ConfigFaceDatasets.training_dir:
        dataset = SoftMaxDatabase(imageFolderDataset=folder_dataset,
                                  transform=transform,
                                  should_invert=False)
    elif folder is ConfigFaceDatasets.testing_dir:
        dataset = LFW_SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                            transform=transform,
                                            should_invert=False)
    else:
        dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform,
                                        should_invert=False,
                                        train_add_noise=train_add_noise)
    batch_size = ConfigNetwork.batch_size_train
    if folder is not ConfigFaceDatasets.training_dir:
        batch_size = ConfigNetwork.batch_size_test
    dataloader = DataLoader(dataset,
                            shuffle=shuffle,
                            num_workers=ConfigNetwork.num_workers,
                            batch_size=batch_size)
    return dataloader