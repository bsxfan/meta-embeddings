import os
os.environ['SIDEKIT'] = 'theano=false,libsvm=false,mpi=false'
from torch.autograd import Variable
from torch import optim
import logging
import torch
from torch import nn
import numpy
from torch.utils.data import DataLoader
from torchvision import transforms
# from torch.optim.lr_scheduler import ReduceLROnPlateau
import h5py
from FaceDatasets import SiameseNetworkDatasetDifficult, SoftMaxDatabase
from FaceNetworks import SoftMaxNetwork, SiameseNetwork, GME_SiameseNetwork, GME_SoftmaxNetwork
from MetaEmbeddings import GaussianMetaEmbedding
from Config import ConfigNetwork, ConfigFaceDatasets
from utils_test import test_model
import sidekit
from sidekit.factor_analyser import fa_model_loop
import copy


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Nike Brümmer, Adrian Bulat"]


def net_distance(data, net, epoch=None):
    sample0, sample1, label = data
    sample0, sample1, label = Variable(sample0).cuda(), Variable(sample1).cuda(), Variable(label).cuda()
    if ConfigNetwork.train_with_softmax:
        # note: still face dependend
        if ConfigNetwork.train_with_meta_embeddings:
            output1_a, output1_B = net(sample0, None)
            output2_a, output2_B = net(sample1, None)
            return GaussianMetaEmbedding.net_distance(data=(output1_a, output1_B, output2_a, output2_B, label), net=net, epoch=epoch)
        else:
            output1 = net(sample0, None)
            output2 = net(sample1, None)
            return SoftMaxNetwork.distance_pairwise_euclidean(output1, output2), label
    else:
        if ConfigNetwork.train_with_meta_embeddings:
            output1_a, output1_B, output2_a, output2_B = net(sample0,sample1)
            return GaussianMetaEmbedding.net_distance(data=(output1_a, output1_B, output2_a, output2_B, label), net=net, epoch=epoch)
        else:
            # note: still face dependend
            output1, output2 = net(sample0, sample1)
            return SiameseNetwork.net_distance(data=(output1, output2, label), net=net, epoch=epoch)


def train_epoch(train_dataloader, net, optimizer, epoch, iteration_number):
    net.train()
    if ConfigNetwork.train_with_softmax:
        if ConfigNetwork.train_with_meta_embeddings:
            GME_SoftmaxNetwork.train_epoch(train_dataloader, net, optimizer, epoch, iteration_number)
        else:
            # same for ConfigNetwork.train_with_meta_embeddings:
            SoftMaxNetwork.train_epoch(train_dataloader, net, optimizer, epoch, iteration_number)
    else:
        if ConfigNetwork.train_with_meta_embeddings:
            # GME_SiameseNetwork.train_epoch(train_dataloader, net, optimizer, epoch, iteration_number)
            GaussianMetaEmbedding.train_epoch(train_dataloader, net, optimizer, epoch, iteration_number)
        else:
            SiameseNetwork.train_epoch(train_dataloader, net, optimizer, epoch, iteration_number)


def train_net(net, train_dataloader, test_dataloader):
    if not os.path.exists(ConfigNetwork.modelname):
        last_model_loaded = False # True # False
        # inits
        iteration_number = 0
        for epoch in range(0,ConfigNetwork.train_number_epochs):
            """
            if ConfigNetwork.learning_rate_scheduler:
                optimizer = optim.Adam(net.parameters(),
                                       lr = ConfigNetwork.learning_rate)
                scheduler = ReduceLROnPlateau(optimizer, 'min')
            else:
            """
            epoch_learning_rate_exponent = max(0, epoch - (ConfigNetwork.learning_rate_defactor_after_epoch - 1))
            parameters = filter(lambda p: p.requires_grad, net.parameters())
            optimizer = optim.Adam(parameters, lr = ConfigNetwork.learning_rate * ConfigNetwork.learning_rate_defactor**epoch_learning_rate_exponent)
            base_file_pattern = os.path.join(ConfigNetwork.storage_dir, '{}_epoch_{}'.format(ConfigNetwork.modelname, epoch))
            epoch_net_file = '{}_model'.format(base_file_pattern)
            if epoch < ConfigNetwork.freeze_ResNet_epochs:
                net.set_ResNet_requires_grad(requires_grad=False)
            else:
                net.set_ResNet_requires_grad(requires_grad=True)
            if not os.path.exists(epoch_net_file) and epoch == 0:
                # init meta embeddings network
                if ConfigNetwork.train_with_meta_embeddings:
                    logging.debug('init B with plda expectation')
                    if not os.path.exists(ConfigNetwork.embeddings_file):
                        dataset = SoftMaxDatabase(
                            imageFolderDataset=ConfigFaceDatasets.dataset_class(root=ConfigFaceDatasets.training_dir),
                            transform=train_dataloader.dataset.transform,
                            should_invert=False)
                        embeddings_loader = DataLoader(dataset,
                                                       shuffle=False,
                                                       num_workers=ConfigNetwork.num_workers,
                                                       batch_size=ConfigNetwork.batch_size_train)
                        softmax_net = net.to_softmaxNetwork()
                        softmax_net.normalize = False
                        with h5py.File(ConfigNetwork.embeddings_file, "a") as embd_file:
                            for i, data in enumerate(embeddings_loader, 0):
                                img0, label = data
                                img0, label = Variable(img0).cuda(), Variable(label).cuda()
                                output0 = super(SoftMaxNetwork, softmax_net).forward_once(img0)
                                embd_file.create_dataset("{}".format(i),
                                                         data=numpy.column_stack(
                                                             (output0.data.cpu().numpy(), label.data.cpu().numpy())),
                                                         compression="gzip",
                                                         fletcher32=True)
                            logging.critical('extracted embeddings')
                    if not os.path.exists(ConfigNetwork.embeddings_file_plda) or not os.path.exists(ConfigNetwork.embeddings_mean_file):
                        data = []
                        with h5py.File(ConfigNetwork.embeddings_file, "r") as h5f:
                            for key, value in h5f.items():
                                data.append(value.value)
                        data = numpy.concatenate(data)
                        embeddings = data[:,:ConfigNetwork.embedding_size]
                        embeddings_mean = embeddings.mean(0)
                        numpy.save(ConfigNetwork.embeddings_mean_file, embeddings_mean)
                        logging.debug('embeddings mean: {}'.format(embeddings.mean(0)))
                        embeddings -= embeddings.mean(0)
                        embeddings = (embeddings.T / numpy.linalg.norm(embeddings, axis=1)).T # prepare cosine distance
                        embedding_labels = data[:, ConfigNetwork.embedding_size:].squeeze()

                        s = sidekit.StatServer()
                        s.modelset = embedding_labels
                        s.segset = numpy.arange(embedding_labels.shape[0]).astype(str)
                        s.stat0 = numpy.ones((embedding_labels.shape[0], 1))
                        s.stat1 = copy.deepcopy(embeddings)
                        s.start = numpy.empty(embedding_labels.shape[0], dtype='|O')
                        s.stop = numpy.empty(embedding_labels.shape[0], dtype='|O')
                        s.validate()
                        ids = numpy.unique(s.modelset)
                        class_nb = ids.shape[0]

                        f = sidekit.FactorAnalyser()
                        rank_f = ConfigNetwork.embedding_size
                        f.plda(s, rank_f=rank_f)
                        f.write(ConfigNetwork.embeddings_file_plda)
                    else:
                        f = sidekit.FactorAnalyser(ConfigNetwork.embeddings_file_plda)

                    e_mu = torch.from_numpy(f.mean).type(torch.FloatTensor)
                    e_B = torch.from_numpy(numpy.linalg.inv(f.Sigma).diagonal()).type(torch.FloatTensor)
                    # e_B = torch.from_numpy(numpy.linalg.inv(f.Sigma)).type(torch.FloatTensor)
                    assert(isinstance(net, GME_SoftmaxNetwork))
                    net = GME_SoftmaxNetwork(num_train_classes=net.num_train_classes,pretrained_siamese_net=net.pretrained_net,expected_mu=e_mu,expected_B=e_B).cuda()
                    logging.debug('init B with plda done')

            if not os.path.exists(epoch_net_file):
                if last_model_loaded:
                    logging.critical('run validation on epoch {}'.format(epoch-1))
                    test_model(database_dir=test_dataloader,
                               net=net,
                               net_distance=net_distance,
                               epoch=None
                               )
                    last_model_loaded = False

                # train model
                if ConfigNetwork.select_difficult_pairs_epoch is not None:
                    # selection of most challenging
                    if epoch == ConfigNetwork.select_difficult_pairs_epoch:
                        net.eval()
                        embeddings_loader = None
                        if not os.path.exists(ConfigNetwork.embeddings_file):
                            dataset = SoftMaxDatabase(imageFolderDataset=ConfigFaceDatasets.dataset_class(root=ConfigFaceDatasets.training_dir),
                                                      transform=train_dataloader.dataset.transform,
                                                      should_invert=False)
                            embeddings_loader = DataLoader(dataset,
                                                           shuffle=False,
                                                           num_workers=ConfigNetwork.num_workers,
                                                           batch_size=ConfigNetwork.batch_size_train)
                            with h5py.File(ConfigNetwork.embeddings_file, "a") as embd_file:
                                for i, data in enumerate(embeddings_loader, 0):
                                    img0, label = data
                                    img0, label = Variable(img0).cuda(), Variable(label).cuda()
                                    output0 = net.forward_once(img0)
                                    embd_file.create_dataset("{}".format(i),
                                                             data=numpy.column_stack((output0.data.cpu().numpy(), label.data.cpu().numpy())),
                                                             compression="gzip",
                                                             fletcher32=True)
                            logging.critical('extracted embeddings for difficult pairs')

                        if not os.path.exists(ConfigNetwork.select_difficult_pairs_idx + '_genuine.npy') or not os.path.exists(ConfigNetwork.select_difficult_pairs_idx + '_impostor.npy'):
                            data = []
                            with h5py.File(ConfigNetwork.embeddings_file, "r") as h5f:
                                for key, value in h5f.items():
                                    data.append(value.value)
                            data = numpy.concatenate(data)
                            embeddings = data[:,:ConfigNetwork.embedding_size]
                            embeddings = (embeddings.T / numpy.linalg.norm(embeddings, axis=1)).T # prepare cosine distance
                            embedding_labels = data[:, ConfigNetwork.embedding_size:]
                            subjects = numpy.unique(embedding_labels.flatten())
                            # concat & score embeddings
                            lbls = embedding_labels + 1 # all labels are IDs 0 to inf, however a "0" label destroy the later lbls@lbls-lbls**2 idea as any other label times zero is zero...
                            # figure out labels, unselect same-sample comparison, i.e. diagonal
                            split_num = int(1000)
                            # assure #embeddings / split_num < ConfigNetwork.select_difficult_pairs_top_N /
                            """
                            if data.shape[0] / split_num > ConfigNetwork.select_difficult_pairs_top_N / 4:
                                split_num = int(data.shape[0] / ConfigNetwork.select_difficult_pairs_top_N * 4)
                            indice_tuples_genuine = numpy.zeros((ConfigNetwork.select_difficult_pairs_top_N, 3)) # idx1, idx2, score
                            indice_tuples_impostor = numpy.zeros((ConfigNetwork.select_difficult_pairs_top_N, 3)) # idx1, idx2, score
                            """
                            # idx1, idx2, score, label1, label2
                            indice_tuples_genuine = numpy.zeros((subjects.shape[0] * ConfigNetwork.select_difficult_pairs_topN_per_subject, 5))
                            indice_tuples_impostor = numpy.zeros((subjects.shape[0] * ConfigNetwork.select_difficult_pairs_topN_per_subject, 5))
                            indice_tuples_genuine[:,2] = -numpy.infty
                            indice_tuples_impostor[:,2] = numpy.infty
                            indice_tuples_genuine[:,3] = numpy.repeat(subjects, ConfigNetwork.select_difficult_pairs_topN_per_subject)
                            indice_tuples_genuine[:,4] = indice_tuples_genuine[:,3]
                            indice_tuples_impostor[:,3] = numpy.repeat(subjects, ConfigNetwork.select_difficult_pairs_topN_per_subject)
                            indice_tuples_impostor[:,4] = -1
                            splits = numpy.array_split(range(embeddings.shape[0]), split_num)
                            split_lens = numpy.array([len(s) for s in splits])
                            for ii, sid in enumerate(splits):
                                if ii % 100 == 0:
                                    logging.debug('emb idx: {}'.format(ii))
                                embedding_scores = - embeddings[sid,:] @ embeddings.T # negative cosine similarity
                                offset = split_lens[:ii].sum()
                                num_rows = embedding_scores.shape[0]
                                num_cols = embedding_scores.shape[1]

                                lbls_scores = numpy.abs((lbls[sid] @ lbls.T - lbls.flatten() ** 2))  # leaves zeros on same class, also: diag
                                # also, remove duplicates due to symmetric comparison
                                center_of_storm = numpy.eye(num_rows, dtype=bool)
                                center_of_storm[numpy.tril_indices(num_rows)] = True
                                eye_idx = numpy.hstack((
                                    numpy.ones((num_rows,offset), dtype=bool),
                                    center_of_storm,
                                    numpy.zeros((num_rows,num_cols - num_rows - offset), dtype=bool)
                                ))
                                embedding_scores[eye_idx] = -100  # exclude same image comparisons
                                lbls_scores[eye_idx] = -100  # same sample = -100, tar = 0, non > 0

                                # for each reference, find difficult pairs
                                batch_subjects = numpy.unique(embedding_labels.flatten()[offset:offset+num_rows])
                                for sidx, subj in enumerate(batch_subjects):
                                    subj_idx_in_list = subjects == subj

                                    batch_row_idx = (embedding_labels[offset:offset+num_rows] == subj).flatten()
                                    batch_row_pos = numpy.argwhere(batch_row_idx)

                                    subj_scores = embedding_scores[batch_row_idx,:]
                                    tar_idx = lbls_scores[batch_row_idx,:] == 0
                                    non_idx = lbls_scores[batch_row_idx,:] > 0
                                    tar_pos = numpy.argwhere(tar_idx)
                                    non_pos = numpy.argwhere(non_idx)

                                    tar = subj_scores[tar_idx]
                                    non = subj_scores[non_idx]

                                    tar_argsort = tar.argsort()[::-1]
                                    non_argsort = non.argsort()

                                    if len(tar_argsort) > 0:
                                        subj_tar_in_list = indice_tuples_genuine[:,3] == subj
                                        subj_tar_idx = numpy.argwhere(subj_tar_in_list).flatten()
                                        subj_topN_tar_argsort = indice_tuples_genuine[subj_tar_in_list,3].argsort()[::-1]
                                        tar_cnt = 0
                                        for nidx in subj_topN_tar_argsort:
                                            if tar_cnt >= tar_argsort.shape[0]:
                                                break
                                            if indice_tuples_genuine[subj_tar_idx[nidx],2] < tar[tar_argsort[tar_cnt]]:
                                                idx_subj_scr = tar_pos[tar_argsort[tar_cnt]]
                                                col = idx_subj_scr[1]
                                                row = batch_row_pos[idx_subj_scr[0]]
                                                indice_tuples_genuine[subj_tar_idx[nidx], 0:3] = numpy.array([offset + row, col, embedding_scores[row, col]]).T
                                                tar_cnt += 1

                                    if len(non_argsort) > 0:
                                        subj_non_in_list = indice_tuples_impostor[:,3] == subj
                                        subj_non_idx = numpy.argwhere(subj_non_in_list).flatten()
                                        subj_topN_non_argsort = indice_tuples_impostor[subj_non_in_list,3].argsort()
                                        non_cnt = 0
                                        for nidx in subj_topN_non_argsort:
                                            if non_cnt >= non_argsort.shape[0]:
                                                break
                                            if indice_tuples_impostor[subj_non_idx[nidx],2] > non[non_argsort[non_cnt]]:
                                                idx_subj_scr = non_pos[non_argsort[non_cnt]]
                                                col = idx_subj_scr[1]
                                                row = batch_row_pos[idx_subj_scr[0]]
                                                indice_tuples_impostor[subj_non_idx[nidx], 0:3] = numpy.array([offset + row, col, embedding_scores[row, col]]).T
                                                non_cnt += 1

                                """
                                # gather idx of topN tar/non scores
                                tar = embedding_scores[lbls_scores == 0]
                                non = embedding_scores[lbls_scores > 0]

                                tar_bool_idx = tar > indice_tuples_genuine[:,2].max()
                                non_bool_idx = non < indice_tuples_impostor[:,2].min()
                                num_top_tar = min(tar.shape[0], max(tar_bool_idx.sum(), numpy.isinf(indice_tuples_genuine[:,2]).sum()), ConfigNetwork.select_difficult_pairs_top_N)
                                num_top_non = min(non.shape[0], max(non_bool_idx.sum(), numpy.isinf(indice_tuples_impostor[:,2]).sum()), ConfigNetwork.select_difficult_pairs_top_N)

                                tar_idx = numpy.argpartition(tar, -num_top_tar)[-num_top_tar:]
                                non_idx = numpy.argpartition(non, num_top_non-1)[:num_top_non]
                                lbl_tar_pos = numpy.argwhere((lbls_scores == 0).flatten()).flatten()
                                lbl_non_pos = numpy.argwhere((lbls_scores > 0).flatten()).flatten()

                                # selection - genuine
                                if num_top_tar > 0:
                                    idx = lbl_tar_pos[tar_idx]
                                    col = idx % num_cols
                                    row = numpy.floor(idx / num_cols).astype(int)
                                    if num_top_tar >= ConfigNetwork.select_difficult_pairs_top_N:
                                        iidx = range(ConfigNetwork.select_difficult_pairs_top_N)
                                    else:
                                        iidx = numpy.argpartition(indice_tuples_genuine[:,2], num_top_tar)[:num_top_tar]
                                    indice_tuples_genuine[iidx, :] = numpy.array([offset + row, col, embedding_scores[row, col]]).T
                                # selection - impostor
                                if num_top_non > 0:
                                    idx = lbl_non_pos[non_idx]
                                    col = idx % num_cols
                                    row = numpy.floor(idx / num_cols).astype(int)
                                    if num_top_tar >= ConfigNetwork.select_difficult_pairs_top_N:
                                        iidx = range(ConfigNetwork.select_difficult_pairs_top_N)
                                    else:
                                        iidx = numpy.argpartition(indice_tuples_impostor[:,2], -num_top_non)[-num_top_non:]
                                    indice_tuples_impostor[iidx, :] = numpy.array([offset + row, col, embedding_scores[row, col]]).T
                                """

                            #indice_tuples_genuine = numpy.array(indice_tuples_genuine)
                            #indice_tuples_impostor = numpy.array(indice_tuples_impostor)
                            #tar_idx = numpy.argpartition(indice_tuples_genuine[:,0], -num_top_tar)[-num_top_tar:]
                            #non_idx = numpy.argpartition(indice_tuples_impostor[:,0], num_top_non)[:num_top_non]
                            """
                            indice_tuples_genuine=numpy.load('jhu/difficult_pairs_idx_genuine.npy')
                            indice_tuples_impostor=numpy.load('jhu/difficult_pairs_idx_impostor.npy')
                            [numpy.asarray([i, embedding_labels[int(indice_tuples_genuine[i, 0])], embedding_labels[int(indice_tuples_genuine[i, 1])], indice_tuples_genuine[i,2]]) for i in range(indice_tuples_genuine.shape[0])]
                            """
                            numpy.save(ConfigNetwork.select_difficult_pairs_idx + '_genuine_all.npy', indice_tuples_genuine)
                            numpy.save(ConfigNetwork.select_difficult_pairs_idx + '_impostor_all.npy', indice_tuples_impostor)
                            indice_tuples_genuine = indice_tuples_genuine[numpy.argwhere(
                                ~numpy.isinf(indice_tuples_genuine[:, 2])).flatten(), :]
                            indice_tuples_impostor = indice_tuples_impostor[numpy.argwhere(
                                ~numpy.isinf(indice_tuples_impostor[:, 2])).flatten(), :]
                            numpy.save(ConfigNetwork.select_difficult_pairs_idx + '_genuine.npy', indice_tuples_genuine)
                            numpy.save(ConfigNetwork.select_difficult_pairs_idx + '_impostor.npy', indice_tuples_impostor)
                            logging.critical('saved difficult pairs')

                        #indice_tuples = numpy.load(ConfigNetwork.select_difficult_pairs_idx + '.npy')
                        indice_tuples_genuine = numpy.load(ConfigNetwork.select_difficult_pairs_idx + '_genuine.npy').astype(int)
                        indice_tuples_impostor = numpy.load(ConfigNetwork.select_difficult_pairs_idx + '_impostor.npy').astype(int)
                        # ... check if the selected scores make sense, or if the topN needs to be reduced or if you want other constraints
                        # new dataloader
                        diff_dataset = SiameseNetworkDatasetDifficult(
                            imageFolderDataset=ConfigFaceDatasets.dataset_class(root=ConfigFaceDatasets.training_dir),
                            indice_tuples_genuine=indice_tuples_genuine,
                            indice_tuples_impostor=indice_tuples_impostor,
                            transform=transforms.Compose([transforms.Scale((100, 100)),
                                                          transforms.ToTensor()
                                                          ]),
                            should_invert=False)
                        train_dataloader = DataLoader(diff_dataset,
                                                      shuffle=False,
                                                      num_workers=ConfigNetwork.num_workers,
                                                      batch_size=ConfigNetwork.batch_size_train)
                        logging.critical('swapped train data loader to difficult pairs')
                    net.train()

                # train an epoch
                train_epoch(train_dataloader=train_dataloader, net=net, optimizer=optimizer, epoch=epoch, iteration_number=iteration_number)
                torch.save(obj=net.state_dict(), f=epoch_net_file)
            else:
                net.load_state_dict(torch.load(epoch_net_file))
                logging.info('loaded model for epoch: {}'.format(epoch))
                last_model_loaded = True
                continue

            logging.critical('run validation on epoch {}'.format(epoch))
            test_model(database_dir=test_dataloader,
                       net=net,
                       net_distance=net_distance,
                       epoch=None
                       )
        torch.save(obj=net.state_dict(), f='{}'.format(ConfigNetwork.modelname))
        logging.info('training completed, model stored.')