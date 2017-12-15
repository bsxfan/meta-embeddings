import sys
import dlib
from torch.utils.data import DataLoader,Dataset
import torchvision.datasets as dset
from Config import ConfigFaceDatasets, ConfigNetwork
from skimage import io
import numpy
import os
os.environ['SIDEKIT'] = 'theano=false,libsvm=false,mpi=true'
from sidekit.sidekit_io import write_dict_hdf5
# from mpi4py import MPI


# > mpiexec -n 16 python landmarks.py &> log-landmarks-mpi
# with 16 as number of CPUs

# comm = MPI.COMM_WORLD
# comm.Barrier()


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Nike Brümmer, Adrian Bulat"]


folders = [
    ConfigFaceDatasets.training_dir,
    ConfigFaceDatasets.validation_dir,
    ConfigFaceDatasets.testing_dir
]

for folder in folders:
    folder_dataset = dset.ImageFolder(root=folder)

    cnn_face_detector = dlib.cnn_face_detection_model_v1(ConfigFaceDatasets.landmark_net_file)

    imgs = numpy.asarray(folder_dataset.imgs)
    num_imgs = imgs.shape[0]
    landmarks = numpy.zeros((num_imgs, 4))
    # batch_imgs_idx = numpy.array_split(numpy.arange(num_imgs), comm.size)
    # batch_landmarks = numpy.zeros((len(batch_imgs_idx[comm.rank]), 4),dtype='int')
    # for idx in batch_imgs_idx[comm.rank]:
    for idx, (f, c) in enumerate(folder_dataset.imgs):
        # f = folder_dataset.imgs[idx][0]
        print("Processing file: {}".format(f))
        img = io.imread(f)
        # The 1 in the second argument indicates that we should upsample the image
        # 1 time.  This will make everything bigger and allow us to detect more
        # faces.
        # dets = cnn_face_detector(img, 1)
        dets = cnn_face_detector(img, 0)

        '''
        This detector returns a mmod_rectangles object. This object contains a list of mmod_rectangle objects.
        These objects can be accessed by simply iterating over the mmod_rectangles object
        The mmod_rectangle object has two member variables, a dlib.rectangle object, and a confidence score.
    
        It is also possible to pass a list of images to the detector.
            - like this: dets = cnn_face_detector([image list], upsample_num, batch_size = 128)
        In this case it will return a mmod_rectangless object.
        This object behaves just like a list of lists and can be iterated over.
        '''
        print("Number of faces detected: {}".format(len(dets)))
        last_confidence = 0
        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(
                i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
            # bounding_box_diag_size = numpy.abs(d.rect.left() - d.rect.right()) + numpy.abs(d.rect.top() - d.rect.bottom())
            if d.confidence > last_confidence:
                # batch_landmarks[idx, : ] = numpy.array([d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()])
                landmarks[idx, :] = numpy.array([d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom()])
                last_confidence = d.confidence

    # reduce
    """
    comm.Barrier()
    sendcounts = numpy.array([4 for idx in range(num_imgs)])
    displacements = numpy.hstack((0, numpy.cumsum(sendcounts)[:-1]))
    if comm.rank == 0:
        landmarks = numpy.zeros((num_imgs, 4))
    else:
        landmarks = None
    comm.Barrier()
    comm.Gatherv(batch_landmarks, [landmarks, sendcounts, displacements, MPI.INT], root=0)
    """

    # if comm.rank == 0:
    if True:
        landmarks_dict = dict(zip(imgs[:,0], landmarks))
        landmark_file = os.path.join(ConfigNetwork.storage_dir, folder.replace('/', '_') + '.h5')
        write_dict_hdf5(landmarks_dict, landmark_file)