# Meta Embeddings Implementation <br> *using PyTorch*
**prototype task: face recognition, targeting speaker recognition</i>**

This package is motivated by Niko Brümmer's idea on **Meta Embeddings**, see:
https://github.com/bsxfan/meta-embeddings

Targeting fast application development, we rather considered face recognition data than the speaker recognition task.
Experiments on synthetic data for outlining theoretic principles are not considered for this code, which rather outlines a hands-on experience.
The code is organized, such that one may easliy adopt classes depending on the visual domain to the acoustic domain.
Therefore, miniconda is employed sustaining compatability to the sidekit project, see:
https://pelennor.univ-lemans.fr/Larcher/sidekit

The code follows a save-load architecture for the sake of fast and reproducible development.
Notably, one needs to keep track of existing files in order to avoid inconsistencies. <br>
(to clean this up is an open todo @andreas)

## Framework employing PyTorch
In order to avoid errors due to ill-maintained code redundancy, a modularized framework is set-up.
Thereby, modules are triggered depending on the outline of a config file.
The experimental baseline comprises training, validation, and testing.
Utility functions are generalize task dependend functions (face, speaker, ...).
Network type specific methods are considered to be part of depending classes specifying a network.
Aiming at 1:1 (pairwise) or n:n / 1:m (all-vs-all / idenfitication) recognition scenaria, the ***Siamese network*** is employed as root class to derviable networks.
<br>
Note: networks are assumed to return *distance scores*, such as negative LLRs.

* Config.py
  * class ConfigNetwork <br>
    `singleton outlining embedding size etc.`
  * class ConfigFaceDatasets <br>
    `singleton outlining storage paths of face datasets` <br>
    `incl. segmentation file (landmarks)`
  * func init_logging <br>
    `first function to be called from main scripts` <br>
    note: the logging is inspired by sidekit
* example_baseline.py <br> steps:
  1) initialize logging module
  2) data loader: train data
  3) data loader: validation data
  4) data loader: init plain network (depending on config file)
  5) epoch training and testing
  6) testing
* utils_data.py
  * func data_loader <br>
    `wrapper providing data loaders, interface to e.g., FaceDatasets`
* utils_test.py
  * func neglogsigmoid
  * func cllr
  * func eer_interpolation <br>
    `based on 200 linear sampled thresholds between (µ(tar), µ(non))`
  * func test_model <br> steps:
    1) `net.eval()`
    2) tar, non scores in batches (1 or more) <br> note: 1 batch scenario for exploiting a training batch
    3) eer approximation (overall)
* utils_train.py
  * func net_distance <br>
    `wrapper interface to network depending scoring function` <br>
    `returning function handle` <br>
    `args: (data_tuple, net, epoch)`
  * func train_epoch <br> steps:
    1) `net.train()`
    2) call the network depending `train_epoch` static function <br>
       `args: (train_dataloader, net, optimizer, epoch, iteration_number)`
  * func train_net <br> steps:
    1) `last_model_loaded = True` flag issues a pre-validation, set to `False`
    2) iterate over epochs:
       1) `optimizer = optim.Adam`, also set learning rate
       2) freeze/unfreeze net layers e.g., the entire ResNet
       3) if `epoch = 0` and `train_with_meta_embeddings`:
          1) sequentially load train embeddings with labels
          2) estimate PLDA model (sidekit)
          3) init GME with expected µ and Sigma
       4) validate (last loaded net) epoch
       5) (optional) selection of difficult pairs <br>
          `e.g., by (negative) dot product`
       6) `train_epoch`
       7) `torch.save(obj=net.state_dict(), f=epoch_net_file)`
    3) save final model

## Network classes
Prototype network classes are provided in ***FaceNetworks.py***.
Networks inherit from `torch.nn.Module` and within the above framework are expected to elaborate on the following functions (with an exemplary code line each):
```python
class FooNetwork(torch.nn.Module):
    # instance functions
    def __init__(self, embedding_size):
        self.fc = torch.nn.Linear(self.num_fc_input * 4 * 4, self.embedding_size)
     
    def set_ResNet_requires_grad(self, requires_grad):
        self.conv1.requires_grad = requires_grad
     
    def forward_once(self, x):
        # forward through conventional ResNet
        x = self.fc(x)
        # normalize
     
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
        for i, data in enumerate(train_dataloader,0):
            img0, img1 , label = data
            output1,output2 = net(img0,img1)
            optimizer.zero_grad()
            loss = criterion(output1,output2, label)
            loss.backward()
            optimizer.step()
            if i % ConfigNetwork.iteration_step == 0 :
                logging.info("Epoch number {},\t  iteration {},\t Current loss {}".format(epoch, i, loss.data[0]))
                iteration_number += ConfigNetwork.iteration_step
```

The following networks are implemented (2017-12-14):
* SiameseNetwork
* SoftMaxNetwork(SiameseNetwork)
* GME_SiameseNetwork(SiameseNetwork)
* GME_SoftmaxNetwork(GME_SiameseNetwork)

Siamese networks are motivated by:
https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch/blob/master/Siamese-networks-medium.ipynb

We started from an object recognition pre-trained ResNet, and examined the CelebA dataset.
The baseline yielded 45% EER, we achieved ~14% EER with training Siamese nets and ~9% EER with Softmax nets on random CelebA validation subset.
So far, we solely looked into GME_SoftmaxNetwork, based on an initial precision estimate via PLDA, a 14% EER resulted when shifting the Softmax to an GME net.
However, after each epoch (regardless of the training scheme), performance declined on rather stable losses.
Embeddings appear to be rather precise per class and also over all classes, i.e. they don't seem to be distributed on a sphere, but rather to fall into clusters.

As such a supervised discriminative training might not be favorable for this particular task.
Targeting generative models, our next steps are based on:
https://github.com/wiseodd/generative-models

For which Pyro might be of particular interest: <br>
http://docs.pyro.ai/ <br>
http://pyro.ai/examples/vae.html

## Meta-Embeddings
Conceptually, the GME networks estimate the natural parameters, to be utilized as GMEs.
The simplified GME is implemented according to:
https://github.com/bsxfan/meta-embeddings/tree/master/code/Niko/matlab/clean/SGME_toolkit

see in particular:<br>
https://github.com/bsxfan/meta-embeddings/blob/780bb7c40140ebf2bf5ad81208be8955bac70b86/code/Niko/matlab/create_plain_GME.m

Contrastive to Niko's code implementing backtracking in Matlab, this implementation refers to depending PyTorch implementations.
Note: PyTorch is not yet stable, we went through a phase of major changes, and more are expected to come. The corresponding PyTorch version (build from git sources) is `0.4.0a0+38f1344`

Regarding PyTorch, the following functions are of interest:
* self.log_expectation()
* GaussianMetaEmbedding.llr_verification(a1, B1, a2, B2) <br>
similarity for pairwise batches
* GaussianMetaEmbedding.llr_identification(a1, B1, a2, B2) <br>
all-vs-all similarity for e.g., single batches vs. softmax

## Data organization
PyTorch assembles (sequential/randomized) batches by `torch.utils.data.DataLoader`.
Underlying dataset are managed by `torch.utils.data.Dataset`, of which we create custom Datasets depending on the preferred data sampling approach.
These Datesets rely on a `DataFolder` class, expecting data structured as:
* subject ID folder (i.e. the class, speaker, ...)
  * sample ID file (i.e. the instance, audio, ...)
  
However, as torch is driven by the vision community, a `ImageFolder(torch.utils.data.Dataset)` class is available, parsing folders solely regarding:
`IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']` files, regarding speaker recognition, the definition of a corresponding `AudioFolder` class would be a good step.
Furthermore, one could consider to employ sidekit file formats as well.

Datasets are organized in `FaceDatasets.py`, similary to the netowrk inheritance concept, again we start with a custom SiameseNetworkDataset from which others inherit:
* func get_data_loader(folder, shuffle=True, train_add_noise=False)
  * read folders, serializing dataset structure
  * declaration of (image) transforms
    1) resize
    2) (optional) add noise, cropping (i.e. truncation), and other degradations
    3) to_tensor
    4) normalize
  * wrap to your favorite dataset sampler (see below)
  * return batch `DataLoader`
* class SiameseNetworkDataset(Dataset)
  * attribute: `subject_to_sample_dict` <br> dictionary mapping classes to instances for easier data sampling
  * attribute: `landmarks` <br> if landmark hdf5 file exists: a dictionary of landmarks per path
  * func `prepare_img(self, img_tuple)` applies landmarks and transforms
  * `Dataset` class utilizes two functions to be overwriten
    * `def __getitem__(self, index)` e.g., prepare image and provide with label
      ```python
        img0 = self.prepare_img(img0_tuple)
        img1 = self.prepare_img(img1_tuple)
        return img0, img1, torch.from_numpy(numpy.array([1 - should_get_same_class], dtype=numpy.float32))
      ```
    * `def __len__(self)` e.g., the amount of samples (images, audios, ...)
* class SiameseNetworkDatasetDifficult(SiameseNetworkDataset) <br>
  init with additional args `(indice_tuples_genuine, indice_tuples_impostor)`
* class SoftMaxDatabase(SiameseNetworkDataset) <br>
  essentially, sequential selection of all images with random resampling
* class GMESoftMaxDatabaseTrain(SiameseNetworkDataset) <br>
  essentially, sequential selection of all images, subject by subject
* class LazyApproxGibbsSampler(SiameseNetworkDataset) <br>
  sequential selection of all images
  * one half of batch as GMESoftMaxDatabaseTrain sampler
  * other half random non-target trials (random selection, no resampling)
* class LFW_SiameseNetworkDataset(SiameseNetworkDataset) <br>
  preparation of data sampler for LFW dataset

Note: landmark segmentation in face recognition finds voice activity detection as its equivalent in speaker recognition.
For further details on pre-extracting landmarks in compatible format, see `landmarks.py`

## Installation
1) install miniconda for python 3
   > wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
   
   > bash Miniconda3-latest-Linux-x86_64.sh
2) Create and set-up environment, exemplary named `sidekit-env`
    > miniconda3/bin/conda create --name sidekit-env python=3.5 anaconda
    
    > source ~/miniconda3/bin/activate sidekit-env
    
    > conda install -c anthol sidekit
    
    > conda install mpi4py cython gcc
    
    > git clone  https://pelennor.univ-lemans.fr/Larcher/sidekit
    
    > cd sidekit
    
    > git checkout -b dev-anautsch origin/dev-anautsch
    
    > cd
    
    > rm -r ~/miniconda3/envs/sidekit-env/lib/python3.5/site-packages/sidekit
    
    > ln -s ~/sidekit ~/miniconda3/envs/sidekit-env/lib/python3.5/site-packages

3) Install PyTorch and Pyro
    > conda install numpy pyyaml mkl setuptools cmake gcc cffi
    
    > conda install -c soumith magma-cuda80
    
    > git clone --recursive https://github.com/pytorch/pytorch
    
    > cd pytorch
    
    > python setup.py install
    
    note: adjusting torch libraries, which were renamed for compability, but cannot be found henceforth
    
    > mv ~/miniconda3/envs/sidekit-env/lib/python3.5/site-packages/torch/_C.* mv ~/miniconda3/envs/sidekit-env/lib/python3.5/site-packages/torch/_C.so

    > mv ~/miniconda3/envs/sidekit-env/lib/python3.5/site-packages/torch/dl.* mv ~/miniconda3/envs/sidekit-env/lib/python3.5/site-packages/torch/dl.so
    
    > conda install -c soumith torchvision
    
    For Pyro:
    ```bash
    git clone https://github.com/uber/pyro.git
    cd pyro
    python setup.py install
    ```

4) prepare landmark detector, see: <br>
    https://github.com/davisking/dlib/blob/master/python_examples/cnn_face_detector.py
    > conda install boost

    > pip install dlib

5) (optional) fix libreadline (if broken)
    > conda remove --force readline

    > pip install readline

6) exit conda environment by
    > source deactivate

## Example
This example explicitly refers to the JHU cluster.

1) login to some other machine to host your screen session
   ```bash
   ssh b11
   ```
2) start screen session
   ```bash
   screen
   ```
3) request 1 GPU on a server node in the cluster
   ```bash
    qlogin -l 'hostname=b1[123456789]\*|c\*,gpu=1,arch=\*64\*,mem_free=25G,ram_free=25G' -now no
    ```
4) activate your favorite conda environment, so that Python libs can be found
   ```bash
   source ~/miniconda3/bin/activate sidekit-env
    ```

5) sanity-check whether or not a GPU is available, if not, maybe sth. went wront in the qlogin
   ```bash
    free-gpu
    ```

6) start Python experiment, pipe std-out and std-err logs to logfile, pass GPU to Python 
   ```bash
   CUDA_VISIBLE_DEVICES=\`free-gpu\` python example_baseline.py &> log-myexperiment-yymmdd
    ```

7) detach from screen, back to the machine you ssh'ed in step (1)
   ```bash
    ctrl a + d
    ```
8) enjoy the show
   ```bash
    tail -f log-myexperiment-yymmdd
    ```

9) after a few epochs this command could be useful depicting the validation EER and some prior debug information, as well as the # of the next epoch
   ```bash
   grep EER -B 9 -A 1 log-myexperiment-yymmdd
   ```
   
Note: keep in mind, the framework follows a save/load architecture. Between experiments (re-) move all unwanted deserialized intermediate results, such as trained model epochs. It might be helpful to extensively play with the `ConfigNetwork.modelname` parameter. An update of this framework could consider a rather hierarchical naming scheme.
