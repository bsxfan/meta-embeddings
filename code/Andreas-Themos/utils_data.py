from FaceDatasets import get_data_loader


__author__ = "Andreas Nautsch, Themos Stafylakis"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Development"
__docformat__ = 'reStructuredText'
__credits__ = ["Niko Brümmer, Adrian Bulat"]


def data_loader(database_dir, shuffle=True, train_add_noise=False):
    # TODO still Face specific
    if not isinstance(database_dir, str): # isinstance(database_dir, DataLoader) or not is :
        dataloader = database_dir
    else:
        dataloader = get_data_loader(database_dir, shuffle=shuffle, train_add_noise=train_add_noise)
    return dataloader