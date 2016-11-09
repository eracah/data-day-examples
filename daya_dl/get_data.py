
# coding: utf-8

# In[1]:

import h5py

import numpy as np

from matplotlib import pyplot as plt
import pickle
import copy


# In[2]:

def get_data(num_each=500):
    #specify how many of each event we want

    # pick a sample background file
    sample_file='/project/projectdirs/dasrepo/dayabay_data/new_1.6TB_data/recon.Neutrino.0021222.Physics.EH2-Merged.P14A-P._0048.h5'

    #open up file and get ims and labels
    bg_file = h5py.File(sample_file)


    # labels of the background ims (3,4, or 5)(muon, flasher, other)
    bg_labels = bg_file['class'][:]

    bg_ims = bg_file['charge'][:]

    #print bg_ims.shape

    bg_inds = np.arange(bg_labels.shape[0]).reshape(bg_labels.shape[0],1)

    muon_ind = list(bg_inds[bg_labels==3.][:num_each])
    flasher_ind = list(bg_inds[bg_labels==4.][:num_each])
    other_ind = list(bg_inds[bg_labels==5.][:num_each])

    bg_ims = np.vstack((
                        bg_ims[muon_ind],
                        bg_ims[flasher_ind],
                        bg_ims[other_ind]))
    bg_labels = np.vstack((
                           bg_labels[muon_ind], 
                           bg_labels[flasher_ind], 
                           bg_labels[other_ind]))

    bg_ims = bg_ims.reshape(bg_ims.shape[0], 8, 24)

    # get ibd pairs

    #make images
    ibd_pair_path = '/project/projectdirs/dasrepo/ibd_pairs/all_pairs.h5'

    sig_file = h5py.File(ibd_pair_path)

    sig_raw = sig_file['ibd_pair_data']

    ibd_charge_pairs = sig_raw[:num_each,:4*8*24].reshape(num_each,4,8,24)[:,[0,2]]

    sig_ims = ibd_charge_pairs.reshape(num_each *2 , 8, 24)

    # make labels (every other im is an ibd_prompt)
    sig_labels = np.ones((sig_ims.shape[0], 1))

    sig_inds = np.arange(sig_ims.shape[0])

    sig_labels[sig_inds % 2 != 0] = 2.



    # combine bg and signal data and labels
    raw_data = np.vstack((bg_ims, sig_ims))

    labels = np.vstack((bg_labels, sig_labels))

    #shuffle the data
    rng = np.random.RandomState(11)

    inds = np.arange(labels.shape[0])

    rng.shuffle(inds)

    raw_data = raw_data[inds]

    labels = labels[inds].reshape(labels.shape[0],)
    return raw_data, labels


# In[7]:

#! ipython nbconvert --to script get_data.ipynb


# In[ ]:



