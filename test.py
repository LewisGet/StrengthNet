#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import time 
import numpy as np
from tqdm import tqdm
import argparse
import fnmatch

from statistics import mean 

import tensorflow as tf
from tensorflow import keras
from model import CNN_BLSTM

import utils   
import random
import config
 

def find_files(root_dir, query="*.wav", include_root_dir=True):
    """Find files recursively.

    Args:
        root_dir (str): Root root_dir to find.
        query (str): Query to find.
        include_root_dir (bool): If False, root_dir name is not included.

    Returns:
        list: List of found filenames.

    """
    files = []
    for root, dirnames, filenames in os.walk(root_dir, followlinks=True):
        for filename in fnmatch.filter(filenames, query):
            files.append(os.path.join(root, filename))
    if not include_root_dir:
        files = [file_.replace(root_dir + "/", "") for file_ in files]

    return files

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate custom waveform files using pretrained MOSnet.")
    parser.add_argument("--rootdir", default=None, type=str,
                        help="rootdir of the waveforms to be evaluated")
    parser.add_argument("--pretrained_model", default="./output/strengthnet.h5", type=str,
                        help="pretrained model file")
    args = parser.parse_args()

    #### tensorflow & gpu settings ####

    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

    tf.debugging.set_log_device_placement(False)
    # set memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    ###################################
    
    # find waveform files
    wavfiles = sorted(find_files(args.rootdir, "*.wav"))
    
    # init model
    print("Loading model weights")
    StrengthNet = CNN_BLSTM()
    model = StrengthNet.build()
    model.load_weights(args.pretrained_model)

    # evaluation
    print("Start evaluating {} waveforms...".format(len(wavfiles)))

    for wavfile in tqdm(wavfiles):
        
        # spectrogram
        mel_sgram = utils.get_melspectrograms(wavfile)
        timestep = mel_sgram.shape[0]
        mel_sgram = np.reshape(mel_sgram,(1, timestep, utils.n_mels))
        # make prediction
        Strength_score, Frame_score, emo_class = model.predict(mel_sgram, verbose=0, batch_size=1)

        labels = []
        for emo_class_score in emo_class:
            emo_class_score = emo_class_score.tolist()
            max_value = max(emo_class_score)
            max_index = emo_class_score.index(max_value)
            labels.append(config.emo_label[max_index])

        print(wavfile, Strength_score, labels)
       

if __name__ == '__main__':
    main()
