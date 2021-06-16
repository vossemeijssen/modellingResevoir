# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 13:06:14 2021

@author: Brian
"""

import json
import numpy as np
import math
import os
import matplotlib.pyplot as plt
import matplotlib

def moving_average(data, N):
    return np.convolve(data, np.ones(N)/N, mode='valid')

def load_run_data(path, filename):
    with open(filename) as f:
        run_data = json.load(f)
    return run_data

def read_all_runs(path, filename, sort='array_id'):
    ## Get list of subfolders:
    list_of_folder_names = next(os.walk(path))[1]
    for bad_folder in ['__pycache__', 'plots']:
        if bad_folder in list_of_folder_names:
            list_of_folder_names.remove(bad_folder)
    if sort == 'array_id':
        list_of_folder_names.sort(key=lambda folder_name: int(folder_name[7: folder_name[7:].find('_') + 7]))
    elif sort == 'sweep_value':
        list_of_folder_names.sort(key=lambda folder_name: float(folder_name[9 + folder_name[8:].find('_'):]))
    else:
        pass
    all_runs = []
    for run_dir in list_of_folder_names:
        dir_path = path + '/' + run_dir
        all_runs.append(load_run_data(dir_path, filename))
    
    return all_runs

def analyseWaveFront(wavefront_indices, W, L):
    wavefront_fft = np.fft.fft(wavefront_indices - np.mean(wavefront_indices))
    nr_waves = np.argmax(abs(wavefront_fft))
    wavefront_period = W/nr_waves
    extension = (max(wavefront_indices) - min(wavefront_indices))
    return (wavefront_period, extension)

dir_path = os.getcwd()
all_runs = read_all_runs(dir_path, 'wavefront.json', None)

extension_curves = []
plt.figure()
cmap = plt.cm.get_cmap('winter')
norm = matplotlib.colors.Normalize(vmin=0, vmax=len(all_runs)-1)

for i, run in enumerate(all_runs):
    extensions = []
    for run_at_t in run['wavefront']:
        wavefront_period, extension = analyseWaveFront(run_at_t, run['W'], run['L'])
        extensions.append(extension)
    
    # extension_curves.append(extension)
    avg = moving_average(extensions, 10)
    
    # plt.plot(extensions, label='Nr of waves: '+ str(run['kNum'][0]))
    plt.plot(avg, label='Nr of waves: '+ str(run['kNum'][0]), color=cmap(norm(i)))

plt.legend()
plt.xlabel('Time step')
plt.ylabel('Wave extension')
plt.savefig('Wave_extensions.svg')


    
















