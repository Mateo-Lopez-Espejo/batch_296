#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:40:25 2017

@author: mle
"""
''' Collection of tools to treat downsampled SSA data. imports from matlab,
    organizes into a 3D matrix with dim0= beep number, dim1=response at a time point, 
    dim2= stimulus identity i.e. freq A or B, deviant, standard or onset. 
    Plot relevant SSA indexes for either single neurons or populations '''

import baphy_utils as bup
import numpy as np
import os
import matplotlib.pyplot as plt
import math as mt
import scipy.signal as sig
from scipy.optimize import curve_fit
from matplotlib import colors
import tkinter as tk
from tkinter import filedialog
from inspect import signature
import pandas as pd
from itertools import product

# defines index in the array third dimension, that correspond to
# the tone (frequency) of the stimulus
# and wheather its Starndard, Deviant or Onset (S D O)
# note that this values should not be changed, since it would damage the code output.

fAD=0; fAS=1; fAO=2
fBD=3; fBS=4; fBO=5
fAp=0; fBp=2
ToneIDList = ['fAD','fAS','fAO','fBD', 'fBS', 'fBO']
ToneIDDict = dict((name, eval(name)) for name in ['fAD', 'fAS', 'fAO', 'fBD', 'fBS', 'fBO'])

pooledIdList = ['fA Pool', 'fAO', 'fB Pool', 'fBO']

Jon = 1 ; Joff = 0  #jitter on, jitter off default index for array location.
relsilid = 0 ; abssilid = 1 # relative silence, absolute silence default indexes fro array location.
#full batch dir with envelope and none files
batch = '/auto/data/code/nems_in_cache/batch296/'
# Single rec example, with 4 recordings (two pairs of jitter/nonJitter as 2 diferent sets of freqs)
# First isolated cell. File has evnelope.
multytestfile  = '/auto/data/code/nems_in_cache/batch296/gus019c-a1_b296_envelope_fs1000'

singletestfile = '/auto/data/code/nems_in_cache/batch296/gus021c-b1_b296_envelope_fs1000'

#list of known outliers basenames
outliers = ['gus025b-c1_b296_envelope_fs1000.mat',]


def single_neuron_plot (Raster_Matrix, Baseline=0, Pip_start=0, Pip_End=0, Constrain_start=0,
                        Constrain_end=0, Frequency=None, Title=None, idx=None, tone_freqs=None):

    #data imput is a 3D matrix with dim0 = repetition(tone number), dim1 =time, dim2 = tone id
    Pip_startIdx = Pip_start
    Pip_endIDX = Pip_End
    title = Title
    resp = Raster_Matrix

    if Frequency !=None:
        starttime = Pip_start/Frequency
        endtime= Pip_End/Frequency
        DwnSmplFacotr = int(np.ceil(Frequency / 200))
        timelabel = 'Seconds'
    elif Frequency == None:
        DwnSmplFacotr = 5
        starttime = Pip_startIdx
        endtime = Pip_endIDX
        timelabel = 'time(no set freq)'
    else: print ('error: frequency should be either None or an int')

    psth = np.nanmean(resp,0)
    DSresp = sig.decimate(psth,DwnSmplFacotr,axis=0, zero_phase=True )


    #color management for the rasters
    bounds = [0,0.5,1]
    FAcmap = colors.ListedColormap(['white', 'firebrick'])
    FAnorm = colors.BoundaryNorm(bounds, FAcmap.N)
    FBcmap = colors.ListedColormap(['white', 'navy'])
    FBnorm = colors.BoundaryNorm(bounds, FBcmap.N)

    # Takes away all aditional NaN padding form matrixes to allow
    # cleaner raster without white space
    rasterlist = list()
    for PipIdent in range(resp.shape[2]):
        trialnum =resp[:,0,PipIdent]
        trialnum = len(trialnum[~np.isnan(trialnum)])
        raster = resp[0:trialnum,0:resp.shape[1],PipIdent]
        rasterlist.append(raster)

    # 3d matrix, dim0 value on a given time point, dim1 stimulus identity
    # and dim2 average(0) or stdev (1)
    avg = 0; stdev = 1
    PSTH_Matrix = np.empty([resp.shape[1],resp.shape[2],2])
    for PipIdent in range(resp.shape[2]):
        PSTH_Matrix[:,PipIdent,avg] = np.nanmean(resp[:,:,PipIdent],0)
        PSTH_Matrix[:,PipIdent,stdev] = np.nanstd(resp[:,:,PipIdent],0)

    # defines x axis for correct timing label, and error shadows
    if Frequency == None:
        x = np.linspace(0,(resp.shape[1])-1,DSresp.shape[0], endpoint=False)
    else:
        x = np.linspace(0,resp.shape[1]/Frequency,DSresp.shape[0], endpoint=False)


    #Defines y limit for equally scalled plotting for all tone paradigms
    ylim = np.nanmax(DSresp[:,[0,1,3,4]])


    plt.figure()
    plt.suptitle(title)
    #plots frequency A responses on left row of figure

        #raster A deviant
    RstrAD=plt.subplot(4,2,1)
    d = rasterlist[fAD]
    RstrAD.axvline(x = Pip_startIdx)
    RstrAD.axvline(x = Pip_endIDX)
    RstrAD.imshow(d, aspect='auto', origin='lower',
                  cmap = FAcmap, norm = FAnorm )
    RstrAD.set_ylabel('Tone number')
    plt.tick_params(axis='x',labelbottom='off')


        #raster B standard
    RstrBS=plt.subplot(4,2,3)
    d = rasterlist[fBS]
    RstrBS.axvline(x = Pip_startIdx)
    RstrBS.axvline(x = Pip_endIDX)
    RstrBS.imshow(d, aspect='auto', origin='lower',
                  cmap = FBcmap, norm = FBnorm )
    RstrBS.set_ylabel('Tone number')
    plt.tick_params(axis='x', labelbottom='off')

    # PSTH AD BS]
    PSTH_AD_BS=plt.subplot(4,2,5)
    d = DSresp[:,[fAD, fBS]] - Baseline
    PSTH_AD_BS.axvline(x = starttime)
    PSTH_AD_BS.axvline(x = endtime)
    PSTH_AD_BS.axhline(y=Baseline)
    PSTH_AD_BS.plot(x, d[:, 0],color='firebrick', ls=':')
    PSTH_AD_BS.plot(x, d[:, 1],color='navy')
    PSTH_AD_BS.set_xlabel(timelabel)
    PSTH_AD_BS.set_ylabel('spike rate')
    plt.ylim(-0.01,ylim)


    # Plot frequency B respones on right row of figure

        # raster B deviant
    RstrBD = plt.subplot(4, 2, 2)
    d = rasterlist[fBD]
    RstrBD.axvline(x=Pip_startIdx)
    RstrBD.axvline(x=Pip_endIDX)
    RstrBD.imshow(d, aspect='auto', origin='lower',
                  cmap=FBcmap, norm=FBnorm)
    plt.tick_params(axis='x', labelbottom='off')

    # raster A standard
    RstrAS=plt.subplot(4,2,4)
    d = rasterlist[fAS]
    RstrAS.axvline(x = Pip_startIdx)
    RstrAS.axvline(x = Pip_endIDX)
    RstrAS.imshow(d, aspect='auto', origin='lower',
                  cmap = FAcmap, norm = FAnorm )
    plt.tick_params(axis='x', labelbottom='off')


    # PSTH BD AS
    PSTH_BD_AS=plt.subplot(4,2,6)
    d = DSresp[:,[fBD, fAS]] - Baseline
    PSTH_BD_AS.axvline(x = starttime)
    PSTH_BD_AS.axvline(x = endtime)
    PSTH_BD_AS.axhline(y=Baseline)
    PSTH_BD_AS.plot(x, d[:, 0], color='navy', ls=':')
    PSTH_BD_AS.plot(x, d[:, 1], color='firebrick')
    PSTH_BD_AS.set_xlabel(timelabel)
    plt.ylim(-0.01, ylim)

    # calculate and plot the composite stndard or deviant PSTH independen of frequency
    # as an average of either deviants or standars for both frequenies

    StandarMean = np.nanmean(DSresp[:, [fAS, fBS]], 1) - Baseline
    DeviantMean = np.nanmean(DSresp[:, [fAD, fBD]], 1) - Baseline

    Cell_PSTH = plt.subplot(4, 2, 7)
    Cell_PSTH.axvline(x=starttime)
    Cell_PSTH.axvline(x=endtime)
    Cell_PSTH.axhline(y=Baseline)
    Cell_PSTH.plot(x, StandarMean, 'k')
    Cell_PSTH.plot(x, DeviantMean, 'g:')
    plt.ylim(-0.01, ylim)
    Cell_PSTH.set_xlabel(timelabel)
    Cell_PSTH.set_ylabel('spike rate')

    if Frequency !=None:
        cons_starttime = Constrain_start/Frequency
        cons_endtime= Constrain_end/Frequency
        cons_Label = 'SI calc constraied between {} and {}'.format(cons_starttime,cons_endtime)
    elif Frequency == None:
        cons_starttime = Pip_startIdx
        cons_endtime = Pip_endIDX
        cons_Label = 'SI calculation constraied between {} and {} seconds'.format(cons_starttime,cons_endtime)
    else:cons_Label=''

    try:
        text = '{}\n' \
               '\n' \
               '{} hz (Red), SI: {} \n' \
               '{} hz (Blue), SI: {} \n' \
               'cell total SI: {}'.format(
            cons_Label,
            tone_freqs[0], idx['SSA_Idx_FA'],
            tone_freqs[1], idx['SSA_Idx_FB'],
            idx['Cell_SSA_Idx'])

    except: text = 'no info on Frequencies or SI'

    textbox = plt.subplot(4, 2, 8)
    textbox.text(0.1, 0.1,text)
    plt.tick_params(axis='both', labelbottom='off',
                    labelleft='off', bottom='off',
                    left='off')
    plt.show()


def SSAidxCalc (folded_dict, tone_constrained='start', baseline_substracction = False, FRo_rate=False ):

    #todo: add standar deviation or any other convenient dispersion metric
    FR_dict = calc_FR_FRo(folded_dict, tone_constrained=tone_constrained, baseline_substraction=baseline_substracction,
                          FRo_rate=FRo_rate)
    AreaUnder = FR_dict['AreaUnder']

    AdevAct = np.nanmean(AreaUnder[:,fAD])
    AstdAct = np.nanmean(AreaUnder[:,fAS])
    BdevAct = np.nanmean(AreaUnder[:,fBD])
    BstdAct = np.nanmean(AreaUnder[:,fBS])

    SIFA=(AdevAct-AstdAct)/(AdevAct+AstdAct) # SSA index for frequecy A
    SIFB=(BdevAct-BstdAct)/(BdevAct+BstdAct) # SSA index for frequency B

    SIN=(AdevAct+BdevAct-AstdAct-BstdAct)/(
         AdevAct+BdevAct+AstdAct+BstdAct)    # SSA index for neuron

    # If any of the indexse is NAN due to divission by cero, this most likely means is a non responsive cell, therefore
    # all indexes are set to NAN to avoid considering these outliers in later calculations.
    idx_list=[SIFA,SIFB,SIN]
    for ii in idx_list:
        if mt.isnan(ii):
            idx_list = np.ndarray([3])
            idx_list[:] = np.nan
            print('division by 0, setting all indexes to NAN')
            break
        else: continue

    index_dict = dict(zip(['SSA_Idx_FA','SSA_Idx_FB','Cell_SSA_Idx'],
                          idx_list))
    return index_dict

def batch_SSA_IDX (dirPath, tone_constrained='start', baseline_substracction = False, FRo_rate=False, isol_threshold=None):
    ''' calls over all the files in a directory in an organized maner. it is important to notice that the exported mat files
    are one dimensional structs where the length along the dimension correspond to "blocks" which vary in jitter (on or off)
    but can also be repeticions of a experiment on the same cell using a different pair of tones.
    The functions returns an 3d array
    dim0(# jitter pairs) corresponding to every pair of jitter/unjitter experiments given a set frequecy pair
    dim1(3) corespond to the type of ssa Index, whether for frequency a, frequency b or cell SI
    dim2(2) correspond to unjittered (0) or jittered(1) experiments'''

    pathlist = file_listing(dirPath, None, filter_outliers=True, stim_type='enve')

    SSA_idx_list = list()
    used_file_list = list()
    used_freqPair_list = list()
    for pp in pathlist:

        loadedMatFile = bup.load_baphy_ssa(pp)
        block_counter = 0
        block_num = len(loadedMatFile)

        print('working on file {} with {} blocks'.format(os.path.basename(pp),block_num))

        # checks for all frequency pairs for all the blocks within a file
        freq_pairs = list()
        for block in loadedMatFile:
            freq_pairs.append(block['Frequencies'])
        unique_freq = [list(x) for x in set(tuple(x) for x in freq_pairs)] # aka jitter pairs

        # define single block idx array, dim0 (unique_frequ) corresponds  to number of different pair frequencies ,
        # dim1(3) correspond to idx type, dim2 (2) correspond to no jitter / jitter

        for jitter_pair in unique_freq:

            single_pair_array = np.ndarray([3, 2])
            single_pair_array[:] = np.NAN

            for block in loadedMatFile:

                # checks if the block belongs to the ongoing pair
                if block['Frequencies'] == jitter_pair:
                    pass
                elif block['Frequencies'] != jitter_pair:
                    continue
                else:
                    print('theres in an error in how jitter pairs are being compared')
                    break

                # check for isolation percentage
                if isol_threshold == None:
                    print('working on block #{}'.format(block_counter))
                    pass
                elif 0 < isol_threshold <= 100:
                    if block['isolation'] >= isol_threshold:
                        print('working on block #{}'.format(block_counter))
                        pass
                    elif block['isolation'] < isol_threshold:
                        print('block {} skiped due to isolation under selected threshold'.format(block_counter))
                        block_counter = block_counter + 1
                        continue
                    else:
                        print('istolation not defines in loaded .mat file block {}, skipping block'.
                              format(block_counter))
                        block_counter = block_counter + 1
                        continue
                else:
                    print('isol_threshold has to be between 0 and 100')
                    break

                # checks stimfmt to apply slicing or just rearrange the already stacked response
                if block['stimfmt'] == 'envelope':
                    print('splicing')
                    folded_dict = fold_tones(block)
                elif block['stimfmt'] == 'none':
                    print('deprecation warning: workinf on file with stimfmt = none')
                    stacked_resp = np.swapaxes(block['resp'],0,1)
                else:
                    print('Error, stimfmt should be either "envelope" or "none"')

                # Calculate indexes by calling SSAidxCalc
                print('calculating all SSA indexes')
                calc_idx = SSAidxCalc (folded_dict, tone_constrained=tone_constrained,
                                       baseline_substracction = baseline_substracction, FRo_rate=FRo_rate)
                SIFA = calc_idx['SSA_Idx_FA']
                SIFB = calc_idx['SSA_Idx_FB']
                SIN = calc_idx['Cell_SSA_Idx']

                if block['stimfmt'] == 'envelope':
                    if folded_dict['Jitter'] == 'On ':
                        single_pair_array[0, Jon ] = SIFA
                        single_pair_array[1, Jon ] = SIFB
                        single_pair_array[2, Jon ] = SIN
                    elif folded_dict['Jitter'] == 'Off':
                        single_pair_array[0, Joff] = SIFA
                        single_pair_array[1, Joff] = SIFB
                        single_pair_array[2, Joff] = SIN
                    else: print('something wrong with jitter')
                elif block['stimfmt'] == 'none':
                    single_pair_array[0, Joff] = SIFA
                    single_pair_array[1, Joff] = SIFB
                    single_pair_array[2, Joff] = SIN
                else: print('error: stimfmt should be either "none" or "envelope" ')

                block_counter = block_counter + 1

            # Checks if the generated pair is empty, if so, does nothing, if one element is found, stores by appending.
            for idx in np.nditer(single_pair_array):
                if np.isnan(idx):
                    continue
                elif ~np.isnan(idx):
                    SSA_idx_list.append(single_pair_array) # Appendds pair
                    used_file_list.append(os.path.basename(pp))
                    used_freqPair_list.append(jitter_pair)
                    break
                else:
                    print('elements jitter-pair ssaidx array shoudl be either NAN or float')


    print('stacking single file arrays')
    SSA_idx_array = np.stack(SSA_idx_list,axis=0)


    output = dict()
    output['index'] = SSA_idx_array
    output['filename'] = used_file_list
    output['freq_pair'] = used_freqPair_list
    return output

def population_SI_plot(ssa_idx_array, stdev_array=None, title= None):
    NoJitter = ssa_idx_array[:, :, Joff]
    NoJitter = NoJitter[~np.isnan(NoJitter)]
    NoJitter = NoJitter.reshape((int(NoJitter.size/3)),3)
    NJSIFA = NoJitter[:, 0]
    NJSIFB = NoJitter[:, 1]
    NJSIN = NoJitter[:, 2]

    Jitter = ssa_idx_array[:, :, Jon]
    Jitter = Jitter[~np.isnan(Jitter)]
    Jitter = Jitter.reshape((int(Jitter.size / 3)), 3)
    JSIFA = Jitter[:, 0]
    JSIFB = Jitter[:, 1]
    JSIN = Jitter[:, 2]


    plt.figure()
    if title == None:
        pass
    else:
        plt.suptitle(title)

    # Scatter plot of the frequency specific SI for each neuron
    # When jitter off

    plt.subplot(221)
    plt.scatter(NJSIFA, NJSIFB, color='firebrick')

    plt.axis([-1,1,-1,1])
    plt.xlabel('SI freq A')
    plt.ylabel('SI freq B')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title('No Jitter Control SI')
    plt.grid(True)

    # Scatter plot of the frequency specific SI for each neuron
    # When jitter on
    plt.subplot(223)
    plt.scatter(JSIFA, JSIFB, color='navy')

    plt.axis([-1, 1, -1, 1])
    plt.xlabel('SI freq A')
    plt.ylabel('SI freq B')
    plt.axhline(0, color='black')
    plt.axvline(0, color='black')
    plt.title('Jitter treatment SI')
    plt.grid(True)

    # histogram of the neuron specific SI

    plt.subplot(222)
    plt.hist(JSIN, alpha=0.5, label='Jitter', facecolor='navy')
    plt.hist(NJSIN, alpha=0.5, label= 'No Jitter', facecolor='firebrick')
    plt.axvline(np.mean(JSIN), color='navy')
    plt.axvline(np.mean(NJSIN), color='firebrick')
    plt.legend(loc='upper right')
    plt.xlabel('SI value')
    plt.ylabel('Number of neurons')
    plt.title('Cell SI')

    # individual pairs and boxplots
    plt.subplot(224)

    box = plt.boxplot([NJSIN,JSIN], patch_artist= True, vert = 0)

    color = ['firebrick', 'navy']
    for patch, color in zip(box['boxes'], color):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    plt.plot(np.swapaxes(ssa_idx_array[:, 2, :], 0, 1), [1, 2],
             color='black', marker='o')

    plt.yticks([1,2],['No jitter', 'Jitter'], color=color)
    plt.gca().invert_yaxis()

    plt.show()


def fold_tones (Matlab_imported_dict):
    ''' input is a single dictionary from the list
    imported using baphy_utils load_baphy_SSA
    it slices responses based on the stimulation envelopes
    and asigns them to their identities, namely which frequency
    and wheter standard deviant or onset.
    returns a dictionary which contains the sliced pips in a 3D matrix
    dim0= tial, dim1= response, dim2= tone type
    tone type: 0 -> Freq A deviant: 1-> Freq A Starndard 2 -> Freq A Onset
                  3 -> Freq B deviant: 4-> Freq B Starndard 5 -> Freq B Onset
    Also count the time between any pair of tones or pairs of tones of the same type, the organizes'''

    # todo account for possible diference between stim freq and response freq
    # todo maybe split silence calculation into a new function to unload charge when silences are not required.

    MatlabDict = Matlab_imported_dict

    resp = np.squeeze(MatlabDict['resp'])
    resp = np.swapaxes(resp,0,1)
    env = MatlabDict['stim']
    env = env.swapaxes(0, 2)
    envelopefreq = MatlabDict['stimf']
    PipDuration = MatlabDict['PipDuration']

    # sets Slice length depending of interval type and tone duration
    piplength = int(np.floor(PipDuration * envelopefreq))
    try:
        if MatlabDict['Jitter']=='On ':
            SliceLength = int(np.floor((PipDuration + MatlabDict['MinInterval']*2) * envelopefreq))
            PrePipSilence = int(np.floor(MatlabDict['MinInterval'] * envelopefreq))
        elif MatlabDict['Jitter']=='Off':
            SliceLength = int(np.floor((PipDuration + MatlabDict['PipInterval']) * envelopefreq))
            PrePipSilence = int(np.floor((MatlabDict['PipInterval']/2) * envelopefreq))
        else: print('Error: undefined if jitter On or Off ')
    except: #if no jitter param asumes jitter off
        SliceLength = int(np.floor((PipDuration + MatlabDict['PipInterval']) * envelopefreq))
        PrePipSilence = int(np.floor((MatlabDict['PipInterval'] / 2) * envelopefreq))

    # lists of the trial slices (tones) depending on the identity of the tone
    F1Onset = [] ; F1Standar = [] ; F1Deviant = []
    F2Onset = [] ; F2Standar = [] ; F2Deviant = []

    # list of the preceding silences depending on the identity of the tone
    F1OSilence = [] ; F1SSilence = [] ; F1DSilence = []
    F2OSilence = [] ; F2SSilence = [] ; F2DSilence = []

    # List of preceding silences dependent on the frequency of the tone (independent  of deviant, starndar, onset)
    freq_sil_list =[[],[],[],[],[],[]]

    # list of the preceding silences no independent of the identity of the tone
    # Note its really an array of 2 dimensions, but im to lazy to initialize a
    # proper nan ndarray with the right shape
    AbsSilenceList = [[],[],[],[],[],[]]

    maxtrials = resp.shape[1]
    halftrial = int(mt.ceil(maxtrials/2))
    trialLen = resp.shape[0]

    for trialcounter in range(maxtrials):
        onset = True
        F1PrevPipEnd = 0 ; F2PrevPipEnd = 0
        F1prevSilence = trialLen ; F2prevSilence = trialLen

        # checks wether envelope 1 has more tones than envelope 2 and sets it as standard
        # Or deviant respectively.
        if np.sum(env[:,trialcounter,0],0) > np.sum(env[:,trialcounter,1],0):
            F1IdStd = True # sets if F1 is the standard tone in the trial
            F2IdStd = False
        elif np.sum(env[:,trialcounter,0],0) < np.sum(env[:,trialcounter,1],0):
            F1IdStd = False
            F2IdStd = True
        else: F1IdStd = True ; F2IdStd = False ; print('rates are equal for both frequencies')

        for ii in range(env.shape[0]-1):
            F1first = env[ii,trialcounter,0]
            F1second = env[ii+1,trialcounter,0]
            F2first = env[ii, trialcounter, 1]
            F2second = env[ii + 1, trialcounter, 1]

            # find start and end of a tone, adds flaking silences
            # and distance since the preceding tones
            # Holds index for later slicing.
            if F1first - F1second == 0 and F2first - F2second == 0:
                continue
            elif F2first-F2second == 1 or F1first-F1second == 1:
                continue
            elif F1first-F1second == -1 and not F2first-F2second == -1:
                PipSlice = resp[(ii - PrePipSilence) : (ii - PrePipSilence + SliceLength), trialcounter]
                F1prevSilence = ii - F1PrevPipEnd
                F1PrevPipEnd = ii + piplength
                F1 = True
            elif F2first-F2second == -1 and not F1first-F1second == -1:
                PipSlice = resp[(ii - PrePipSilence) : (ii - PrePipSilence + SliceLength), trialcounter]
                F2prevSilence = ii - F2PrevPipEnd
                F2PrevPipEnd = ii + piplength
                F1 = False
            else: print('Error: envelopes should be booleans')

            # defines absolute pevious silence based on tone specific previous silence

            AbsPrevSilence = min([F1prevSilence, F2prevSilence])

            if onset == True:
                if F1 == True:
                    F1Onset.append(PipSlice)
                    F1OSilence.append(F1prevSilence)
                    AbsSilenceList[fAO].append(AbsPrevSilence)
                    onset = False
                elif F1 == False:
                    F2Onset.append(PipSlice)
                    F2OSilence.append(F2prevSilence)
                    AbsSilenceList[fBO].append(AbsPrevSilence)
                    onset = False

            elif onset == False:
                if F1 == True:
                    if F1IdStd == True:
                        F1Standar.append(PipSlice)
                        F1SSilence.append(F1prevSilence)
                        AbsSilenceList[fAS].append(AbsPrevSilence)
                    elif F1IdStd == False:
                        F1Deviant.append(PipSlice)
                        F1DSilence.append(F1prevSilence)
                        AbsSilenceList[fAD].append(AbsPrevSilence)

                elif F1 == False:
                    if F2IdStd == False:
                        F2Deviant.append(PipSlice)
                        F2DSilence.append(F2prevSilence)
                        AbsSilenceList[fBD].append(AbsPrevSilence)
                    elif F2IdStd == True:
                        F2Standar.append(PipSlice)
                        F2SSilence.append(F2prevSilence)
                        AbsSilenceList[fBS].append(AbsPrevSilence)

    #organizes list of list into 2d arrays (slice, time) for each of the tone paradigms

    F1Onset = np.asarray(F1Onset) ; F1Standar = np.asarray(F1Standar) ; F1Deviant = np.asarray(F1Deviant)
    F2Onset = np.asarray(F2Onset) ; F2Standar = np.asarray(F2Standar) ; F2Deviant = np.asarray(F2Deviant)

    SliceArrayList = [F1Deviant, F1Standar, F1Onset,
                  F2Deviant, F2Standar, F2Onset]

    RelSilenceList = [F1DSilence, F1SSilence, F1OSilence,
                    F2DSilence, F2SSilence, F2OSilence]

    # creates a 3d NaN array of shape(slice, slice length, tone id)
    # so when the 2d slice arrays are stacked
    # any diference in number of slices is padded with NaN

    dimlen=list()
    for ss in SliceArrayList:
        dimlen.append(ss.shape[0])
    maxlen = max(dimlen)
    responses = np.empty((maxlen,SliceLength,6)) ; responses[:] = np.nan
    x = 0
    # place each 2d slice array in the proper 3rd dim index of the NaN array.
    # This is list order sensitive, which is suboptimal
    for ToneType in SliceArrayList:
        responses[:ToneType.shape[0],:ToneType.shape[1],x] = ToneType
        x = x+1

    #creates a 3d Nan Array of shape (Slice, relative or absolute(2), tone id)
    Silences = np.empty((maxlen,2,6)) ; Silences[:] = np.nan
    # parse the relative  silences into dim2 = 0 and the right tone id
    x = 0
    for relativeSilence in RelSilenceList:
        Silences[:len(relativeSilence),relsilid,x] = relativeSilence
        x = x+1
    # parse the absolute silences into dim2 = 1 and the right tone id
    x = 0
    for absoluteSilence in AbsSilenceList:
        Silences[:len(absoluteSilence),abssilid, x] = absoluteSilence
        x = x + 1



    PreStimSilence = MatlabDict['prestim']*envelopefreq
    baseline = np.nanmean(resp[:PreStimSilence,:])
    folded_dict = dict.fromkeys(['resp', 'Baseline',
                                   'PipStart','PipEnd','SliceLen','Jitter'
                                   'respf'])
    folded_dict['resp'] = responses
    folded_dict['Baseline'] = baseline
    folded_dict['PipStart'] = PrePipSilence
    folded_dict['PipEnd'] = int(PrePipSilence + (PipDuration*envelopefreq))
    folded_dict['SliceLen'] = SliceLength
    try:
        folded_dict['Jitter'] = MatlabDict['Jitter']
    except:
        folded_dict['Jitter'] = 'Off'
    folded_dict['respf'] = MatlabDict['respf']
    folded_dict['silences'] = Silences
    return folded_dict

def path_plot (PathIn=None, FRo_rate=False, std_dev_pooling=False, force_asymptote=False, silence_type='rel',
               elements_per_bin=False,  tone_constrained='start', baseline_substraction=False, isol_threshold=None):
    if PathIn == None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename()
    else: path = PathIn

    if os.path.isdir(path):
        IDX = batch_SSA_IDX(dirPath = path, tone_constrained=tone_constrained, baseline_substracction = False,
                            FRo_rate=FRo_rate, isol_threshold=isol_threshold)
        population_SI_plot(IDX['index'], None, path)

        FIT = batch_adaptation_fit(dirPath = path, FRo_rate=FRo_rate, std_dev_pooling=std_dev_pooling,
                                   force_asymptote=force_asymptote, silence_type=silence_type,
                                   elements_per_bin=elements_per_bin,  tone_constrained=tone_constrained,
                                   baseline_substraction=baseline_substraction, isol_threshold=isol_threshold)
        population_fit_plot(FIT, path, std_dev_pooling = std_dev_pooling)
        ultimate_dict = dict(['idx','fit'])
        ultimate_dict['idx'] = IDX
        ultimate_dict['fit'] = FIT

        print('behold the ultimate dict!!')
        return ultimate_dict

    else:
    #elif os.path.isfile(path):
        loadedMat = bup.load_baphy_ssa(path)
        basename = os.path.basename(path)


        if loadedMat[0]['stimfmt'] == 'envelope':
            print ('working on file {}, containing {} blocks'.format(os.path.basename(path),len(loadedMat)))

            #plots one figurefor jitter on and jitter off each
            block_counter=0
            for block in loadedMat:

                #do I really need to acount for isolation for single cell plots??
                if isol_threshold == None:
                    print('working on block #{}'.format(block_counter))
                    pass
                elif 0 <= isol_threshold <= 100:
                    if block['isolation'] >= isol_threshold:
                        print('working on block #{}'.format(block_counter))
                        pass
                    elif block['isolation'] < isol_threshold:
                        print ('block {} skiped due to isolation under selected threshold'.format(block_counter))
                        block_counter = block_counter + 1
                        continue
                    else:
                        print('istolation not defines in loaded .mat file block {}, skipping block'.
                              format(block_counter))
                        block_counter = block_counter + 1
                        continue
                else:
                    print('isol_threshold has to be between 0 and 100'); break

                print ('splicing block #{}'.format(block_counter))
                folded_dict = fold_tones(block)

                #checks if baseline is to be substaracted and parses the rigth values to do so
                if baseline_substraction == True:
                    baseline = folded_dict['Baseline']
                elif baseline_substraction == False:
                    baseline = 0
                else: print('baseline_substraction shoudl be True or False')

                # checks if the tone response integrations is to be constrained to the tone, until the end of the
                # slice, or not at all. Parses the rigth values to do so.
                if tone_constrained == 'all':
                    constrain_start = folded_dict['PipStart']
                    constrain_end = folded_dict['PipEnd']
                elif tone_constrained == 'no':
                    constrain_start = 0
                    constrain_end = folded_dict['SliceLen']
                elif tone_constrained == 'start':
                    constrain_start = folded_dict['PipStart']
                    constrain_end = folded_dict['SliceLen']
                else:
                    print("Invalid Tone_constrained Parameter. choose either 'all', 'start' or 'no' (default)")

                # checks what type of preceding silence is to be used for adaptation calculations
                # parses the right keyword for adaptation_fit function to use
                if silence_type == 'abs':
                    sil_type = 'abs'
                elif silence_type   == 'rel':
                    sil_type = 'rel'
                else: print("silence_type for adaptation calculation should be either 'abs' to acount for the silence "
                            "since any preceding tone types; or 'rel' to acount only for the silence since the preceding "
                            "tone of the same type"); break

                #parces other important data
                matrix = folded_dict['resp']
                Pip_start = folded_dict['PipStart']
                Pip_end = folded_dict['PipEnd']
                freq = folded_dict['respf']
                title ='{0}, Jitter {1}'.format(basename, folded_dict['Jitter'])
                tone_freqs = block['Frequencies']

                print('calculating ssa indexes')
                idx = SSAidxCalc (folded_dict, tone_constrained=tone_constrained,
                                  baseline_substracction=baseline_substraction, FRo_rate=FRo_rate)

                print('plotting raster and PSTH')
                single_neuron_plot(matrix, baseline, Pip_start, Pip_end, constrain_start, constrain_end, freq, title, idx=idx, tone_freqs=tone_freqs)

                # calculates adaptation as a function of exponential curve fitting.
                print(' fitting exponential curve to time dependant adaptacion')
                fitting = adaptacion_fit(folded_dict, FRo_rate=FRo_rate, std_dev_pooling=std_dev_pooling,
                                         force_asymptote=force_asymptote, silence_type=silence_type,
                                         elements_per_bin=elements_per_bin, tone_constrained=tone_constrained,
                                         curves=True, scatter=True)

                print('plotting responses against delay')
                neuron_adaptation_plot(fitting,silence_type,title, tone_freqs)

                block_counter = block_counter + 1

        elif loadedMat[0]['stimfmt'] == 'none':
            print('deprecation warning: working file with stimfmt = none')
            for block in loadedMat:
                freq = block['respf']
                matrix = block['resp'],0,1
                title = '{}, Jitter Off'.format(basename)
                #PrePipSilence = int(np.floor((block['PipInterval'] / 2) * block['stimf']))
                PrePipSilence = 0
                #PipLength = int(np.floor(block['PipDuration'] * block['stimf']))
                PipLength = 0
                #SliceLength = int(np.floor((block['PipDuration'] + block['PipInterval']) * block['stimf']))
                SliceLength = 0

                if tone_constrained == 'all':
                    constrain_start = PrePipSilence
                    constrain_end = PrePipSilence + PipLength
                elif tone_constrained == 'no':
                    constrain_start = 0
                    constrain_end = 0
                elif tone_constrained == 'start':
                    constrain_start = PrePipSilence
                    constrain_end = SliceLength
                else:
                    print("Invalid Tone_constrained Parameter. choose either 'all', 'start' or 'no' (default)")

                single_neuron_plot(matrix, 0, constrain_start, constrain_end, freq, title)
                continue

def neuron_adaptation_plot (fittings, silence_type='rel', filename='no filename', tone_freqs=None):
    # todo: add  values of fitted parameters, generalize to raaster pairs instead of stacked_dict

    sil_resp_pairs = fittings['scatter'][:,:,:,0] # this slice takes away the last dim which contains variance values on pos[1]
    params = fittings['params']
    # Parese parameter values for printing, limiting the number of decimals to 3
    parA = ['{:+.3f}'.format(pr) for pr in params[:, 0]]
    parB = ['{:+.3f}'.format(1/pr) for pr in params[:, 1]] # gets the inverse of rate, i.e. time constant.
    curves = fittings['curves']
    if tone_freqs != None:
        NameFA = '{} Hz'.format(tone_freqs[0]) ; NameFB = '{} Hz'.format(tone_freqs[1])

    else:  NameFA = 'FreqA' ; NameFB = 'FreqB'

    if silence_type == 'rel':
        timing = 'relative to previous tone of same type'
    elif silence_type == 'abs':
        timing = 'relative to previous tone of any type'
    else:
        print("error: abs_or_rel should be either 'abs' or 'rel'")

    # Legacy printing of non pooled adaptacion TODO make a cleaner version that can work with n number of tone types
    if sil_resp_pairs.shape[1] == 6 :
        #defines figures and 2 horizontal subplots for each frequency
        plt.figure()
        plt.suptitle('{}, spikes after delay {}'.format(filename, timing))

        plt.subplot(1,2,1)
        plt.scatter(sil_resp_pairs[:, fAD, 0] , sil_resp_pairs[:, fAD, 1], marker='v', alpha=1, label='{} Deviant  '.format(NameFA), color='red')
        plt.scatter(sil_resp_pairs[:, fAS, 0] , sil_resp_pairs[:, fAS, 1], marker='^', alpha=1, label='{} Standard '.format(NameFA), color='orange')
        #plt.scatter(sil_resp_pairs[:, fAO, 0] , sil_resp_pairs[:, fAO, 1], marker='o', alpha=1, label='{} Onset    '.format(NameFA), color='yellow')
        plt.plot(curves[fAD][0, :], curves[fAD][1, :], linestyle='-', label='{} Deviant exp fit a={}; b={}'.format(NameFA,parA[fAD],parB[fAD]) ,color='red')
        plt.plot(curves[fAS][0, :], curves[fAS][1, :], linestyle='-', label='{} Standard exp fit a={}; b={}'.format(NameFA,parA[fAS],parB[fAS]) ,color='orange')
        #plt.plot(curves[fAO][0, :], curves[fAO][1, :], linestyle='-', label='{} Onset exponential fit'.format(NameFA) ,color='yellow')

        plt.title('{}'.format(NameFA))
        plt.xscale('log')
        plt.xlabel('log of time in s')
        plt.ylabel('spikes per tone')
        plt.legend(loc='upper right')

        plt.subplot(1,2,2)
        plt.scatter(sil_resp_pairs[:, fBD, 0] , sil_resp_pairs[:, fBD, 1], marker='v', alpha=1, label='{} Deviant  '.format(NameFB), color='blue')
        plt.scatter(sil_resp_pairs[:, fBS, 0] , sil_resp_pairs[:, fBS, 1], marker='^', alpha=1, label='{} Standard '.format(NameFB), color='purple')
        #plt.scatter(sil_resp_pairs[:, fBO, 0] , sil_resp_pairs[:, fBO, 1], marker='o', alpha=1, label='{} Onset    '.format(NameFB), color='cyan')
        plt.plot(curves[fBD][0, :], curves[fBD][1, :], linestyle='-', label='{} Deviant exp fit a={}; b={}'.format(NameFB,parA[fBD],parB[fBD]) ,color='blue')
        plt.plot(curves[fBS][0, :], curves[fBS][1, :], linestyle='-', label='{} Standard exp fit a={}; b={}'.format(NameFB,parA[fBS],parB[fBS]) ,color='purple')
        #plt.plot(curves[fBO][0, :], curves[fBO][1, :], linestyle='-', label='{} Onset exponential fit'.format(NameFB) ,color='cyan')

        plt.title('{}'.format(NameFB))
        plt.xscale('log')
        plt.xlabel('log of time in s')
        plt.ylabel('spikes per tone')
        plt.legend(loc='upper right', fontsize='large')

        plt.show()
    elif sil_resp_pairs.shape[1] == 4 :

        fig, ax = plt.subplots()
        fig.suptitle('{}, spikes after delay {}'.format(filename, timing))

        ax.scatter(sil_resp_pairs[:, fAp, 0], sil_resp_pairs[:, fAp, 1], marker='v', alpha=0.7, label='{} pool  '.format(NameFA), color='firebrick')
        ax.scatter(sil_resp_pairs[:, fBp, 0], sil_resp_pairs[:, fBp, 1], marker='^', alpha=0.7, label='{} pool '.format(NameFB), color='navy')
        ax.plot(curves[fAp][0, :], curves[fAp][1, :], linestyle='-', label='{} pool exp fit a={}; b={}'.format(NameFA, parA[fAp], parB[fAp]), color='firebrick')
        ax.plot(curves[fBp][0, :], curves[fBp][1, :], linestyle='-', label='{} pool exp fit a={}; b={}'.format(NameFB, parA[fBp], parB[fBp]), color='navy')

        ax.set_title('Pooled std_dev for both freq')
        ax.set_xscale('log')
        ax.set_xlabel('log of time in second ')
        ax.set_ylabel('FR')
        ax.legend(loc='upper right', fontsize='large')



def quality_check (PathIn=None):
    if PathIn == None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename()
    else: path = PathIn

    #setting color map for stim raster
    bounds = [0, 0.5, 1.5,2]
    Fcmap = colors.ListedColormap(['white', 'firebrick','navy'])
    Fnorm = colors.BoundaryNorm(bounds, Fcmap.N)

    mat = bup.load_baphy_ssa(path)
    recnum = len(mat)
    fig, axs = plt.subplots(2, 2)
    axs = axs.ravel()
    fig.suptitle(path)

    for rec in range(len(mat)):
        resp = np.squeeze(mat[rec]['resp'])
        env = mat[rec]['stim']
        env[1,:,:] = env[1,:,:]*2
        env = np.max(env,0)

        js=resp.shape
        resp = resp[0:js[0], 0:js[1]]
        psth = np.nanmean(resp, 0)

        axs[rec].plot((psth / np.nanmax(psth)) * env.shape[0], alpha=0.9)
        axs[rec].imshow(env, aspect='auto', origin='lower', cmap=Fcmap, norm=Fnorm, interpolation='nearest', alpha=0.7)

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


def quality_checkv2 (PathIn=None):
    if PathIn == None:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename()
    else: path = PathIn

    #setting color map for stim raster
    bounds = [0, 0.5, 1.5,2]
    Fcmap = colors.ListedColormap(['white', 'firebrick','navy'])
    Fnorm = colors.BoundaryNorm(bounds, Fcmap.N)

    mat = bup.load_baphy_ssa(path)
    fig, axs = plt.subplots(int(np.ceil(len(mat)/2)), len(mat))
    try:
        axs = axs.ravel()
    except: pass
    fig.suptitle(path)

    for sp in range(len(mat)):

        env = mat[sp]['stim']
        env[1, :, :] = env[1, :, :] * 2
        env = np.max(env, 0)


        fig.suptitle(path)
        try:
            axs[sp].imshow(env, aspect='auto', origin='lower', cmap=Fcmap, norm=Fnorm, interpolation='nearest', alpha=0.7)
        except:
            axs.imshow(env, aspect='auto', origin='lower', cmap=Fcmap, norm=Fnorm, interpolation='nearest',
                           alpha=0.7)

        for ii in range(mat[sp]['resp'].shape[0]):
            resp = mat[sp]['resp'][ii,0,:] + ii - 0.5
            try:
                axs[sp].plot((resp / np.nanmax(mat[0]['resp'])) * (env.shape[0]/mat[sp]['resp'].shape[0]), alpha=0.9, color='black')
            except:
                axs.plot((resp / np.nanmax(mat[0]['resp'])) * (env.shape[0] / mat[sp]['resp'].shape[0]), alpha=0.9,
                             color='black')

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.show()

def adaptacion_fit (folded_dict=None, FRo_rate=False, std_dev_pooling=False, force_asymptote=False, silence_type='rel',
                    elements_per_bin=False, tone_constrained='start', baseline_substraction=False, curves=False, scatter=False):

    # Defines the function to be fitted
    def func(x, a, b):
        return a * (1 -(np.exp(-b * x)))
    param_num = len(signature(func).parameters) - 1

    #if asked for number of parameters to fit give the number skipping all other computations
    if folded_dict == None:
        if force_asymptote == False:
            print('number of model parameters to fit: {}'.format(param_num))
            return param_num
        elif force_asymptote == True:
            print('number of model parameters to fit: {}, and forcing asymptote'.format(param_num-1))
            return param_num
    elif folded_dict != None:
        pass
    else: print('ask_param_num should be either True or False')

    #set silence type and parces the proper slice o the silences array
    if silence_type == 'abs':
        silence_ID = abssilid
        silences = folded_dict['silences'][:, silence_ID, :] / folded_dict['respf']
    elif silence_type == 'rel':
        silence_ID = relsilid
        silences = folded_dict['silences'][:, silence_ID, :] / folded_dict['respf']
    else: print("choose either absolute 'abs' or relative 'rel' silence type")

    # Integrates the responses along time. If FRo_rate = True, presents this response as (FR / FRo)
    FRo_dict = calc_FR_FRo(folded_dict, tone_constrained=tone_constrained, baseline_substraction=baseline_substraction,
                           FRo_rate=FRo_rate)
    AreaUnder = FRo_dict['AreaUnder']

    # set function with forced asymptote
    if force_asymptote == True:
        fA_asymptote = FRo_dict['asymptotes'][0]
        fB_asymptote = FRo_dict['asymptotes'][1]
        def forced_func_A(x,b,c):
            return  fA_asymptote * (1 - (c * np.exp(-b * x)))

        def forced_func_B(x,b,c):
            return  fB_asymptote * (1 - (c * np.exp(-b * x)))


    # If True, pools deviant and standard time-response-pairs for each frequency
    if std_dev_pooling == False:
        workingToneTypes = ToneIDList
        pass
    elif std_dev_pooling == True:
        print('pooling standard and deviant values')
        workingToneTypes = pooledIdList

        resp_list = list()
        sil_list = list()
        for tt in range (AreaUnder.shape[1]):
            resp = AreaUnder[~np.isnan(AreaUnder[:, tt]), tt]
            sil = silences[~np.isnan(silences[:, tt]), tt]
            resp_list.append(list(resp))
            sil_list.append(list(sil))

        pooled_resp_list = [resp_list[fAD]+resp_list[fAS], resp_list[fAO], resp_list[fBD]+resp_list[fBS], resp_list[fBO]]
        pooled_sil_list = [sil_list[fAD]+sil_list[fAS], sil_list[fAO], sil_list[fBD]+sil_list[fBS], sil_list[fBO]]

        max_resp_list = list()
        for resp in pooled_resp_list:
            max_resp_list.append(len(resp))
        respdimlen = max(max_resp_list)
        AreaUnder = np.ndarray([respdimlen,4])
        AreaUnder[:] = np.nan

        max_sil_list = list()
        for sil in pooled_sil_list:
            max_sil_list.append(len(sil))
        sildimlen = max(max_sil_list)
        silences = np.ndarray([sildimlen,4])
        silences[:] = np.nan

        # Organizes lists of different length into ndarray with nan padding
        for ii, resp_sil in enumerate(zip(pooled_resp_list,pooled_sil_list)):
            resp = resp_sil[0]
            sil = resp_sil[1]
            AreaUnder[:len(resp),ii] = resp
            silences[:len(sil),ii] = sil
    else:
        print('std_dev_pooling should be bool ')

    # Binns the response acording to the desired number of elements per bin.
    # stack silences and responses into a 4d array with dim0(#tones) tone number; dim1(6) tone type; dim2(2) sil(0) resp(1)
    # dim3(2) mean(0) or var (1). If binning is skiped, mean is raw points and var is nan padding.
    binned_scatter = scatter_binning(AreaUnder,silences,elements_per_bin=elements_per_bin)

    if curves == True:
        curve_list = list()

    fit_params = list()
    fit_covar = list()

    for tt in range(binned_scatter.shape[1]):
        print('fitting {}'.format(workingToneTypes[tt]))
        # removes nan and uses only means for curve fitting.
        x = binned_scatter[:, tt, 0, 0][~np.isnan(binned_scatter[:, tt, 0, 0])]
        y = binned_scatter[:, tt, 1, 0][~np.isnan(binned_scatter[:, tt, 1, 0])]

        try:
            halfway = binned_scatter.shape[1]/2
            if force_asymptote == False:
               popt, pcov = curve_fit(func,x,y)

            elif force_asymptote == True:
                if tt < halfway: # fits equation with forced freq A asymptotote
                    forced_popt, forced_pcov = curve_fit(forced_func_A, x, y)
                    # Add the forced asymptote to the begining of the parameter array. adds NAN at the beggining of the covar
                    popt = np.ndarray(forced_popt.shape[0]+1)
                    popt[0] = fA_asymptote
                    popt[1:] = forced_popt
                    pcov = np.ndarray(forced_pcov.shape[0]+1)
                    pcov[:] = np.nan
                    pcov[1:,:] = forced_pcov

                elif tt >= halfway: # fits equation with forced freq B asymptote
                    forced_popt, forced_pcov = curve_fit(forced_func_B, x, y)
                    # Add the forced asymptote to the begining of the parameter array. adds NAN at the beggining of the covar
                    popt = np.ndarray(forced_popt.shape[0] + 1)
                    popt[0] = fB_asymptote
                    popt[1:] = forced_popt
                    pcov = np.ndarray(forced_pcov.shape[0] + 1)
                    pcov[:] = np.nan
                    pcov[1:, :] = forced_pcov


            #print('fited vars: {} ; covariance: {}'.format(popt,pcov))

            fit_params.append(popt)
            try:
                fit_covar.append(pcov)
            except: pass

            if curves == True:
                # using the ecuation and fitted parameters draw a curve starting from a lin or log space in x
                print('drawing curve for {}'.format(workingToneTypes[tt]))
                cx = np.logspace(np.log10(np.min(x)),np.log10(np.max(x)),20)
                cy = func(cx,*popt)
                curve = np.stack([cx, cy], 0)
                curve_list.append(curve)
        except: #if no fit is possible fill the values with NAN
            popt = np.empty([param_num])
            popt[:] = np.nan
            pcov = np.empty([param_num,2])
            pcov[:] = np.nan
            fit_params.append(popt)
            fit_covar.append(pcov)
            print('coudl not fit model for {}, parsing NAN as values'.format(workingToneTypes[tt]))

            if curves == True:
                # If drawing the curve is not possible, most likely for onset tones, fills list with NAN pair.
                print('drawing NAN placeholder for {} curve'.format(workingToneTypes[tt]))
                cx = np.array([np.nan])
                cy = np.array([np.nan])
                curve = np.stack([cx, cy], 0)
                curve_list.append(curve)

    fit_params = np.asarray(fit_params)
    try:
        fit_covar = np.asarray(fit_covar)
    except: pass
    fittings = dict()
    fittings['params'] = fit_params
    fittings['covar'] = fit_covar
    if curves == True:
        fittings['curves'] = curve_list

    if scatter == True:
        fittings['scatter'] = binned_scatter

    return fittings

def batch_adaptation_fit (dirPath, FRo_rate=False, std_dev_pooling=False, force_asymptote=False, silence_type='rel',
                          elements_per_bin=False,  tone_constrained='start', baseline_substraction=False, isol_threshold=None):

    pathlist = file_listing(dirPath, animals=['gus'], filter_outliers=True, stim_type='enve')

    # check number of fitted parameters to create empty arrays of the proper dimentions
    param_num = adaptacion_fit()

    fit_param_list = list()
    fit_covar_list = list()

    used_file_list = list()
    used_freqPair_list = list()
    for pp in pathlist:

        loadedMatFile = bup.load_baphy_ssa(pp)
        block_counter = 0
        block_num = len(loadedMatFile)

        print('working on file {} with {} blocks'.format(os.path.basename(pp),block_num))

        # checks for all frequency pairs for all the blocks within a file
        freq_pairs = list()
        for block in loadedMatFile:
            freq_pairs.append(block['Frequencies'])
        unique_freq = [list(x) for x in set(tuple(x) for x in freq_pairs)] # aka jitter pairs of the same freq

        for jitter_pair in unique_freq:

            if std_dev_pooling == False:
                # defines the unit or parameters and covar for a single jitter pair, dim0(2) jitter off(0) on (1);
                # dim1(6) tone type, dim2(param_num) parameter of the fitted model
                single_pair_params = np.ndarray([2,6,param_num])
                single_pair_params[:] = np.nan
                # dim3 (2) por some reason i dont understand covarianza has this aditional dimension of this size.
                single_pair_covar = np.ndarray([2,6,param_num,2])
                single_pair_covar[:] = np.nan
            elif std_dev_pooling == True:
                # defines the unit or parameters and covar for a single jitter pair, dim0(2) jitter off(0) on (1);
                # dim1(4) tone type, dim2(param_num) parameter of the fitted model
                single_pair_params = np.ndarray([2, 4, param_num])
                single_pair_params[:] = np.nan
                # dim3 (2) por some reason i dont understand covarianza has this aditional dimension of this size.
                single_pair_covar = np.ndarray([2, 4, param_num, 2])
                single_pair_covar[:] = np.nan

            for block in loadedMatFile:

                # checks if the block belongs to the ongoing pair
                if block['Frequencies'] == jitter_pair:
                    pass
                elif block['Frequencies'] != jitter_pair:
                    continue
                else:
                    print('theres in an error in how jitter pairs are being compared')
                    break

                # check for isolation percentage
                if isol_threshold == None:
                    print('working on block #{}'.format(block_counter))
                    pass
                elif 0 <= isol_threshold <= 100:
                    if block['isolation'] >= isol_threshold:
                        print('working on block #{}'.format(block_counter))
                        pass
                    elif block['isolation'] < isol_threshold:
                        print('block {} skiped due to isolation under selected threshold'.format(block_counter))
                        block_counter = block_counter + 1
                        continue
                    else:
                        print('istolation not defines in loaded .mat file block {}, skipping block'.
                              format(block_counter))
                        block_counter = block_counter + 1
                        continue
                else:
                    print('isol_threshold has to be between 0 and 100')
                    break

                print('splicing')
                stacked_dict = fold_tones(block)

                # Fits parameters by calling adaptacion_fit
                print('fitting model')
                fitting = adaptacion_fit(stacked_dict, FRo_rate=FRo_rate, std_dev_pooling=std_dev_pooling,
                                         force_asymptote=force_asymptote, silence_type=silence_type,
                                         elements_per_bin = elements_per_bin, tone_constrained=tone_constrained,
                                         baseline_substraction=baseline_substraction, curves=False, scatter=False)
                params = fitting['params']
                covar = fitting['covar']

                if block['stimfmt'] == 'envelope':
                    if stacked_dict['Jitter'] == 'On ':
                        single_pair_params[Jon,:,:] = params
                        single_pair_covar[Jon,:,:,:] = covar
                    elif stacked_dict['Jitter'] == 'Off':
                        single_pair_params[Joff,:,:] = params
                        single_pair_covar[Joff,:,:,:] = covar
                    else: print('something wrong with jitter')
                elif block['stimfmt'] == 'none':
                    single_pair_params[Joff, :, :] = params
                    single_pair_covar[Joff, :, :, :] = covar
                else: print('error: stimfmt should be either "none" or "envelope" ')

                block_counter = block_counter + 1

            # Checks if the generated pair is empty, if so, does nothing, if one element is found, stores by appending.
            for idx in np.nditer(single_pair_params):
                if np.isnan(idx):
                    continue
                elif ~np.isnan(idx):
                    fit_param_list.append(single_pair_params) # Appendds pair
                    fit_covar_list.append(single_pair_covar)
                    used_file_list.append(os.path.basename(pp))
                    used_freqPair_list.append(jitter_pair)
                    break
                else:
                    print('elements jitter-pair params array shoudl be either NAN or float')

    print('stacking single file arrays')
    fit_param_array = np.stack(fit_param_list, axis=0)
    fit_covar_array = np.stack(fit_covar_list,axis=0)

    pop_fit_dict = dict()

    # 4d parameter array. dim0= file; dim1= Jitter on(1)off(2); dim2= tone type (6); dim3= model param
    pop_fit_dict['params'] = fit_param_array
    # 5d covariance array. dim0= file; dim1= Jitter on(1)off(2); dim2= tone type (6); dim3= model param; dim4= ???
    pop_fit_dict['covar'] = fit_covar_array
    pop_fit_dict['filename'] = used_file_list
    pop_fit_dict['freq_pair'] = used_freqPair_list

    return pop_fit_dict

def population_fit_plot(pop_fit_dict, title=None, std_dev_pooling=True):
    # plots the fitted parameters and compares between pairs of jitter and nonjitered. does this for each non onset tone type
    # and for each parameter

    # what do i what to show and in what order of importance? how jitter alters the fitted parameteres. jitter is seen as
    # two adyacend boxplots. also how it changes if being standard or deviant (another pair of boxplots)
    # this completes a single subplot.
    # next subplot in the horizonta, does the same for frequency b
    # next subplot in the vertical does the same for parameter 2

    # todo create organizing array or iterator over wich to iterate and call proper indexes for boxplotting.
    dev_or_std= ['standard', 'deviant']
    jitter = ['no Jitter', 'Jitter']
    param_num = adaptacion_fit()
    param_name = ['asymptote, [spikes/s]', 'Time constant [s]']
    frequencies = ['freq A ', 'freq B']
    popt = pop_fit_dict['params'].copy()
    # Gets the reciprocal of the rate i.e, the time constant, for more meaningfull visualization
    popt[:,:,:,1] = np.reciprocal(popt[:,:,:,1])

    f, axarr = plt.subplots(param_num, 2)
    if title == None:
        pass
    else:
        f.suptitle(title)

    if std_dev_pooling == False:
        for param, freq in product(range(param_num), range(len(frequencies))):

            deviant = 0 + (freq * 3)
            standar = 1 + (freq * 3)  # either 0 and 1; or 3 and 4

            subplot = freq+ 1 + (param * 2)

            box1 = popt[:, Joff, deviant, param]  # all points (nan inclusive) tone as standard non jittered
            nanBox1 = box1[~np.isnan(box1)]
            box2 = popt[:, Jon , deviant, param]  # all points (nan inclusive) tone as standard jittered
            nanBox2 = box2[~np.isnan(box2)]
            box3 = popt[:, Joff, standar, param]  # all points (nan inclusive) tone as deviant  non jittered
            nanBox3 = box3[~np.isnan(box3)]
            box4 = popt[:, Jon , standar, param]  # all points (nan inclusive) tone as deviant  jittered
            nanBox4 = box4[~np.isnan(box4)]

            # plt.xlabel(frequencies[freq]) ; plt.ylabel('fitted {}'.format(param_name[param]))

            box = axarr[param, freq].boxplot([nanBox1, nanBox2, nanBox3, nanBox4], patch_artist=True)

            for DoS, pp in product(range(len(dev_or_std)),range(len(box1))):
                toneType = DoS + (freq * 3)

                pair = popt[pp, :, toneType, param]
                plt.plot([1.25 + (DoS * 2), 1.75 + (DoS * 2)], pair, '-k', marker='o', alpha=0.3)

            color = ['firebrick', 'navy', 'red', 'blue']
            for patch, color in zip(box['boxes'], color):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)

        plt.xticks([1, 2, 3, 4], ['deviant No jitter-', 'deviant Jitter-', 'standard No Jitter-', 'standard Jitter'], )
        # plt.gca().invert_yaxis()

    elif std_dev_pooling == True:
        for param, freq in product(range(param_num), range(len(frequencies))):
            # current axis, aka subplot
            curax = axarr[param, freq]
            # x axis, jitter off
            x = popt[:,Joff, freq*2, param] # for scatter
            xnan = x[~np.isnan(x)]  # for boxplot

            # y axis Jitter On
            y = popt[:,Jon, freq*2, param]  # for scatter
            ynan = y[~np.isnan(y)]  # for boxtplot
            # Plots scatter of points hopefully following the y=x diagonal
            curax.scatter(x,y, color='black', alpha = 0.8)
            # Store ticks and limits  for boxplot placement and for solving range and ticks issues after boxplot fucks it
            tcksx = curax.get_xticks()
            tcksy = curax.get_yticks()
            limx = curax.get_xlim()
            limy = curax.get_ylim()
            #defines the range of the data as the upper limit minos the lower limit for both x and y
            ranx = max(limx) - min(limx)
            rany = max(limy) - min(limy)
            fract = 20 #fraction of the range that the boxplot ocupies
            # value to shift the limit, and define the width of the boxplot
            nranx = ranx / fract
            nrany = rany / fract

            HboxPos = limy[0]
            VboxPos = limx[0]

            # plots boxplots for Jitter off (x axis) and jitter On (y axis)
            Hbox = curax.boxplot(xnan, positions = [HboxPos], widths=[nrany], vert = False, patch_artist=True)
            Vbox = curax.boxplot(ynan, positions = [VboxPos], widths=[nranx], vert = True,  patch_artist=True)

            #setcolors
            Hbox['boxes'][0].set_facecolor('red')  # jitter off
            Vbox['boxes'][0].set_facecolor('blue') # jitter on

            # set ticks since boxplot delets them
            curax.set_xticks(tcksx)
            curax.set_yticks(tcksy)
            #values of ticks
            curax.xaxis.set_ticklabels(tcksx)
            curax.yaxis.set_ticklabels(tcksy)

            # Set limis of subplot in x and y to acmodate the boxplots
            curax.set_xlim([limx[0] - nranx ,limx[1]])
            curax.set_ylim([limy[0] - nrany ,limy[1]])

            # Draws a diagonal y = x line.
            curax.plot(limx,limx, ls="--", c=".3")

            # adds x and y lables with units, as well as subplot titels with fited parameter and frequency
            if param == 0:
                curax.set_ylabel('max FR / FRo jitter on')
                curax.set_xlabel('max FR / FRo jitter off')
                if freq == 0 :
                    curax.set_title('fitted Asymptote, frequency A ')
                elif freq == 1:
                    curax.set_title('fitted Asymptote, frequency B')

            elif param == 1:
                curax.set_ylabel('seconds, jitter on')
                curax.set_xlabel('seconds, jitter off')
                if freq == 0 :
                    curax.set_title('fitted time constant, frequency A ')
                elif freq == 1:
                    curax.set_title('fitted time constant, frequency B')

            elif param == 2:
                curax.set_ylabel('min FR / FRo jitter on')
                curax.set_xlabel('min FR / FRo jitter off ')
                if freq == 0 :
                    curax.set_title('fitted minimuum response, frequency A ')
                elif freq == 1:
                    curax.set_title('fitted minimuum response, frequency B')

    plt.show()

def export_DF (idx_array, names_list):
    # todo, make it be able to export 3rd dim (jitter) of arrays. Right now only export non jittered 2d arrays
    dataframe = pd.DataFrame(idx_array[:, :, 0], index=names_list, columns=['SIFA', 'SIFB', 'SIcell'])

    try:
        # as for a empty excell spreadsheet to write the dataframe
        print('waiting for output excell file selection')
        root = tk.Tk()
        root.withdraw()
        xcellpath = filedialog.askopenfilename()
        dataframe.to_excel(xcellpath, sheet_name='bathc296 SI dump')

    except:
        print('not saving to an excell file, output in console if asigned'); pass

    return dataframe

def filter (imput, isol_threshold=None, activ_threshold=None, name=None, counter=None  ):

    if os.path.isfile(imput):
        if name != None:
            if imput == name:
                return imput
            else:
                return  None

    elif isinstance(imput,dict):
        if counter == None:
            block_counter = 'unknown'
        elif isinstance(counter,int):
            block_counter = counter

        # check for isolation percentage
        if isol_threshold != None:
            block = imput
            if isol_threshold == 0:
                print('working on block #{}'.format(block_counter)) # todo move this to the final block retunr
                pass
            elif 0 < isol_threshold <= 100:
                if block['isolation'] >= isol_threshold:
                    print('working on block #{}'.format(block_counter)) # todo move this to the final block return
                    pass
                elif block['isolation'] < isol_threshold:
                    print('block {} skiped due to isolation under selected threshold'.format(block_counter))
                    block_counter = block_counter + 1
                    return None
                else:
                    print('istolation not defines in loaded .mat file block {}, skipping block'.
                          format(block_counter))
                    block_counter = block_counter + 1
                    return None
            else:
                print('isol_threshold has to be between 0 and 100')
                return None

        # Checks for activity level
        elif activ_threshold != None:
            resp = imput['resp']
            if activ_threshold == 0:
                pass
            elif 0 < activ_threshold: # TODO: check if the PSTH for both frequencies at standar surpace an area under threshold
                pass

            else:
                print('activ_threshold sould be => 0')
                return None

        elif isol_threshold != None and activ_threshold != None:
            print('error: can filter only isolation or activity at a time')
            return None

def file_listing (dirPath, animals=None, filter_outliers = True, stim_type ='enve'):
    # initializes the list of paths to be returned after filtering
    pathlist = list()

    for filename in os.listdir(dirPath):
        # chechs if is matlab file
        if filename.endswith(".mat"):
            pass
        else:
            continue

        # Checks is envelope or none or any
        stimType = filename[16:20]
        if stim_type == None:
            pass
        elif stim_type in ['enve','none']:
            if stimType == stim_type:
                pass
            else:
                continue
        else: print("error : stim_type should be ither None(no filtering), 'envlope' or 'none'")

        animalname = filename[0:3]
        if animals == None:
            pass
        elif isinstance(animals, list):
            if animalname in animals:
                pass
            else:
                continue

        # Checks if is in list of recognized outliers
        if filter_outliers == True:
            if filename in outliers:
                continue
            else:
                pass

        filePath = os.path.join(dirPath, filename)
        pathlist.append(filePath)

    return  pathlist

def calc_FR_FRo(folded_dict, tone_constrained='start', baseline_substraction = False, FRo_rate= False):
    # Calculates the ratio between the firing rate (FR) and the Inicial Firing rate (FRo), the later comes from calculating
    # the firing rate for the integrated means of the onset responses. the ratios are done frequency-wise, i.e. fAS / fAO,
    # fAD / fAO... and so on. Returns a 2d array of dim0(#tones) tone number , dim1(6) tonetype. the function essentially
    # reduces the responses array by one dimension, namely time.

    # Sets tone constrictions
    if tone_constrained == 'all':
        constrain_start = folded_dict['PipStart']
        constrain_end = folded_dict['PipEnd']
    elif tone_constrained == 'no':
        constrain_start = 0
        constrain_end = folded_dict['SliceLen']
    elif tone_constrained == 'start':
        constrain_start = folded_dict['PipStart']
        constrain_end = folded_dict['SliceLen']
    else:
        print("Invalid Tone_constrained Parameter. choose either 'all', 'start' or 'no'")

    # Sets baseline for substraction
    if baseline_substraction == False:
        baseline = 0
    elif baseline_substraction == True:
        baseline = folded_dict['Baseline']
    else:
        print('baseline should be either on(true) or off (False)')

    resp = folded_dict['resp']
    FRoFreqA = np.sum(np.nanmean(resp[:,constrain_start:constrain_end,fAO],0),0)
    FRoFreqB = np.sum(np.nanmean(resp[:,constrain_start:constrain_end,fBO],0),0)
    integrated_resp = np.sum(resp[:, constrain_start:constrain_end, :], axis=1) - baseline

    if FRo_rate == False:

        integ_dict = dict()
        integ_dict['AreaUnder'] = integrated_resp
        integ_dict['asymptotes'] = [FRoFreqA,FRoFreqB]
        return integ_dict

    elif FRo_rate == True:
        FR_ratio = np.ndarray(integrated_resp.shape)
        FR_ratio[:] = np.nan
        for ii in range(integrated_resp.shape[1]):
            if ii < 3: #ratios for frequency A
                FR_ratio[:, ii] = integrated_resp[:,ii] / FRoFreqA
            elif ii >= 3 :
                FR_ratio[:, ii] = integrated_resp[:, ii] / FRoFreqB
        FR_dict = dict()
        FR_dict['AreaUnder'] = FR_ratio
        FR_dict['asymptotes'] = [FRoFreqA, FRoFreqB]
        return FR_dict

def scatter_binning(AreaUnder, silence_array, elements_per_bin=False):

    # stack silences and responses into a 4d array with dim0(#tones) tone number; dim1(#toneTypes) tone type;
    # dim2(2) sil(0) resp(1) dim3(2) mean(0) or var (1). If binning is skiped, mean is raw points and var is nan padding.
    binned_values = np.ndarray([silence_array.shape[0], AreaUnder.shape[1], 2, 2])
    binned_values[:] = np.nan
    binned_values[:, :, :, 0] = np.stack([silence_array, AreaUnder], 2)
    if elements_per_bin == False:
        #skipps binning
        pass
    # Checks for binning to either fit all the points or first get the mean of time bins and then do the fitting over the means
    # Cheks if binning will be done and initializes params
    elif isinstance(elements_per_bin,int):
        # checks if the standard and deviant have been pooled
        if binned_values.shape[1] == 6:
            ttnum = 6
            pooled = False
        elif binned_values.shape[1] == 4:
            ttnum = 4
            pooled = True
        else:
            print('error: the array imput for binning have wrong dimention sizes,\n'
                  'dim1 should be either 4 (pooled std dev) or 6 (independent std dev)')
            return None

        # counts max number of time points between tone types, this to set an upper limit to binning
        num_of_sil = list()
        for mm in range(silence_array.shape[1]):
            notnan = silence_array[~np.isnan(silence_array[:, mm]), mm]
            unique_time = set(notnan)
            num_of_sil.append(len(unique_time))
        # defines the minimum number of different silences betweetn tonetypes excluding onset
        if pooled == False:
            max_elem_per_bin = min (np.asarray(num_of_sil)[[0,1,3,4]])
        elif pooled == True:
            max_elem_per_bin = min(np.asarray(num_of_sil)[[0,2]])

        num_of_bins = int(np.ceil(np.max(num_of_sil)/elements_per_bin))
        # this 4d array has dim0(#bins)singe tones, dim1(tt) tone type; dim2(2) silence(0) or response(1); dim3(2) mean(0) or var(1)
        binned_values = np.ndarray([num_of_bins, ttnum, 2, 2])
        binned_values[:] = np.nan
        print('binning... \n'
              'Maximum number of elements per bin:{} \n'
              'Maximum number of silences:{}'.format(max_elem_per_bin,int(np.max(num_of_sil))))

        if 0 < elements_per_bin <= max_elem_per_bin and isinstance(elements_per_bin,int):
            for tt in range(binned_values.shape[1]): # Iterates over tone types.
                notnansil = silence_array [~np.isnan(silence_array[:,tt]), tt]
                sil_set = set(notnansil)
                sil_set = np.sort(list(sil_set))
                if elements_per_bin !=1:
                    bin_array = list()
                    for ss, sil  in enumerate(sil_set):
                        ran = range(0,len(sil_set),elements_per_bin)
                        if ss in ran:
                            try:
                                bin_array.append(sil)
                            except:
                                bin_array.append(sil+1)
                        else:
                            continue
                # When elements_per_bin is set to 1, binns by every unique time point
                elif elements_per_bin == 1:
                    bin_array = sil_set

                bin_idx = np.digitize(silence_array[:, tt], bin_array)
                # proper printing depending on if dev std pooled or not
                if pooled == False:
                    print('tone type: {}, number of bins: {}'.format(ToneIDList[tt],len(set(bin_idx))))
                elif pooled == True:
                    print('tone type: {}, number of bins: {}'.format(pooledIdList[tt],len(set(bin_idx))))
                for ii in set(bin_idx):
                    binned_values[ii-1, tt, 0, 0] = np.nanmean(silence_array[bin_idx == ii, tt])
                    binned_values[ii-1, tt, 1, 0] = np.nanmean(AreaUnder[bin_idx == ii, tt])
                    binned_values[ii-1, tt, 0, 1] = np.nanvar(silence_array[bin_idx == ii, tt])
                    binned_values[ii-1, tt, 1, 1] = np.nanvar(AreaUnder[bin_idx == ii, tt])
        else:
            print('error: elements_ per_bin should be between 1 and {}'.format(max_elem_per_bin))
    else:
        print('elements_per_bin sould be either False (no binning) or interger > 0')
        return None

    return binned_values

def batch_fold_tones (dirPath, animal=None, filter_outliers=True, stim_type = 'enve' ):
    # This functions is to bypass the need of folding every file every time and instead doing it once, folding for the
    # hole batch and organizing the results in a dataframe. this DF can be easily pickled and sved for later.
    # although the rest of the code is not yet ready to parse the data from this DF
    pathlist = file_listing(dirPath, animals=animal, filter_outliers=filter_outliers, stim_type=stim_type)

    pair_counter = 0
    block_counter = 0

    # Defines a pandas dataframe with columns file name, frequency , pair ID, jitter type, data ndarray
    folded_DF = pd.DataFrame(data=None, index=None, columns=['file name', 'pair ID', 'Jitter', 'data_dict'])

    for pp in pathlist:
        loadedMatFile = bup.load_baphy_ssa(pp)

        num_of_blocks = len(loadedMatFile)

        print('working on file {} with {} blocks'.format(os.path.basename(pp), num_of_blocks))

        # checks for all frequency pairs for all the blocks within a file
        freq_pairs = list()
        for block in loadedMatFile:
            freq_pairs.append(block['Frequencies'])
        unique_freq = [list(x) for x in set(tuple(x) for x in freq_pairs)]  # aka jitter pairs

        for jitter_pair in unique_freq:
            block_in_pair_counter = 0

            for bb, block in enumerate(loadedMatFile):
                # checks if the block belongs to the ongoing pair
                if block['Frequencies'] == jitter_pair:
                    pass
                elif block['Frequencies'] != jitter_pair:
                    continue
                else:
                    print('theres in an error in how jitter pairs are being compared')
                    break

                # folds the block by calling fold_tones
                print('folding block {}'.format(bb))
                folded_block  = fold_tones(block)

                print('adding to data frame')
                folded_DF.set_value(block_counter, 'file name', os.path.basename(pp))
                folded_DF.set_value(block_counter, 'pair ID', pair_counter)
                folded_DF.set_value(block_counter, 'Jitter', block['Jitter'])
                folded_DF.set_value(block_counter, 'data_dict', folded_block)

                block_counter = block_counter + 1
                block_in_pair_counter = block_in_pair_counter + 1

            # fills an empty rwo in if a block does not have a jitter pair, sets the proper jitter value for the empty pair
            if block_in_pair_counter < 2:
                print ('filling absent pair element with NAN')
                paired_jitter_val = folded_DF.iloc[-1]['Jitter']
                if paired_jitter_val == 'On ':
                    folded_DF.set_value(block_counter, 'file name', os.path.basename(pp))
                    folded_DF.set_value(block_counter, 'pair ID ', pair_counter)
                    folded_DF.set_value(block_counter, 'Jitter', 'Off')

                elif paired_jitter_val == 'Off':
                    folded_DF.set_value(block_counter, 'file name', os.path.basename(pp))
                    folded_DF.set_value(block_counter, 'pair ID', pair_counter)
                    folded_DF.set_value(block_counter, 'Jitter', 'On ')

                else: # assumes that if there is no jitter value, the absent block of the pair has Jitter On
                    folded_DF.set_value(block_counter, 'file name', os.path.basename(pp))
                    folded_DF.set_value(block_counter, 'pair ID', pair_counter)
                    folded_DF.set_value(block_counter, 'Jitter', 'On ')

                block_counter = block_counter + 1
                block_in_pair_counter = block_in_pair_counter + 1

            pair_counter = pair_counter + 1

    return folded_DF

def get_file_name(cellid, root='/auto/data/code/nems_in_cache/batch296/', fs='1000'):
    return  str(root+cellid+'_b296_envelope_fs{}'.format(fs))

def fastplot(cellid):
    path_plot(get_file_name(cellid))





