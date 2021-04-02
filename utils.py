import scipy.io as sio
from collections import defaultdict
import numpy as np
import random
import scipy.signal
from model import *

def notch_filter(signal,f_R,fs):
    B,A = scipy.signal.iirnotch(f_R,int(f_R/10),fs)
    return scipy.signal.lfilter(B, A, signal, axis=0)
def bandpass(signal,band,fs):
    B,A = scipy.signal.butter(5, np.array(band)/(fs/2), btype='bandpass')
    return scipy.signal.lfilter(B, A, signal, axis=0)
def shuffled_data(data,label):
    permutation = np.random.permutation(data.shape[0])
    shuffled_data = data[permutation, :, :]
    shuffled_label = label[permutation, :]
    return shuffled_data, shuffled_label
##################################################################################
def train_data_extraction_from_mat(Subject):
    path_mat = sio.loadmat('data/' + 'Subject_' + Subject + '_' + 'Train' + '.mat')
    Flashings = path_mat['Flashing'];Signals = path_mat['Signal']; StimulusCodes = path_mat['StimulusCode'];
    StimulusTypes =  path_mat['StimulusType'];TargetChars =  path_mat['TargetChar'][0]
    return Flashings,Signals,StimulusCodes,StimulusTypes,TargetChars
def ERP_location(Flashing):
    loactions = [];Counts = 0
    for loaction,trigger in enumerate(Flashing):
        if trigger == int(1):
            Counts += 1
            if Counts == 24:
                loactions.append(loaction);Counts = 0
    return np.array(loactions)
def Single_trial_target(TargetChar):
    sgl_Target = []
    screen=[['A','B','C','D','E','F'],
            ['G','H','I','J','K','L'],
            ['M','N','O','P','Q','R'],
            ['S','T','U','V','W','X'],
            ['Y','Z','1','2','3','4'],
            ['5','6','7','8','9','_']]
    for i in range(0,6):
        for j in range(0,6):
            if TargetChar  == screen[i][j]: 
                sgl_Target += [i+7,j+1]
    return sgl_Target
def Single_local_information(loactions,StimulusCode):
    loca_dicts = defaultdict(list)
    for i,code in enumerate(StimulusCode[loactions]):
        if int(code) == 1: loca_dicts[1].append(loactions[i])
        if int(code) == 2: loca_dicts[2].append(loactions[i])
        if int(code) == 3: loca_dicts[3].append(loactions[i])
        if int(code) == 4: loca_dicts[4].append(loactions[i])
        if int(code) == 5: loca_dicts[5].append(loactions[i])
        if int(code) == 6: loca_dicts[6].append(loactions[i])
        if int(code) == 7: loca_dicts[7].append(loactions[i])
        if int(code) == 8: loca_dicts[8].append(loactions[i])
        if int(code) == 9: loca_dicts[9].append(loactions[i])
        if int(code) == 10: loca_dicts[10].append(loactions[i])
        if int(code) == 11: loca_dicts[11].append(loactions[i])
        if int(code) == 12: loca_dicts[12].append(loactions[i])
    
    return loca_dicts
def ERP_extraction(Signal,loca_dict,samps):
    seq =5
    ERPs = []
    for response in loca_dict:
        res_start =  response + 1;res_end =  response + 1 + 120
        ERP = scipy.signal.resample(Signal[res_start:res_end,:], samps)
        ERPs += [ERP]
    ERPs  = np.array(ERPs)
    ###
    seqs = np.array(range(int(seq)))
    ERPs = ERPs[seqs,:,:]
    ###
    ERP = np.mean(ERPs,axis=0)
    return ERP
def Single_trial_ERP(Signal,sgl_Target,loca_dicts,samps):
    Target_ERPs = [];NonTarget_ERPs = []
    Tar_0 = sgl_Target[0];Tar_1 = sgl_Target[1]
    for i in range(1,13):
        X_ERP = ERP_extraction(Signal,loca_dicts[i],samps)
        if i == Tar_0:     Target_ERPs.append(X_ERP)
        elif i == Tar_1:   Target_ERPs.append(X_ERP)
        else:NonTarget_ERPs.append(X_ERP)
    Target_ERPs = np.array(Target_ERPs);NonTarget_ERPs = np.array(NonTarget_ERPs)
    return Target_ERPs,NonTarget_ERPs  
def train_data_and_label(Subject,samps):
    Flashings,Signals,StimulusCodes,StimulusTypes,TargetChars = train_data_extraction_from_mat(Subject)
    Target = [];NonTarget = []
    for i in range(len(Flashings)):
        Flashing = Flashings[i];Signal = Signals[i];StimulusCode = StimulusCodes[i];TargetChar = TargetChars[i]
        Signal = bandpass(Signal,[0.1,20.0],240)
        loactions = ERP_location(Flashing)
        sgl_Target =Single_trial_target(TargetChar)
        loca_dicts = Single_local_information(loactions,StimulusCode)
        Target_ERPs,NonTarget_ERPs = Single_trial_ERP(Signal,sgl_Target,loca_dicts,samps)
        Target += [Target_ERPs]; NonTarget += [NonTarget_ERPs]
    Target = np.array(Target);NonTarget = np.array(NonTarget)
    Target = Target.reshape(-1,samps,64)
    NonTarget = NonTarget.reshape(-1,samps,64)
    Target_label = np.ones((Target.shape[0],1),dtype=np.int)
    NonTarget_label = np.zeros((NonTarget.shape[0],1),dtype=np.int)
    
    return  Target,Target_label,NonTarget,NonTarget_label
######################################################################
def test_data_extraction_from_mat(Subject):
    path_mat = sio.loadmat('data/' + 'Subject_' + Subject + '_' + 'Test' + '.mat')
    Flashings = path_mat['Flashing'];Signals = path_mat['Signal']; StimulusCodes = path_mat['StimulusCode']
    if Subject == 'A':
        TargetChars = 'WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU'
    if Subject == 'B':
        TargetChars = 'MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR'
    return Flashings,Signals,StimulusCodes,TargetChars
def test_data_and_label(Subject,samps):
    
    Flashings,Signals,StimulusCodes,TargetChars = test_data_extraction_from_mat(Subject)
    Target = [];NonTarget = []
    for i in range(len(Flashings)):
        Flashing = Flashings[i];Signal = Signals[i];StimulusCode = StimulusCodes[i];TargetChar = TargetChars[i]
        Signal = bandpass(Signal,[0.1,20.0],240)
        loactions = ERP_location(Flashing)
        sgl_Target =Single_trial_target(TargetChar)
        loca_dicts = Single_local_information(loactions,StimulusCode)
        Target_ERPs,NonTarget_ERPs = Single_trial_ERP(Signal,sgl_Target,loca_dicts,samps)
        Target += [Target_ERPs]; NonTarget += [NonTarget_ERPs]
    Target = np.array(Target);NonTarget = np.array(NonTarget)
    Target = Target.reshape(-1,samps,64)
    NonTarget = NonTarget.reshape(-1,samps,64)
    Target_label = np.ones((Target.shape[0],1),dtype=np.int)
    NonTarget_label = np.zeros((NonTarget.shape[0],1),dtype=np.int)
    return  Target,Target_label,NonTarget,NonTarget_label