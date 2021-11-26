# STNN-for-P300-detection
Link: Spatial-Temporal Neural Network for P300 Detection.
# Introduction
* Spatial-Temporal Neural Network for P300 Detection, written in Pytorch.  
* This is a novel DL-based model for P300 detection, providing a validated biological EEG signal processing technology.  
* **We encourage to use of this model in various EEG processing tasks.**
# Requirements
* Python  
* Pytorch  
* scipy
* collections
* numpy
* random
# Materials
[BCI Competition III- dataset II](http://www.bbci.de/competition/iii) provided by Wadsworth Center, NYS Department of Health (Jonathan R. Wolpaw, Gerwin Schalk, Dean Krusienski)  The goal is to estimate to which letter of a 6-by-6 matrix with successively intensified rows resp. columns the subject was paying attention to; data from 2 subjects
36 classes, 64 EEG channels (0.1-60Hz), 240Hz sampling rate, 85 training and 100 test trials, recorded with the BCI2000 system 
# Run on the device
* Download Subject_A_Test.mat Subject_A_Train.mat Subject_B_Test.mat Subject_B_Train.mat from 
  [Data sets-Data set II: ‹P300 speller paradigm›](http://www.bbci.de/competition/iii)
* Create a new folder "data" and load the downloaded data.
* Run demo.ipynb
* Trust me, you will reach an accuracy greater than more than 90% after 1-5 mins
# Paper Citation
* Once the paper is accepted, the model **model.py** will be published as soon as possible.
