#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 13:21:33 2022

@author: pbaldwin
"""




import numpy as np
from sys import argv
import csv
import sys, os
import pandas as pd
from math import *
import matplotlib
import scipy
from scipy.special import eval_legendre;
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import glob;
#from  EMAN2 import *
import importlib
from matplotlib.colors import LinearSegmentedColormap

CurrentWD =  os.getcwd()

import socket

import argparse
import LibWidgetsNov2021
import logging

if 0: print('finished importing')

#%%

if 0:
    print('determined machines')
    
    MachineName =socket.gethostname() 
    print(MachineName)
    # Prime: Phil's  machine at Baylor
    if MachineName =='prb': os.chdir('/home/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # Prime: Windows Desktop West Cornwall
    if MachineName =='DESKTOP-7MEI3GK': os.chdir('D:/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # Prime: Inspiron Laptop from San Diego
    if MachineName =='DESKTOP-PNSJ06C': os.chdir('C:/Users/prbpr/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # Prime: Linux Machine from San Diego
    if MachineName =='phil-Precision-Tower-3620': os.chdir('/home/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # For West Cornwall Windows 10
    if 0: os.chdir('D:\pbaldwin\Dropbox\Salk17\GridDataOntoSphere\WidgetsForSamplingDevelopment')
    # For inertia
    if 0: os.chdir('/mnt/md0/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # For MacBook
    if 0: os.chdir('/Users/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')

   
    


#%%          Section 1;   Import Necessary Strings and Variables


DEFAULT_ROOT_STRING = 'SCFAnalysis'
C1=1;

#   January 2022

"""
     Usage

   python SCFJan2022.py AngleFileName.par 
      
  
    Creates a figure of the sampling in the same directory as Anglefilename.par with name based on the par file.

    --FourierRadius    one can change the integer value at which the Sampling is plotted, and the SCF evaluatte

    --NumberToUse      the default value is the minimum of 10000 or the total number in the file. One can try to increase this number
    
    --3DFSCMap  eventually we will look at correlations of the resolution and the sampling
    
    --RootOutputName the default for the logging file is 'SCFAnalysis.txt'. The root can be changed.

    --TiltAngle    the data can be tilted in silico...


"""
 
    

parser = argparse.ArgumentParser(description="Calculate SCF parameter and make plots. \
                                \t\t  Name of plot is based on input Angle File. \
                                \n The SCF is how much the SSNR has likely been attenuated due to projections not being distributed uniformly. \
                                \n The SCF, SCF* parameters are described here: \
                                \n Baldwin, P.R. and D. Lyumkis, Non-uniformity of projection distributions attenuates \
                            \n resolution in Cryo-EM.Prog Biophys Mol Biol, 2020. 150: p. 160-183.")

parser.add_argument("FileName"   ,         type=str, help="the name of the File of Angles; Psi Theta Rot in degrees  ")
parser.add_argument("--FourierRadius",    type=int,   default=50, help="Fourier radius (int) of the shell on which sampling is evaluated")
parser.add_argument("--NumberToUse",       type=int, default=10000, help="the number of projections to use, if you don't want to use all of them")
parser.add_argument("--3DFSCMap",        default='./', help=" the 3DFSC map, if one wants to correlate Sampling/Resolution; currently not implemented  ")
parser.add_argument("--RootOutputName", default=DEFAULT_ROOT_STRING, help=" the root name for logging outputs. Default is SCFAnalysis ")
parser.add_argument("--TiltAngle",       type=int,default=0, help="tilt angle")
parser.add_argument("--Sym",       type=str,default='', help="symmetry: Icos, Oct, Tet, Cn, or Dn. If tilt specified, then Sym =C1")



args                = parser.parse_args()

# print('finished constructing parser')
#%%

FileName             = args.FileName
FourierRadiusInt     = args.FourierRadius;  # Specify Fourier Radius
NumberToUse          = args.NumberToUse     # if we don't want to use all of them
RootOutputName       = args.RootOutputName  # This will access all the files from the FSC side 
ThreeDFSCMap         = args.RootOutputName  # If we want to look at correlations of ThreeDFSCMap
TiltAngle            = args.TiltAngle;      # This is the TiltAngle
SymNow               = args.Sym;         # This is Symmmetry but not number of C or D

FileNamewPath= os.path.join(CurrentWD,FileName);
print('FileNamewPath')  


if (TiltAngle>0) & (len(SymNow)>0):
    print('Symmetry ignored because Tilt was specified ')
    SymNow='C'
    
if len(SymNow)==0: SymNow='C'

NumberToUse=10000
AnswerBrowse =    LibWidgetsNov2021.browse_button(FileNamewPath, NumberToUse)

OutPutAngles          = AnswerBrowse[0]
AllProjPointsInitial  = AnswerBrowse[1]
AllProjPoints         = AnswerBrowse[2]
AllProjPointsB4Sym    = AnswerBrowse[3] 
NumberToUseInt        = AnswerBrowse[4]
 

#logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logging.basicConfig(filename=RootOutputName+'.txt', level=logging.INFO)

#_logger = logging.getLogger(__file__)
#_logger.setLevel(logging.INFO)

#%%

#Reconfigure Symmetry
ListOfKnownSymmetries = ['Icos','Oct','Tet']
CSymNow =1

if len(SymNow)>1:# Symmetry was declared, or it would be 'C'
    if (SymNow[0] =='C') | (SymNow[0]=='D'):
        CSymNowStr = SymNow[1:]
        if CSymNowStr.isdigit():
            CSymNow = np.int(CSymNowStr)
            SymNow = SymNow[0]
        else:
            print('Illegal C or D symmetry')
            sys.exit(0)
    elif SymNow not in ListOfKnownSymmetries:
        print('Unknown  symmetry')
        print('Available symmetries are Icos, Oct, Tet, Cn or Dn')
        sys.exit(0)
        
#%%

if SymNow in ['C','D']:
    print('Sym='+SymNow+' CSymNow='+str(CSymNow))
    logging.info('Symmetry='+str(SymNow)+str(CSymNow))
else:
    print('Sym='+SymNow)
    logging.info('Symmetry='+str(SymNow))
    
#sys.exit(0)    

logging.info('Tilt='+str(TiltAngle))

#%% 
TiltInDegB      =TiltAngle
SpiralVec, NumFib =  LibWidgetsNov2021.CreateSpiral(FourierRadiusInt)


NumberForEachTilt =90;
#TiltInDegB =30;
if 0:
    SymNow = 'C';

# print(SymNow+' at line 157')
Answer = LibWidgetsNov2021.get_InnerProdBooleanSum1(SpiralVec,OutPutAngles, \
                 FourierRadiusInt,NumberForEachTilt, \
               AllProjPointsB4Sym,TiltInDegB, SymNow,CSymNow)


InnerProdBooleanSum0 = Answer[0]
InnerProdBooleanSum1 = Answer[1]
AllProjAndTilts      = Answer[2]
TiltedNVecs          = Answer[3]
AllProjPoints        = Answer[4] 
#NVecs0       = Answer[5]
#
FigDpi =100
    
FolderPath = CurrentWD

Answer =  LibWidgetsNov2021.PlotSamp(InnerProdBooleanSum1 , FolderPath,  FigDpi,FourierRadiusInt,FileName,TiltInDegB )

   

GoodVoxels         = Answer[0]
RecipSamp          = Answer[1]
NOver2k            = Answer[2]
NumberOfZeros      = Answer[3]
FractionOfZeros    = Answer[4]  
FractionOfNonZeros = Answer[5]
QkoverPk           = Answer[6]
SCF0               = Answer[7]
SCFStar            = Answer[8]

      
logging.info('Number of zeros='+str(NumberOfZeros))
logging.info('Fraction of zeros='+str(FractionOfZeros))
logging.info('QkoverPk='+str(QkoverPk))
logging.info('SCF ='+str(SCF0))
logging.info('SCFStar='+str(SCFStar))
  
if 0:
    SCFConditionByTilt[jFile,jTilt] = SCF0
    SCFStarConditionByTilt[jFile,jTilt] = SCFStar
    FracZerosConditionByTilt[jFile,jTilt] = FractionOfZeros

#print(SCFStar)

