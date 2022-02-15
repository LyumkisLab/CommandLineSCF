# -*- coding: utf-8 -*-
"""
Created on Tue Dec 14 05:27:25 2021

@author: Philip Baldwin
"""
#os.chdir('D:/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment/')
#get_ipython().run_line_magic('matplotlib', 'inline')



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



#%%
import socket

if 0:
    # Prime: Phil's  machine at Baylor
    if socket.gethostname() =='prb': os.chdir('/home/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # Prime: Inspiron Laptop from San Diego
    if socket.gethostname() =='DESKTOP-PNSJ06C': os.chdir('C:/Users/prbpr/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # Prime: Linux Machine from San Diego
    if  socket.gethostname() =='phil-Precision-Tower-3620': os.chdir('/home/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # For West Cornwall Windows 10
    if 0: os.chdir('D:\pbaldwin\Dropbox\Salk17\GridDataOntoSphere\WidgetsForSamplingDevelopment')
    # For inertia
    if 0: os.chdir('/mnt/md0/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')
    # For MacBook
    if 0: os.chdir('/Users/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/WidgetsForSamplingDevelopment')

from time import *
import shutil;
from math import *
#from numbapro import cuda
#from numba import *
#import SphereLibrary 
#import LibSphere 
# from SphereLibraryB import *
import tkinter as tk
#import IPython.nbformat.current as nbf
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#import tkdocviewer
from matplotlib.colors import LinearSegmentedColormap
import LibSphereWidgets
import mrcfile


#%%

CWD= os.getcwd()
#CWD='/home/pbaldwin/Dropbox/GroupArticles/Salk/JPBMTiltwDmitry/Data/ribosome, April 2020'

if 0: print(CWD)

#%%
if 0:

    try:
        os.chdir('/Users/pbaldwin/Dropbox/Salk17/OneOverSp/')
    except:
        try:
            os.chdir('C:\\Users\\phili\\Dropbox\\Salk17\\OneOverSp\\')
        except:
            os.chdir('/mnt/md0/pbaldwin/Dropbox/Salk17/OneOverSp/')
    
    
    import LibSphereSalkJuly28th2019 as libSphereGeneral


#%%

if 0:
    
    try:
        os.chdir('/Users/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/')
    except:
        try:
            os.chdir('C:\\Users\\phili\\Dropbox\\Salk17\\GridDataOntoSphere/')
        except:# For Andromeda
            os.chdir('/mnt/md0/pbaldwin/Dropbox/Salk17/GridDataOntoSphere/')
    
    import LibSphere
    
#if 0:
#    nb = nbf.read(open('WidgetsForSampling.py', 'r'), 'py')
#    nbf.write(nb, open('test.ipynb', 'w'), 'ipynb')


#%%

os.chdir(CWD)


#%%    Begin Functions for Phil's Sampling and Tilt Tool

def AngleFile2nVec(AngleFileName):
    #print(os.getcwd())
    #return
    OutPutAngles = np.loadtxt(AngleFileName)
    NumAngles = np.shape(OutPutAngles)[0]
    #ZerosVec = np.zeros(NumAngles)
    psiVec   = OutPutAngles[:,0]*np.pi/180.0;
    thetaVec = OutPutAngles[:,1]*np.pi/180.0;
    phiVec  = OutPutAngles[:,2]*np.pi/180.0;

    nVecx = np.sin(thetaVec)*np.cos(phiVec)
    nVecy = np.sin(thetaVec)*np.sin(phiVec)
    nVecz = np.cos(thetaVec)
    #The angle file, FileNameOfAngles, should be formatted as three columns for the Z Y Z convention:
    #that is 'PSI', 'THETA', 'PHI'.
    #The first column is irrelevant for the purposes of the sampling. It is PSI. The second column is the tilts: which probably is positive and ranges from 0 to 180.

    nVec = np.vstack((nVecx, nVecy,nVecz)).T
    return nVec,OutPutAngles
    # nVec =  AngleFile2nVec(AngleFileName)


#%%   browse_button

def browse_button(filename, NumberToUse):
    # 0. Allow user to select a directory and store it in global var
    # called folder_path
    # 1. Reads in AllProjPoints
    #
    if 0:
        global filedir,folder_path
        global OutPutAngles, AllProjPointsInitial,AllProjPoints,AllProjPointsB4Sym
        global sAllProjPointsInitial, sAllProjPoints,sAllProjPointsB4Sym
        global NumberToUseInt, GraphedAlready
    #import LibSphere
    import random

    GraphedAlready=0
    #  filedir= filedialog.askdirectory(parent=win,initialdir='.',title='Navigate to txt files')
    #  os.chdir(filedir)
    
    if 0:
        filename = filedialog.askopenfilename(title='Pick three column angle file',filetypes = (("txt files","*.txt"),("par files","*.par"),("all files","*.*")))
    print(filename)

    fnparts=filename.split('/')
    #sfnparts=len(fnparts)

    if 0:
        filedir='/'
        for jparts,filestr in enumerate(fnparts[1:-1]):
            filedir+=filestr+'/'
        os.chdir(filedir)
        
        #root.filename =  filedialog.askopenfilename(title = "Select file",filetypes = (("txt files","*.txt"),("all files","*.*")))
        filenameNow = fnparts[-1]
    if 0:
        folder_path.set(filenameNow)
    #AllProjPoints,OutPutAngles =  LibSphere.AngleFile2nVec(filename)
    
    AllProjPointsInitial,OutPutAngles =  AngleFile2nVec(filename)
    sAllProjPointsInitial= np.shape(AllProjPointsInitial)[0]
    print('From Browsebutton; sAllProjPointsInitial = %g'%sAllProjPointsInitial)
    
    if 0: 
        NumberReadIn.set('NumberReadIn= %g' %sAllProjPointsInitial)
    NumberReadIn = sAllProjPointsInitial
 
    if 1:
        if sAllProjPointsInitial>10000:
            Rands= random.sample(range(0, sAllProjPointsInitial), 10000)
            AllProjPoints = AllProjPointsInitial[Rands,:]
            sAllProjPoints = np.shape(AllProjPoints)[0]
        else:
            AllProjPoints = AllProjPointsInitial
            sAllProjPoints=sAllProjPointsInitial
    
   
    if 0:
        eps=.12
        for jVec in range(sAllProjPoints):
            vv=eps*np.random.randn(3)
            PointNow = AllProjPoints[jVec,:]
            ww = PointNow +vv
            PointNow = ww/np.linalg.norm(ww)
            AllProjPoints[jVec,:]=PointNow
        

    if 0: 
        lblNumberToUse.config(text="NumberToUse="+str(sAllProjPoints))
    NumberToUseInt = sAllProjPoints
    #NumberToUse.set('NumberReadIn= %g' %sAllProjPoints)
    if 0: NumberToUse.set('%g' %NumberToUseInt)
    NumberToUse = NumberToUseInt
    #AllProjPoints = np.asarray(AllProjPoints)
    #PlotProj()
    #PlotSamp()
    #print(nVec[0,:])
    
    AllProjPointsB4Sym  =  AllProjPoints.copy()
    sAllProjPointsB4Sym =  sAllProjPoints

    return  OutPutAngles, AllProjPointsInitial,AllProjPoints, \
        AllProjPointsB4Sym, NumberToUseInt 



#%%    This is for FSC_Map Map

      
#%%    Correct FSC curves; plot corrected vs uncorrected; save data; not currently active
def CorrectFSC_Curves():          

    global   SpiralVecRound,Map3DFSCSpiral;
    
    
    #[nx1,ny1,nz1] =Map1Data.shape
    #[nx2,ny2,nz2] =Map2Data.shape
    [nx,ny,nz] =Map3DFSC.shape
    nxOver2= nx//2

    SpiralVecRound=np.rint(SpiralVec*FourierRadiusInt).astype(int)
    Map3DFSCSpiral = np.zeros(NumFib)
    
    for jFib in range(NumFib):
        SpiralVecNow = SpiralVecRound[jFib,:]+nx//2;
        Map3DFSCNow = Map3DFSC[SpiralVecNow[0],SpiralVecNow[1],SpiralVecNow[2]]
        Map3DFSCSpiral[jFib] = Map3DFSCNow
        
    print('Finished Map Along Spiral')
    
#%%
def CreateSpiral(FourierRadiusInt):
    
    import LibSphereWidgets
    global SpiralVec, NumFib
          
    forceNumberofRaystothisinteger = np.int(2*np.pi*FourierRadiusInt*FourierRadiusInt);
    if 1:
        forceNumberofRaystothisinteger = np.min([forceNumberofRaystothisinteger,5000])
    #nVec = LibSphere.GetIFibDirections(deltaTheta,forceNumberofRaystothisinteger)
    #SpiralVec = LibSphere.GetIFibDirections_Now(forceNumberofRaystothisinteger)
    SpiralVec = LibSphereWidgets.GetIFibDirections_Now(forceNumberofRaystothisinteger)
    
    ThetaVec = np.linspace(0,np.pi/2,forceNumberofRaystothisinteger)
    PhiVec = np.random.rand(forceNumberofRaystothisinteger)*2*np.pi
    
    XVec=np.sin(ThetaVec)*np.cos(PhiVec)
    YVec=np.sin(ThetaVec)*np.sin(PhiVec)
    ZVec=np.cos(ThetaVec)
    
    
    if 0:
        SpiralVec = np.vstack((XVec,YVec,ZVec)).T

    
    NumFib=np.shape(SpiralVec)[0]

    return SpiralVec, NumFib

    #The angle file, FileNameOfAngles, should be formatted as three columns for the Z Y Z convention:
    #that is 'PSI', 'THETA', 'PHI'.
    #For the sampling program, the first column is irrelevant for the purposes of the sampling. It is the PSI angle. The second column is the tilts: which probably is positive and ranges from 0 to 180.

#%%

def PlotProj():
    global GraphedAlready
    global axProj,figureProj
    
    #if sAllProjPointsB4Sym<sAllProjPoints: #a symmetry has been defined
        
    CosTheta = np.asarray(AllProjPoints[:,2])
    CosThetaB = CosTheta.copy()
    IndsZZneg = np.where(CosTheta<0)[0]
    
    XX = np.asarray(AllProjPoints[:,0])
    YY = np.asarray(AllProjPoints[:,1])
    
    XXB= XX.copy()
    YYB= YY.copy()
    
    XXB[IndsZZneg] *=-1
    YYB[IndsZZneg] *=-1
    CosThetaB[IndsZZneg] *=-1
    
    CosTheta2 = CosThetaB*CosThetaB
    SinThetaB = np.sqrt(1-CosTheta2)
    ThetaB = np.arcsin(SinThetaB)
    FuncThetaB = ThetaB/(np.pi/2)
    FuncThetaB = np.sin(ThetaB/2)/(np.sin(np.pi/4))
    
    XXFinal=XXB.copy()
    YYFinal=YYB.copy()

    GoodInds= np.where(np.abs(SinThetaB)>.0001)[0]
    XXFinal[GoodInds] = XXB[GoodInds]*FuncThetaB[GoodInds]/SinThetaB[GoodInds]
    YYFinal[GoodInds] = YYB[GoodInds]*FuncThetaB[GoodInds]/SinThetaB[GoodInds]

    if GraphedAlready==0:
        global axProj, axProjScatter,canvasNow

        deltaTheta= 90; # 3.8
        #LP1LP2Over2 = (LMax+1)*(LMax+2)//2;
         
        # Do Proj Plot
    #   # Create  plot of sampling
        figureProj = plt.Figure( figsize=(3.5,3.5), dpi=FigDpi)
        canvasNow = FigureCanvasTkAgg(figureProj, win)
        #canvasNow.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        
        if 0:
            axProj = figureProj.add_subplot(111,projection='3d')
            #df1.plot(kind='bar', legend=True, ax=ax1)
            #axProj.plot(np.arange(10))
            axProjScatter = axProj.scatter(AllProjPoints[:,0], AllProjPoints[:,1], \
                AllProjPoints[:,2], s=1, c='b', marker='.')
            axProj.set_title('Original Proj Directions')
            #figureProj.set_xlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
            axProj.set_xlim(-1, 1)                    # viewrange for z-axis should be [-4,4] 
            axProj.set_ylim(-1, 1)                    # viewrange for z-axis should be [-4,4] 
            axProj.set_zlim(-1, 1)                    # viewrange for z-axis should be [-4,4] 
        else:
            axProj = figureProj.add_subplot(111)
            if sAllProjPoints>10000000:
                axProjScatter = axProj.hist2d(XXFinal.flatten(),YYFinal.flatten(),bins=40)
            else:
                axProjScatter = axProj.scatter(XXFinal.flatten(),YYFinal.flatten(), \
                                               s=1, c='b', marker='.')
            axProj.set_title('Original Proj Dirs')
            #figureProj.set_xlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
            axProj.set_xlim(-1.1, 1.1)                    # viewrange for z-axis should be [-4,4] 
            axProj.set_ylim(-1.1, 1.1)                    # viewrange for z-axis should be [-4,4] 
            
        canvasNow.get_tk_widget(). place(relx=0.7, rely=0.0066);
        print('Not graphed Before')
    if GraphedAlready==1:
        #axProjScatter._offsets3d = (AllProjPoints[:,0], AllProjPoints[:,1], AllProjPoints[:,2])
        axProj.cla()
        if sAllProjPoints>10000000:
            axProjScatter = axProj.hist2d(XXFinal.flatten(),YYFinal.flatten(),bins=40)
        else:
            axProjScatter = axProj.scatter(XXFinal.flatten(),YYFinal.flatten(), \
                                           s=1, c='b', marker='.')
        axProj.set_title('Original Proj Directions')
        #figureProj.set_xlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
        axProj.set_xlim(-1.1, 1.1)                    # viewrange for z-axis should be [-4,4] 
        axProj.set_ylim(-1.1, 1.1)                    # viewrange for z-axis should be [-4,4] 
        canvasNow.draw()
        print('Graphed again')

    #canvasNow.draw()
    #plt.plot(InnerProdBooleanSum1)
    #plt.hist2d(XXFinal.flatten(),YYFinal.flatten(),bins=80)

#%%

def PlotCorrectedFSCNaive():
    global GraphedAlreadyFSCNaive
    global axFSCnaive,figureFSCnaive
    if GraphedAlreadyFSCNaive==0:
        global axFSCnaiveLine,canvasNowFSCnaive

        figureFSCnaive = plt.Figure( figsize=(3,3), dpi=FigDpi)
        axFSCnaive = figureFSCnaive.add_subplot(111)
        canvasNowFSCnaive = FigureCanvasTkAgg(figureFSCnaive, win)
        canvasNowFSCnaive.get_tk_widget().place(relx=0.78, rely=0.186);
        #canvasNow.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        axFSCnaiveLine = axFSCnaive.plot(InvAngstroms,FSCObs,'k',InvAngstroms,FSCstar,'r--')
        axFSCnaive.set_title('original and corrected FSC (using 0s) vs spat freq', fontsize=7)
        axFSCnaive.set_xlabel('Spatial Frequency (Inv Angs)')        
        axFSCnaive.set_ylabel('FSC')        
        print('Not graphed Before')
        canvasNowFSCnaive.draw()
        GraphedAlreadyFSCNaive==1
        
    if GraphedAlreadyFSCNaive==1:
        #axProjScatter._offsets3d = (AllProjPoints[:,0], AllProjPoints[:,1], AllProjPoints[:,2])
        axFSCnaive.cla()
        axFSCnaiveLine = axFSCnaive.plot(InvAngstroms,FSCObs,'k',InvAngstroms,FSCstar,'r--')
        axFSCnaive.set_title('original FSC and corrected FSC (using zeros) vs spatial frequency')
        axFSCnaive.set_xlabel('Spatial Frequency (Inverse Angstroms)')        
        axFSCnaive.set_ylabel('FSC')        
        print('Not graphed Before')
        canvasNowFSCnaive.draw()
        print('Graphed again')

    #canvasNow.draw()
    #plt.plot(InnerProdBooleanSum1)
    #plt.hist2d(XXFinal.flatten(),YYFinal.flatten(),bins=80)


#%%

def PlotCorrectedFSCAggressive():
    global GraphedAlreadyFSCaggro
    global axFSCaggro,figureFSCaggro,axFSCaggroLine
    import datetime
    
    today = datetime.date.today()
    d4 = today.strftime("%b-%d-%Y")
 
    if GraphedAlreadyFSCaggro==0:
        global axFSCaggroLine,canvasNowFSCaggro

        figureFSCaggro = plt.Figure( figsize=(3,3), dpi=FigDpi)
        axFSCaggro = figureFSCaggro.add_subplot(111)
        canvasNowFSCaggro = FigureCanvasTkAgg(figureFSCaggro, win)
        canvasNowFSCaggro.get_tk_widget(). place(relx=0.76, rely=0.99+TiltHMShift);
        #canvasNow.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
        axFSCaggroLine = axFSCaggro.plot(InvAngstromsAgro,FSCObsAgro,'k',\
                                         InvAngstromsAgro,FSCstarAgro,'r--')
        axFSCaggro.set_title('original and corrected FSC (using 3DFSC) vs spat freq', fontsize=7)
        axFSCaggro.set_xlabel('Spatial Frequency (Inv Angs)')        
        axFSCaggro.set_ylabel('FSC')   
        axFSCaggro.legend(['Original FSC','Corrected FSC (EofK)'])
        print('Not graphed Before')
        canvasNowFSCaggro.draw()
        GraphedAlreadyFSCaggro==0
        
        files = ['','*'] 
        #FSCPics = tk.filedialog.asksaveasfile( title='Save picture of original FSC and corrected FSC vs spatial frequency ( do not add extension) ',filetypes = files, defaultextension = files) 
        FSCPics = tk.filedialog.asksaveasfile( title='Save picture of original FSC and corrected FSC vs spatial frequency ( do not add extension) ') 
        OutName = FSCPics.name+d4+'.jpg'
        figureFSCaggro.savefig(OutName)
        print('Finished writing out '+OutName)
        
    if GraphedAlreadyFSCaggro==1:
        #axProjScatter._offsets3d = (AllProjPoints[:,0], AllProjPoints[:,1], AllProjPoints[:,2])
        axFSCaggro.cla()
        axFSCaggroLine = axFSCaggro.plot(InvAngstroms,FSCObs,'k',InvAngstroms,FSCstar,'r--')
        axFSCaggro.set_title('original and corrected FSC (using 3DFSC, Eofk) vs spatial frequency')
        axFSCaggro.set_xlabel('Spatial Frequency (Inverse Angstroms)')        
        axFSCaggro.set_ylabel('FSC')        
        axFSCaggro.legend(['Original FSC','Corrected FSC (EofK)'])
        canvasNowFSCaggro.draw()
        print('Graphed again')
    

    #canvasNow.draw()
    #plt.plot(InnerProdBooleanSum1)
    #plt.hist2d(XXFinal.flatten(),YYFinal.flatten(),bins=80)

#%%

def get_InnerProdBooleanSum1(SpiralVec,OutPutAngles,FourierRadiusInt, \
                             NumberForEachTilt,AllProjPointsB4Sym,TiltInDegB, SymNow, CSymNow=1):
        
    # Uses
    import LibSphereWidgets
    
    #if 0:
        #global InnerProdBooleanSum0, InnerProdBooleanSum1
        #global NVecs0, AllProjAndTilts,TiltedNVecs
        #global AllProjPoints,sAllProjPoints,sAllProjPointsB4Sym
    
    
    AllProjPoints  = AllProjPointsB4Sym.copy()
    sAllProjPoints = sAllProjPointsB4Sym
    
    sVec = np.shape(AllProjPoints)[0]
    
    PsiVec   = OutPutAngles[:,0]; 
    #PsiVec   = np.linspace(0,360,sVec)
    PsiVec   = 360.0*np.random.rand(sVec)
    #PsiVec   = 0*np.ones(sVec)
    ThetaVec = OutPutAngles[:,1]
    PhiVec   = OutPutAngles[:,2]
    
    if 0: print(SymNow+' at line 480, LibWidgetsNov2021')
    
    if TiltInDegB>0:
        TiltedNVecs= np.zeros_like(AllProjPoints)
        
        CTilt=  np.cos(TiltInDegB*np.pi/180.0)
        STilt=  np.sin(TiltInDegB*np.pi/180.0)
        TiltMat = np.array([[CTilt,0,-STilt],[0,1,0],[STilt,0,CTilt]])
        
        CPsi = np.cos(PsiVec*np.pi/180.0)
        SPsi = np.sin(PsiVec*np.pi/180.0)
     
        CTheta = np.cos(ThetaVec*np.pi/180.0)
        STheta = np.sin(ThetaVec*np.pi/180.0)
        
        CPhi = np.cos(PhiVec*np.pi/180.0)
        SPhi = np.sin(PhiVec*np.pi/180.0)
           
        
        for jAngle in range(sVec):
            
            CNow     = CPsi[jAngle] ;   SNow = SPsi[jAngle] ;
            PsiMat   = np.array([[CNow,SNow,0],[-SNow,CNow,0],[0,0,1]])
            
            CNow     = CTheta[jAngle] ; SNow = STheta[jAngle] ;
            ThetaMat = np.array([[CNow,0,-SNow],[0,1,0],[SNow,0,CNow]])
            
            CNow     = CPhi[jAngle]  ; SNow = SPhi[jAngle] ;
            PhiMat   = np.array([[CNow,SNow,0],[-SNow,CNow,0],[0,0,1]])
            
            TiltPsi  = np.dot(TiltMat,PsiMat)
            ThetaPhi = np.dot(ThetaMat,PhiMat)
            
            All4 = np.dot(TiltPsi,ThetaPhi);
            
            TiltedNVecs[jAngle,:]=All4[2,:]
  
        InnerProdBooleanSum0,InnerProdBooleanSum1,AllProjAndTilts = \
            LibSphereWidgets.ReturnSamplingLoop(SpiralVec,TiltedNVecs,FourierRadiusInt,0,NumberForEachTilt)
        if 0: print('Finished Calculating InnerProdBooleanSum1 at 518 in PlotProj')
        return   InnerProdBooleanSum0, InnerProdBooleanSum1, AllProjAndTilts,TiltedNVecs, AllProjPoints
      
        

    if SymNow   == "Icos":
        #NVecs0= libSphereGeneral.AnglesToVecsIcosahedral(AllProjPoints,-1);
        NVecs0= LibSphereWidgets.AnglesToVecsIcosahedral(AllProjPoints,-1);
    elif SymNow == "Oct":
        NVecs0= LibSphereWidgets.AnglesToVecsOctahedral(AllProjPoints,-1);
    # PRB needs to import this from Relion files
    elif SymNow == "Tet" :
        if 0: print('Tet at line 531')
        NVecs0= LibSphereWidgets.AnglesToVecsTetrahedral(AllProjPoints,-1);
    elif SymNow == "D":
        CorD=2;
        NVecs0= LibSphereWidgets.AnglesToVecsCD(AllProjPoints,-1,CSymNow,CorD)# CorD: 1 means C, 2 means D
    elif SymNow == "C":
        CorD=1;
        NVecs0= LibSphereWidgets.AnglesToVecsCD(AllProjPoints,-1,CSymNow,CorD)# CorD: 1 means C, 2 means D
    else:
        print('Symmetry was not recorded')
        #print('CSym = '+str(CSym))
    
    NVecs0  =  np.asarray(NVecs0)
    sNVecs0  =  np.shape(NVecs0)[0]
    TiltedNVecs = NVecs0.copy()
    
    AllProjPoints = NVecs0.copy(); # AllProjPoints gets redefined here to the symmetrized version
    sAllProjPoints = sNVecs0
   
#    InnerProdBooleanSum0,InnerProdBooleanSum1,AllProjAndTilts = \
#        LibSphere.ReturnSampling(SpiralVec,NVecs0,FourierRadiusInt,TiltInDegB,NumberForEachTilt)
    print('Examining sAllProjPoints=%g'%np.shape(AllProjPoints)[0])
  
    InnerProdBooleanSum0,InnerProdBooleanSum1,AllProjAndTilts = \
        LibSphereWidgets.ReturnSamplingLoop(SpiralVec,NVecs0,FourierRadiusInt,0,NumberForEachTilt)
        
    if 0: print('Finished Calculating InnerProdBooleanSum1 at 553 in PlotProj')
    
    return  InnerProdBooleanSum0, InnerProdBooleanSum1, AllProjAndTilts,TiltedNVecs, AllProjPoints
    
#%%
    
def MapSamplingToColor(SpiralData,FR):
    # We want to map 0 to 0
    #  linear
    #   sqrt(mean) to 1/3
    #    log
    #   mean to 2/3
    #    log
    #    N to   1
    
    ColorVec = np.zeros_like(SpiralData)
    
    meanSpiral =  np.mean(SpiralData)
    minSpiral  =  np.min(SpiralData)
    maxSpiral  =  np.max(SpiralData)
    
    N = 2* FR* meanSpiral;
    rootMeanSpiral = np.sqrt(meanSpiral)
    if minSpiral > 0:
        point1 = minSpiral
    else:
        point1 = np.min([np.sqrt(meanSpiral),meanSpiral])
        
    point2 =  np.max([np.sqrt(meanSpiral),meanSpiral])
    point3 = maxSpiral
    
    
    
    vv01 = np.where(SpiralData <= point1)[0]
    vv12 = np.where((SpiralData > point1)&(SpiralData<=point2))[0]
    vv23 = np.where((SpiralData > point2)&(SpiralData<=point3))[0]
    vv3  = np.where(SpiralData > point3)[0]
    
    
    ColorVec[vv01]= SpiralData[vv01]/point1/3;
    ColorVec[vv12]= 1/3 * (1+  np.log(SpiralData[vv12]/point1)/(np.log(point2/point1)));
#   The following line was the original line, but incorrect; modified November 18th, 2020
#    ColorVec[vv23]= 1/3 * (1 + 2*np.log(SpiralData[vv23]/point2)/(np.log(point3/point2))); 
    ColorVec[vv23]= 1/3 * (2 + np.log(SpiralData[vv23]/point2)/(np.log(point3/point2)));
    ColorVec[vv3]= 1;

    return ColorVec,point1,point2,point3
# the 3 points should correspond to 1/3 , 2/3, 1
    
#%%
        
def PlotSamp(InnerProdBooleanSum1 , FolderPath,  FigDpi,FourierRadiusInt, FileName,TiltInDegB):
#   # Create  plot of sampling
    
    from matplotlib import cm
    
    os.chdir(FolderPath)

    #global SCF0, SCFStar, GoodVoxels, figureSamp
    
    print('Started  PlotSamp')
    
    figureSamp = plt.figure(figsize=(4.5,3), dpi=FigDpi)
    axSamp = figureSamp.add_subplot(111,projection='3d')
    #bar1 = FigureCanvasTkAgg(figure1, win)
    
    
    if 0:
        bar1 = FigureCanvasTkAgg(figureSamp, win)
        bar1.get_tk_widget().place(relx=0.3, rely=0.0066);#changed from relx=0.3 Nov 22nd 2021 PRB
    #df1.plot(kind='bar', legend=True, ax=ax1)
    #axProj.plot(np.arange(10))
    
    colors = [(1, 0, 0), (1,1,0),(0,1,0), (0, 0, 1)]  # R -> Y -> B
    colors = [(1, 0, 0), (1,1,0), (0, 0, 1)]  # R -> Y -> B
    n_bin = 30   # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    # Create the colormap
    cmPRB = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    SpiralData = np.asarray(InnerProdBooleanSum1).flatten()
    DataOnSpiral,pnt1,pnt2,pnt3 = MapSamplingToColor(SpiralData,FourierRadiusInt)
  
    precision=2;
    
    pnt0str = "{:.{}f}".format( 0, precision )
    pnt1str = "{:.{}f}".format( pnt1, precision )
    pnt2str = "{:.{}f}".format( pnt2, precision )
    pnt3str = "{:.{}f}".format( pnt3, precision )
    
    
    IndMin = np.argmin(DataOnSpiral); DataOnSpiral[IndMin]=0
    #IndMax = np.argmax(DataOnSpiral); DataOnSpiral[IndMax]=0
    
    #axSamp.scatter(SpiralVec[:,0], SpiralVec[:,1], SpiralVec[:,2], \
    axSampScatter =axSamp.scatter(SpiralVec[:,0], SpiralVec[:,1], SpiralVec[:,2], \
                   c=DataOnSpiral, \
                   marker='o',cmap=cmPRB); # cmap='RdBu',cm.coolwarm_r
    axSamp.set_title('Sampling Map; FR='+str(FourierRadiusInt))
    
    SaveFigName = FileName[:-4] +'Tilt'+str(int(TiltInDegB))+'.jpg'
    plt.savefig(SaveFigName, dpi=FigDpi)
    #plt.show(axSampScatter)
    cbar = figureSamp.colorbar(axSampScatter,ticks=[0,1/3,2/3,1],pad=0.2)
    cbar.ax.set_yticklabels([pnt0str,pnt1str, pnt2str, pnt3str])  # vertically oriented colorbar
    #print('Made Changes')
    plt.savefig(SaveFigName, dpi=FigDpi)

    GoodVoxels= np.where(InnerProdBooleanSum1>0)[0];
    
    RecipSamp = 1./InnerProdBooleanSum1[GoodVoxels]
    NOver2k = np.mean(InnerProdBooleanSum1)
    NumberOfZeros = NumFib -len(GoodVoxels);
    FractionOfZeros = NumberOfZeros/NumFib
    FractionOfNonZeros = 1- NumberOfZeros/NumFib
    QkoverPk =FractionOfZeros/FractionOfNonZeros

    SCF0=  1/(np.mean(RecipSamp)*NOver2k )
    if 0: SCF0_value.set('SCF= %.3f' % SCF0)   
    
    if 0: FracOfZeros_value.set('FractionOfZeros= %.2f  percent' % (100*FractionOfZeros))   

    SCFStar=  SCF0/(1+ QkoverPk*NOver2k*SCF0 );  # This was changed  April 2020
    SCFStar=  SCF0/(1+ QkoverPk*NOver2k*SCF0/2 ) #  
    if 0: SCFStar_value.set('SCF*= %.3f' % SCFStar)   
    
    print('Number Of Zeros = ' + str(NumberOfZeros))
    print('FractionOfZeros= %.3f' % FractionOfZeros)
    print('SCF = %.3f' % SCF0)
    print('SCF*= %.3f' % SCFStar)
    print('Finished  PlotSamp')

    #figureProj.set_xlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    #figureProj.set_ylim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    #figureProj.set_zlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 

    return GoodVoxels ,  RecipSamp , NOver2k ,  NumberOfZeros,  \
        FractionOfZeros ,  FractionOfNonZeros,  QkoverPk, SCF0, SCFStar 

    #axProj.scatter(SpiralVec[:,0], SpiralVec[:,1], SpiralVec[:,2], c=InnerProdBooleanSum1, marker='o')
#%%

def PlotFracZeros():
    import LibSphereWidgets
    global figureFracZeros,FractionOfZerosArray
    FRMax= 60
    FourierRadiusRange= np.arange(FRMax)
    FractionOfZerosArray= np.zeros(FRMax)
    
    FractionOfZerosArray[0] =0
    
    
    for Count,FourierRadiusNow in enumerate(FourierRadiusRange):
        if Count ==0: continue
        TiltInDegB=0;
        forceNumberofRaystothisintegerNow = np.int(2*np.pi*FourierRadiusNow*FourierRadiusNow);
        #SpiralVecNow = LibSphere.GetIFibDirections_Now(forceNumberofRaystothisintegerNow)
        SpiralVecNow = LibSphereWidgets.GetIFibDirections_Now(forceNumberofRaystothisintegerNow)
        NumFibNow=np.shape(SpiralVecNow)[0]
       
        InnerProdBooleanSum0Now,InnerProdBooleanSum1Now,AllProjAndTiltsNow = \
            LibSphereWidgets.ReturnSampling(SpiralVecNow,NVecs0,FourierRadiusNow, \
                                     TiltInDegB,NumberForEachTilt)
            
        GoodVoxelsNow= np.where(InnerProdBooleanSum1Now>0)[0];
        NumberOfZerosNow = NumFibNow -len(GoodVoxelsNow);
        FractionOfZerosNow = NumberOfZerosNow/NumFibNow
        FractionOfZerosArray[Count] = FractionOfZerosNow
        
    #plt.plot(FractionOfZerosArray)
    figureFracZeros = plt.Figure( figsize=(3,3), dpi=FigDpi )
    axFracZeros = figureFracZeros.add_subplot(111)
    #bar1 = FigureCanvasTkAgg(figure1, win)
    #barFracZeros = FigureCanvasTkAgg(figureFracZeros, lblFracZerosPlot)
    barFracZeros = FigureCanvasTkAgg(figureFracZeros, win)
    #barFracZeros.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
    barFracZeros.get_tk_widget().place(relx=0.75,rely=.186)
    #df1.plot(kind='bar', legend=True, ax=ax1)
    #axProj.plot(np.arange(10))
    axFracZerosLine = axFracZeros.plot(FourierRadiusRange,FractionOfZerosArray)
    axFracZeros.set_title('Fraction Unfilled vs FR')
    #figureProj.set_xlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    #axFracZeros.set_xlim(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    axFracZeros.set_xlabel('Fourier Radius')        
    axFracZeros.set_ylabel('Frac Of Zeros')        
    if 0: 
        np.savez('FractionOfZerosArray',FractionOfZerosArray)
    barFracZeros.draw()


#%%       
def PlotHistSamp():
    global figureHistSamp, axHistSamp
#   # Create  plot of sampling
    figureHistSamp = plt.Figure(figsize=(3.5,2.5), dpi=FigDpi)
    axHistSamp = figureHistSamp.add_subplot()
    #bar1 = FigureCanvasTkAgg(figure1, win)
    bar1 = FigureCanvasTkAgg(figureHistSamp, win)
    bar1.get_tk_widget().place(relx=0.32, rely=0.186)
    #df1.plot(kind='bar', legend=True, ax=ax1)
    #axProj.plot(np.arange(10))
    if 1:
        axHistSamp.hist(InnerProdBooleanSum1,100,log=True)
        axHistSamp.set_title('log hist of Sampling Map vals; FR='+str(FourierRadiusInt))
        axHistSamp.set_xlim(0, 1.1*np.max(InnerProdBooleanSum1))   # viewrange for z-axis should be [-4,4]
    if 0:
        plt.hist(InnerProdBooleanSum1,100,log=True)
        plt.title('log hist of Sampling Map vals; FR='+str(FourierRadiusInt))
        plt.xlim(0, 1.1*np.max(InnerProdBooleanSum1))   # viewrange for z-axis should be [-4,4]
    #figureProj.set_ylim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    #figureProj.set_zlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 


    #axProj.scatter(SpiralVec[:,0], SpiralVec[:,1], SpiralVec[:,2], c=InnerProdBooleanSum1, marker='o')
    
#%%
if 0:
    def change_SymDropdownB(*args):
        SymNowB = varSymB.get();
        SymNowTotalB = SymNowB;
        #lblSym.setvar("Symmetry = ")

#%%
def change_SymDropdown(*args):
    global SymNow
    SymNow = varSym.get();
    #SymNowTotal = SymNow;
    print( SymNow )
    if 0:
        ChoicesSymB = { '1','2','3','4','5' }
        if SymNow=='C':
            global varSymB
            varSymB       = tk.StringVar()
            varSymB.set('1') # set the default option
            popupMenuSymB = tk.OptionMenu(win,varSymB,*ChoicesSymB)
            varSymB.trace('w', change_SymDropdownB)
            print(varSymB)
        
#%%
def change_CSymDropdown(*args):
    global CSymNow
    CSymNow = int(varCSym.get());
    print( CSymNow )
    #SymNowTotal = SymNow;

    #lblSym.setvar("Symmetry = ")

# link function to change dropdown
#%%   What happens when NumberToUse Entry box is filled

def Get_Num2Use_Entry(event=None):
    import random
    global NumberToUseInt
    global AllProjPoints, sAllProjPoints
    NumberToUseInt= int( EntryNumberToUse.get() )
    NumberToUseInt = np.min([NumberToUseInt,sAllProjPointsInitial])
    
    NumberToUse.set(str(NumberToUseInt))
    EntryNumberToUse.icursor(0)
    lblNumberToUse.config(text="NumberToUse="+str(NumberToUseInt))
    
    Rands= random.sample(range(0, sAllProjPointsInitial), NumberToUseInt)
    
    AllProjPoints = AllProjPointsInitial[Rands,:]
    sAllProjPoints = np.shape(AllProjPoints)[0]

    
    print(NumberToUseInt)

#%%   What happens when Get_FourierRadius_Entry is filled

def Get_FourierRadius_Entry(event=None):
    global FourierRadiusInt
    FourierRadiusInt= int( EntryFourierRadius.get() )
    FourierRadiusStr.set(str(FourierRadiusInt))
    lblFourierRadiusA.config(text="FourierRadius = "+str(FourierRadiusInt))
    EntryFourierRadius.icursor(0)
    print(FourierRadiusInt)

#%%
def SliderGetTiltValue(event=None):
    global TiltInDeg
    TiltInDeg= int( sliderTilt.get() )
    #print(TiltNow)
    

#%%

def SliderGetSigma4EofK(event=None):
    global Sigma4EofK
    Sigma4EofK= sliderSigma4EofK.get() 
    #print('Sigma4EofK, from slider get = ' + str(Sigma4EofK))
    #stdlogEofKSpiralC = np.std(logEofKSpiralC)
    #factor =Sigma4EofK;
    #UpperCutoff = meanlogEofKSpiralC+ factor*stdlogEofKSpiralC
    #PopoutPlotsEofkvsSampling(InnerProdBooleanSum1,logEofKSpiralC,meanlogEofKSpiralC,UpperCutoff)

    
#%%   What happens when NumberToUse Entry box is filled

def CalcPlots(event=None):
    
    #print(EnterNumberToUse.get())
    global GraphedAlready
    
    print(SymNow)
    CreateSpiral()
    get_InnerProdBooleanSum1(0)
    PlotProj()
    PlotSamp()
    PlotHistSamp()
    GraphedAlready=1
    print(matplotlib.get_backend())

#%%  
def PlotProjTiltSection(events=None):

    global figureProjB
    CosTheta = np.asarray(TiltedNVecs[:,2])
    CosThetaB = CosTheta.copy()
    IndsZZneg = np.where(CosTheta<0)[0]


    XX = np.asarray(TiltedNVecs[:,0])
    YY = np.asarray(TiltedNVecs[:,1])
    
    XXB= XX.copy()
    YYB= YY.copy()
    
    XXB[IndsZZneg] *=-1
    YYB[IndsZZneg] *=-1
    CosThetaB[IndsZZneg] *=-1
    
    CosTheta2 = CosThetaB*CosThetaB
    SinThetaB = np.sqrt(1-CosTheta2)
    ThetaB = np.arcsin(SinThetaB)
    FuncThetaB = ThetaB/(np.pi/2)
    FuncThetaB = np.sin(ThetaB/2)/(np.sin(np.pi/4))
    
    XXFinal=XXB.copy()
    YYFinal=YYB.copy()

    GoodInds= np.where(np.abs(SinThetaB)>.0001)[0]
    XXFinal[GoodInds] = XXB[GoodInds]*FuncThetaB[GoodInds]/SinThetaB[GoodInds]
    YYFinal[GoodInds] = YYB[GoodInds]*FuncThetaB[GoodInds]/SinThetaB[GoodInds]

    
    winHeight = win.winfo_height();
    fgs = winHeight/400;
    print('fgs =%f'%fgs)
    #figureProjB = plt.Figure( figsize=(fgs,fgs), dpi=50 )
    figureProjB = plt.Figure( figsize=(3.5,3.5), dpi=FigDpi )
    if 0:
        axProjB = figureProjB.add_subplot(111,projection='3d')


    
    axProjB = figureProjB.add_subplot(111)
    #bar1 = FigureCanvasTkAgg(figure1, win)
    #canvasNow = FigureCanvasTkAgg(figureProj, lblProjPlot)
    canvasNowB = FigureCanvasTkAgg(figureProjB, win)
   
    canvasNowB.get_tk_widget(). place(relx=0.7, rely=0.433);
    #df1.plot(kind='bar', legend=True, ax=ax1)
    #axProj.plot(np.arange(10))
    if 0:
        axProjScatterB = axProjB.scatter(AllProjAndTilts[:,0], AllProjAndTilts[:,1], \
            AllProjAndTilts[:,2],  c='b', marker='o')
        axProjScatterB = axProjB.scatter(TiltedNVecs[:,0], TiltedNVecs[:,1], \
            TiltedNVecs[:,2], s=1, c='b', marker='.')
        
    axProjScatterB = axProjB.scatter(XXFinal, YYFinal, \
             s=7, c='b', marker='.')
         
    axProjB.set_title('Tilted Proj Directions')
    #figureProj.set_xlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    axProjB.set_xlim(-1.1, 1.1)                    # viewrange for z-axis should be [-4,4] 
    axProjB.set_ylim(-1.1, 1.1)                    # viewrange for z-axis should be [-4,4] 
    if 0:
        axProjB.set_zlim(-1.1, 1.1)                    # viewrange for z-axis should be [-4,4] 
    canvasNowB.draw()
    print('Graphed PlotProjTiltSection')
  
    
#%%
def PlotSampTiltSection():
    print('Hi from PlotSampTiltSection')
#   # Create  plot of sampling
    global SCF0, SCFStar, NOver2k
    global GoodVoxels,figureSampTilt
    
    print('Started  PlotSamp')
    figureSampTilt = plt.Figure(figsize=(4.5,3), dpi=FigDpi)
    axSampTilt = figureSampTilt.add_subplot(111,projection='3d')
    #bar1 = FigureCanvasTkAgg(figure1, win)
    bar2 = FigureCanvasTkAgg(figureSampTilt, win)
    bar2.get_tk_widget().place(relx=0.3, rely=0.433)
    #df1.plot(kind='bar', legend=True, ax=ax1)
    #axProj.plot(np.arange(10))

   
    colors = [(1, 0, 0), (1,1,0), (0, 0, 1)]  # R -> Y -> B
    n_bin = 30   # Discretizes the interpolation into bins
    cmap_name = 'my_list'
    # Create the colormap
    cmPRB = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    SpiralData = np.asarray(InnerProdBooleanSum1).flatten()
    DataOnSpiral,pnt1,pnt2,pnt3 = MapSamplingToColor(SpiralData,FourierRadiusInt)
  
    precision=2;
    
    pnt0str = "{:.{}f}".format( 0, precision )
    pnt1str = "{:.{}f}".format( pnt1, precision )
    pnt2str = "{:.{}f}".format( pnt2, precision )
    pnt3str = "{:.{}f}".format( pnt3, precision )
    
    
    IndMin = np.argmin(DataOnSpiral); DataOnSpiral[IndMin]=0
    #IndMax = np.argmax(DataOnSpiral); DataOnSpiral[IndMax]=0
    
    #axSamp.scatter(SpiralVec[:,0], SpiralVec[:,1], SpiralVec[:,2], \
    axSampTiltScatter = axSampTilt.scatter(SpiralVec[:,0], SpiralVec[:,1], \
                  SpiralVec[:,2], \
                   c=np.asarray(DataOnSpiral).flatten(), marker='o',cmap=cmPRB)
     
    axSampTilt.set_title('Sampling Map after Tilt; FR='+str(FourierRadiusInt))

    #plt.show(axSampScatter)

    cbar = figureSampTilt.colorbar(axSampTiltScatter,ticks=[0,1/3,2/3,1],pad=0.2)
    cbar.ax.set_yticklabels([pnt0str,pnt1str, pnt2str, pnt3str])  # vertically oriented colorbar

    GoodVoxels= np.where(InnerProdBooleanSum1>0)[0];
    
    RecipSamp = 1./InnerProdBooleanSum1[GoodVoxels]
    NOver2k = np.mean(InnerProdBooleanSum1)
    NumberOfZeros = NumFib -len(GoodVoxels);
    FractionOfZeros = NumberOfZeros/NumFib
    FractionOfNonZeros = 1- NumberOfZeros/NumFib
    QkoverPk =FractionOfZeros/FractionOfNonZeros

    SCF0Tilt=  1/(np.mean(RecipSamp)*NOver2k )
    SCF0Tilt_value.set('SCFTilt= %.3f' % SCF0Tilt)   
    
    FracOfZerosTilt_value.set('FractionOfZerosTilt= %.2f  percent' % (100*FractionOfZeros))   

    SCFStarTilt=  SCF0Tilt/(1+ QkoverPk*NOver2k*SCF0Tilt )
    SCFStarTilt_value.set('SCFTilt*= %.3f' % SCFStarTilt)   
    
    print('Number Of Zeros = ' + str(NumberOfZeros))
    print('FractionOfZeros= %.3f' % FractionOfZeros)
    print('SCF0Tilt= %.3f' % SCF0Tilt)
    print('SCF*Tilt= %.3f' % SCFStarTilt)
    print('Finished  PlotSampTilt')

    #figureProj.set_xlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    #figureProj.set_ylim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    #figureProj.set_zlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    
    
#%%
def PlotHistSampTiltSection():
    print('Hi from PlotHistSampTiltSection')
   
    global axHistTiltSamp,figureHistTiltSamp
#   # Create  plot of sampling
    winHeight = win.winfo_height();
    fgs = winHeight/1600;
    figureHistTiltSamp = plt.Figure(figsize=(3.5,2.5), dpi=FigDpi)
    axHistTiltSamp = figureHistTiltSamp.add_subplot()
    #bar1 = FigureCanvasTkAgg(figure1, win)
    bar1 = FigureCanvasTkAgg(figureHistTiltSamp, win)
    bar1.get_tk_widget().place(relx=0.32, rely=0.616)
    #df1.plot(kind='bar', legend=True, ax=ax1)
    #axProj.plot(np.arange(10))
    if 1:
        axHistTiltSamp.hist(InnerProdBooleanSum1,100,log=True)
        axHistTiltSamp.set_title('log hist of Sampling Map vals; tilt='+str(TiltInDeg))
        axHistTiltSamp.set_xlim(0, 1.1*np.max(InnerProdBooleanSum1))   # viewrange for z-axis should be [-4,4]
    if 0:
        plt.hist(InnerProdBooleanSum1,100,log=True)
        plt.title('log hist of Sampling Map vals; FR='+str(FourierRadiusInt))
        plt.xlim(0, 1.1*np.max(InnerProdBooleanSum1))   # viewrange for z-axis should be [-4,4]
    #figureProj.set_ylim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 
    #figureProj.set_zlim3d(-1, 1)                    # viewrange for z-axis should be [-4,4] 

    
    
#%%   What happens when NumberToUse Entry box is filled

def CalcWithTilts(event=None):
    
    #print(EnterNumberToUse.get())
    global GraphedAlreadyTilts
    
    #print(SymNow)
    get_InnerProdBooleanSum1(TiltInDeg)
    PlotProjTiltSection()
    PlotSampTiltSection()
    PlotHistSampTiltSection()

#%%
def save_ProjPlot(): 
    files = [('Jpg Files', '*.jpg'),('Png Files', '*.png')]             
    fileNow = tk.filedialog.asksaveasfile(filetypes = files, defaultextension = files) 
    figureProj.savefig(fileNow.name,dpi=SaveFigDpi)

#%%    
def save_SampPlot(): 
    files = [('Jpg Files', '*.jpg'),('Png Files', '*.png')]             
    fileNow = tk.filedialog.asksaveasfile(filetypes = files, defaultextension = files) 
    figureSamp.savefig(fileNow.name,dpi=SaveFigDpi)
   
#%%
def save_HistPlot(): 
    files = [('Jpg Files', '*.jpg'),('Png Files', '*.png')]             
    fileNow = tk.filedialog.asksaveasfile(filetypes = files, defaultextension = files) 
    figureHistSamp.savefig(fileNow.name,dpi=SaveFigDpi)
 
#%%    
def save_ZeroPlot(): 
    files = [('Jpg Files', '*.jpg'),('Png Files', '*.png')]             
    fileNow = tk.filedialog.asksaveasfile(filetypes = files, defaultextension = files) 
    figureFracZeros.savefig(fileNow.name,dpi=SaveFigDpi)

#%%
def save_ProjTiltPlot(): 
    files = [('Jpg Files', '*.jpg'),('Png Files', '*.png')]             
    fileNow = tk.filedialog.asksaveasfile(filetypes = files, defaultextension = files) 
    figureProjB.savefig(fileNow.name,dpi=SaveFigDpi)


#%%
def save_SampTiltPlot(): 
    files = [('Jpg Files', '*.jpg'),('Png Files', '*.png')] 
    fileNow = tk.filedialog.asksaveasfile(filetypes = files, defaultextension = files) 
    figureSampTilt.savefig(fileNow.name,dpi=SaveFigDpi)


#%%
def save_HistTiltPlot(): 
    files = [('Jpg Files', '*.jpg'),('Png Files', '*.png')]             
    fileNow = tk.filedialog.asksaveasfile(filetypes = files, defaultextension = files) 
    figureHistTiltSamp.savefig(fileNow.name,dpi=SaveFigDpi)

#%%
def PlotEofkvsSampling():
    from matplotlib.figure import Figure
    global sliderSigma4EofK,FractionOfZerosArrayAgroNow
    global SuspiciousInds, QkOverPkAgro
    global SCFStarAgro,SCFStarAgro_value
    
    NOver2k = np.mean(InnerProdBooleanSum1)

    meanlogEofKSpiralC = np.mean(logEofKSpiralC)
    stdlogEofKSpiralC = np.std(logEofKSpiralC)
    factor =Sigma4EofK;
    UpperCutoff = meanlogEofKSpiralC+ factor*stdlogEofKSpiralC

    SuspiciousInds = np.where(logEofKSpiralC > UpperCutoff)[0]
        
    FractionOfZerosArrayAgroNow = len(SuspiciousInds)/len(InnerProdBooleanSum1)
    FractionOfNonZerosAgroNow = 1- FractionOfZerosArrayAgroNow
    
    QkOverPkAgro=FractionOfZerosArrayAgroNow/FractionOfNonZerosAgroNow
    SCFStarAgro = SCF0/(1+QkOverPkAgro*NOver2k * SCF0/2)
    
    SCFStarAgro_value.set('SCF*Agro = %.3f'% SCFStarAgro)
    print('SCF*Agro = %.3f'% SCFStarAgro)
    
    print(FractionOfZerosArrayAgroNow)
    
    print('Sigma4EofK from line 170, plot = ' + str(Sigma4EofK))
    
    figureEofK = Figure(figsize=(2.8,2.2),dpi=150)
    axEofk = figureEofK.add_subplot(111)
    axEofk.set_title('logEofKSpiralC vs InnerProdBooleanSum1',fontsize=8)
    axEofk.scatter(InnerProdBooleanSum1  ,logEofKSpiralC,s=1)
    axEofk.tick_params(axis='both', which='major', labelsize=5)
    axEofk.set_xlabel( ' Sampling (InnerProdBooleanSum1) ',fontsize=5)
    axEofk.set_ylabel( ' log(EofK) ',fontsize=5)
    
    
    #plot = ax.plot(x_vals, y_vals, 'k-')
    #ax.set_xlabel('Time')
    #ax.set_ylabel('Numbers')
    x1= np.min(InnerProdBooleanSum1)
    x2 = np.max(InnerProdBooleanSum1)
    y1= meanlogEofKSpiralC; y2=meanlogEofKSpiralC
    axEofk.plot([x1, x2], [y1, y2], color='k', linestyle='-', linewidth=2)
    y1= UpperCutoff; y2=UpperCutoff
    axEofk.plot([x1, x2], [y1, y2], color='r', linestyle='-', linewidth=2)
    bar1 = FigureCanvasTkAgg(figureEofK, toplevelEofk)
    bar1.get_tk_widget().place(relx=0.1, rely=0.1066)


#%%
#def PopoutPlotsEofkvsSampling(InnerProdBooleanSum1,logEofKSpiralC):
def PopoutPlotsEofkvsSampling():
    from matplotlib.figure import Figure
    global sliderSigma4EofK
    global toplevelEofk
    global Sigma4EofK
    global InnerProdBooleanSum1,logEofKSpiralC
    
    toplevelEofk = tk.Toplevel()
    toplevelEofk.geometry("600x600")
    toplevelEofk.title("Eofk Stat Tool")


    #PlotEofkvsSampling(InnerProdBooleanSum1,logEofKSpiralC)
    PlotEofkvsSampling()

    #canvas = FigureCanvasTkAgg(figure, master=toplevel)
    #canvas.draw()
    
    sliderSigma4EofK      = tk.Scale(master=toplevelEofk, from_=0, to=4,length =150, \
                               resolution=0.01, orient='h',  \
                               command=SliderGetSigma4EofK)
    sliderSigma4EofK.set(Sigma4EofK)
    sliderSigma4EofK.place(relx=0.4,rely=.7)


    lblSigma = tk.Label(master=toplevelEofk,text="SigmaForEofk")
#    lblSigma.place(relx=0.47,rely=.91+TiltHMShift)
    lblSigma.place(relx=0.432,rely=.77)

    btnReplot = tk.Button(toplevelEofk,text='Replot After Changing Sigma',\
        command=PlotEofkvsSampling)
        
    btnReplot.place(relx=0.14,rely=0.83)



    lblSigmaInfo = tk.Label(master=toplevelEofk,text= \
            "Choosing  sigma identifies suspicious voxels for correcting FSCs")
#    lblSigma.place(relx=0.47,rely=.91+TiltHMShift)
    lblSigmaInfo.place(relx=0.132,rely=.92)

    
    btnDestroy = tk.Button(toplevelEofk,text='Close Eofk tool',command=toplevelEofk.destroy)
    btnDestroy.place(relx=0.8,rely=0.9)

    
    print('Sigma4EofK from line 229, toplevel = ' + str(Sigma4EofK))
 
#%%     Start     Tkinter  GUI
"""
#  Section 0. Initialize

win = tk.Tk()
win.title("Sampling and Tilting Tool")#,bg='#add8e6')
win.geometry("800x1300") # should be 800x1300

global TiltInDeg, Sigma4EofK   #  From Sliders
global NumberForEachTilt
global FourierRadiusStr,FourierRadiusInt
global SpiralVec, NumFib
global varSym
global NumberReadIn, NumberToUse 
global NumberToUseInt
global SCF0_value, SCFStar_value, FractionOfZeros_value
global SCF0Tilt_value, SCFStarTilt_value, FracOfZerosTilt_value
global GraphedAlready, GraphedAlreadyFSCNaive, GraphedAlreadyFSCaggro
global Sigma4EofK;

#global sliderTilt
#global GraphedAlready;# changes from 0 to 1 after plotting

GraphedAlready=0
GraphedAlreadyFSCNaive =0
GraphedAlreadyFSCaggro = 0 
TiltInDeg         =  0
NumberForEachTilt = 90
FourierRadiusInt  = 23
FourierRadiusInt  = 50
FigDpi=50
SaveFigDpi=150
Sigma4EofK =1


NumberReadIn    = tk.StringVar()
NumberToUse     = tk.StringVar()
SCF0_value      = tk.StringVar()
SCFStar_value   = tk.StringVar()
SCFStarAgro_value   = tk.StringVar()
FracOfZeros_value   = tk.StringVar()
folder_path  = tk.StringVar()
varSym       = tk.StringVar()
varCSym       = tk.StringVar()
FourierRadiusStr = tk.StringVar()
FourierRadiusStr.set(str(FourierRadiusInt))

SCF0Tilt_value      = tk.StringVar()
SCFStarTilt_value   = tk.StringVar()
FracOfZerosTilt_value   = tk.StringVar()
ThreeDFSC_path       = tk.StringVar()
FSC_Agro_path     = tk.StringVar()
FSC2CorrectwZeros_path  = tk.StringVar()
#FourierRadius   = tk.StringVar()

ChoicesSym = ['C','D','Tet','Oct','Icos' ]
varSym.set('C') # set the default option
SymNow = 'C'
ChoicesCSym = [ '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16' ,'17']
varCSym.set('1') # set the default option
CSymNow = 1
#  Section 1. Define Widgets

lblTitle = tk.Label(master=win,text='Sampling and Tilting Tool',bg='#add8e6', \
                borderwidth=5    ,highlightbackground='black')
lblFileNameRead     = tk.Label(master=win,text="File Read In is "); # Above the name of the file 
lblAngFileName      = tk.Label(master=win,textvariable=folder_path); # Contains the name of the file 
lblNumberReadIn     = tk.Label(master=win,textvariable=NumberReadIn);  # Contains a label
lblNumberToUse      = tk.Label(master=win,text="NumberToUse="); # Contains the name of the file 
EntryNumberToUse    = tk.Entry(master=win,textvariable=NumberToUse)
EntryNumberToUse.bind('<Return>', Get_Num2Use_Entry )
#print(EntryNumberToUse.get())

#lblNumberToUseB      = tk.Label(master=win,text="NumberToUse="+str(NumberToUseInt)); # Contains the name of the file 
#lblFourierRadius     = tk.Label(master=win,text="FourierRadius=23 to begin"); # Contains the name of the file 
lblFourierRadiusA     = tk.Label(master=win,text="FourierRadius = "+str(FourierRadiusInt)); # Contains the name of the file 
#lblFourierRadiusB     = tk.Label(master=win,textvariable=FourierRadiusStr); # Contains the name of the file 
EntryFourierRadius   = tk.Entry(master=win,textvariable=FourierRadiusStr)
EntryFourierRadius.bind('<Return>', Get_FourierRadius_Entry )
#print(EntryFourierRadius.get())


buttonSelectAngFile = tk.Button(text="Select 3 col angle file", \
                                command=browse_button)
buttonCalc        = tk.Button(master=win, text='Calc Plots', command=CalcPlots)
 
button_save_ProjPlot = tk.Button(master=win, text = 'Save', command =  save_ProjPlot) 
button_save_SampPlot = tk.Button(master=win, text = 'Save', command =  save_SampPlot) 
button_save_HistPlot = tk.Button(master=win, text = 'Save', command = save_HistPlot)
button_save_ZeroPlot = tk.Button(master=win, text = 'Save', command = save_ZeroPlot) 

 
button_save_ProjTiltPlot = tk.Button(master=win, text = 'Save', command =  save_ProjTiltPlot) 
button_save_SampTiltPlot = tk.Button(master=win, text = 'Save', command =  save_SampTiltPlot) 
button_save_HistTiltPlot = tk.Button(master=win, text = 'Save', command = save_HistTiltPlot)

lblTiltSect = tk.Label(master=win,text='Tilt Section',bg='lightblue', \
                      justify='left',width=60 );  # Contains the samp plot
#EnterNumberToUse.focus_set()

if 0:
    heightBar=0
    canvasNow = tk.Canvas(master=win)
    canvasNow.create_rectangle(0, heightBar,800, heightBar+10,outline="#fb0", fill="#fb0")
    canvasNow.place(relx=0.02,rely=.433)

#canvasNow.pack(expand = True, fill = "both")
#canvasNow.pack(expand = True)


#lblFracZerosPlot    = tk.Label(master=win);

#lblProjPlot         = tk.Label(master=win);  # Contains the proj plot
#lblSampPlot         = tk.Label(master=win);  # Contains the samp plot
#lblHistSampPlot     = tk.Label(master=win);  # Contains the samp plot
lblSCFuntilted      = tk.Label(master=win,textvariable=SCF0_value);  # Contains the samp plot
lblSCFStarUntlt     = tk.Label(master=win,textvariable=SCFStar_value);  # Contains the samp plot
lblFracZeros        = tk.Label(master=win,textvariable=FracOfZeros_value);  # Contains the samp plot

lblSCFTilted        = tk.Label(master=win,textvariable=SCF0Tilt_value);  # Contains the samp plot
lblSCFStarTilted    = tk.Label(master=win,textvariable=SCFStarTilt_value);  # Contains the samp plot
lblFracZerosTilted  = tk.Label(master=win,textvariable=FracOfZerosTilt_value);  # Contains the samp plot

sliderTilt          = tk.Scale(master=win, from_=0, to=90,length =200, \
                               orient='h', command=SliderGetTiltValue)
print(sliderTilt.get())
buttonCalc0s = tk.Button(text="Investigate Zeros", \
                                command=PlotFracZeros)
    
buttonFixFSC       = tk.Button(text="Correct this FSC",command=CorrectFSC)
lblFixFSC_FN       = tk.Label(master=win,textvariable=FSC2CorrectwZeros_path)

buttonCalcWithTilts = tk.Button(text="Recalc Sampling", \
                                command=CalcWithTilts)
    

lblTiltsInDeg = tk.Label(text="Tilt In Degrees")



#lblSym = tk.Label(master=win, text="Symmetry = "+varSym.get())
lblSym = tk.Label(master=win, text="Symmetry = ")
lblCSym = tk.Label(master=win, text="C or D Sym # = ")


# Dictionary with options
#popupMenuSym = tk.OptionMenu(win,varSym,ChoicesSym[0],*ChoicesSym)
popupMenuSym = tk.OptionMenu(win,varSym,*ChoicesSym)
varSym.trace('w', change_SymDropdown)

popupMenuCSym = tk.OptionMenu(win,varCSym,*ChoicesCSym)
varCSym.trace('w', change_CSymDropdown)


#    Placements

lblTitle.place(relx=0., rely=0.)
buttonSelectAngFile.place(relx=0.02,rely=0.033); # This is the browse button
lblFileNameRead.place(relx=0.02, rely=0.08)
lblAngFileName.place(relx=0.02, rely=0.1); # This is where I will type out the ~/Dropbox/Salk17/GridDataOntoSpherename of the file
lblNumberReadIn.place(relx=0.02, rely=0.1333)
lblNumberToUse.place(relx=0.02, rely=0.1533)
EntryNumberToUse.place(relx=0.02,rely=0.1733)
#lblNumberToUseB.place(relx=0.02,rely=0.355)
lblSym.place(relx=0.02,rely=0.213)
lblCSym.place(relx=0.12,rely=0.213)

popupMenuSym.place(relx=0.03,rely=0.24)
popupMenuCSym.place(relx=0.13,rely=0.24)

lblFourierRadiusA.place(relx=0.02,rely=.273)
#lblFourierRadiusB.place(relx=0.14,rely=.41)
EntryFourierRadius.place(relx=0.02,rely=.293)
buttonCalc.place(relx=0.02,rely=.32)
TiltPlShift =0.03
TiltHMShift = TiltPlShift -.3

lblTiltSect.place(relx=0.02, rely=.36+TiltPlShift)
sliderTilt.place(relx=0.02,rely=.38+TiltPlShift)
lblTiltsInDeg.place(relx=0.1,rely=0.42+TiltPlShift)


lblSCFuntilted.place(relx=0.35,rely=.30+TiltPlShift)
lblSCFStarUntlt.place(relx=0.35,rely=.313+TiltPlShift)
lblFracZeros.place(relx=0.35,rely=.327+TiltPlShift)  # Contains the samp plot

if 1:
    lblSCFTilted.place(relx=0.35,rely=.75+TiltPlShift)          # Contains the samp plot
    lblSCFStarTilted.place(relx=0.35,rely=.763+TiltPlShift)    # Contains the samp plot
    lblFracZerosTilted.place(relx=0.35,rely=.776+TiltPlShift)  # Contains the samp plot


#lblFracZerosPlot.place(relx=0.75,rely=.28)  ;
buttonCalc0s.place( relx=0.8,rely=.337)
buttonFixFSC.place( relx=0.8,rely=.36 )
lblFixFSC_FN.place( relx=0.75,rely=.38 )


button_save_ProjPlot.place(relx=0.92,rely=0.1333)
button_save_SampPlot.place(relx=0.52,rely=0.1333)
button_save_HistPlot.place(relx=0.52,rely=0.305)
button_save_ZeroPlot.place(relx=0.92,rely=0.305)


if 1:
    buttonCalcWithTilts.place(relx=0.02,rely=0.5333+TiltPlShift)
    button_save_ProjTiltPlot.place(relx=0.92,rely=0.5733+TiltPlShift)
    button_save_SampTiltPlot.place(relx=0.52,rely=0.5633+TiltPlShift)
    button_save_HistTiltPlot.place(relx=0.52,rely=0.725+TiltPlShift)

#button_Correct3DFSC    = tk.Button(text="Correct FSC curve", \
#                                   command=CorrectFSCcurve)
    

#button_Correct3DFSC.place(relx=0.75,rely=0.37)

#button_CorrectFSC = tk.Button(text="Correct FSC", \
#                                    command=CorrectFSC_Curve)

#lblProjPlot.place(relx=0.7, rely=0.01);# This is where the figure goes
#lblHistSampPlot.place(relx=0.3, rely=0.28);# This is where the figure goes
#lblSampPlot.place(relx=0.3, rely=0.01);# This is where the figure goes

# Choose a symmetry


#%%   This is the FSC section

if 0:
    lblFSCSect = tk.Label(master=win,text='FSC* Section',bg='lightblue', \
                          justify='left',width=60 );  # Contains the samp plot
    
    
    
    buttonSelectMapMapFSC = tk.Button(text="Select MapMapFSC", \
                                    command=browse_buttonFSC_MapMap)
                          
    lblFSCSect.place(relx=0.02, rely=.85+TiltPlShift)
    lblMapMapFSCNameRead     = tk.Label(master=win,text="FSC File Read In is "); 
    lblMapMapFSCNameRead.place(relx=0.02, rely=.9+TiltPlShift)
    
    
    buttonSelectMapMapFSC.place(relx=0.02,rely=0.923+TiltPlShift)
    
if 1:
    lblHalfMapSect = tk.Label(master=win,text='Half Map Section',bg='lightblue', \
                          justify='left',width=60 );  # Contains the samp plot
    
    
    buttonSelectHalfMap1 = tk.Button(text="Select HalfMap1", \
                                    command=browse_buttonHalfMap1)
    #buttonSelectHalfMap2 = tk.Button(text="Select HalfMap2", \
    #                                command=browse_buttonHalfMap1)
        
    lblHalfMapSect.place(relx=0.02, rely=.82 +TiltHMShift)
    
    lblHalfMap1NameRead   = tk.Label(master=win,text="HalfMap1  is "); # unplaced
    lbl3DFSCRead          = tk.Label(master=win,text="3DFSC is ");  
    lbl3DFSCFN            = tk.Label(master=win,textvariable=ThreeDFSC_path); # Contains the name of the 3DFSC file 
    buttonSelect3DFSC    = tk.Button(text="Select 3DFSC.mrc", \
                                    command=browse_button3DFSC)

    lbl_FSC_Agro_FN       = tk.Label(master=win,textvariable=FSC_Agro_path); # Contains the name of the FSC file to be corrected aggressively
    buttonFixFSCAggressive = tk.Button(text="Correct  FSC aggressively",command=CorrectFSCAggressive)
                          
    
    buttonAnalyze3DFSC = tk.Button(text="Analyze 3DFSC.mrc", \
                                    command=Analyze3DFSC)
    #lblHalfMap2NameRead  = tk.Label(master=win,text="HalfMap2 is "); 
    
    #lblHalfMap1NameRead.place(relx=0.02, rely=.9)
    #lblHalfMap2NameRead.place(relx=0.4, rely=.9)
    lbl3DFSCRead.place(relx=0.02, rely=.87+TiltHMShift)
    lbl3DFSCFN.place(relx=0.1, rely=.87+TiltHMShift)
    buttonSelect3DFSC.place(relx=0.02,rely=0.91+TiltHMShift)
    buttonAnalyze3DFSC.place(relx=0.02,rely=0.935+TiltHMShift)


    lblSCFStarAgro   = tk.Label(master=win,textvariable=SCFStarAgro_value);  # 
    lblSCFStarAgro.place(relx=0.65,rely=0.95+TiltHMShift)



    lbl_FSC_Agro_FN.place(relx=0.7, rely=.88+TiltHMShift)
    buttonFixFSCAggressive.place(relx=0.8,rely=.9+TiltHMShift)
    
    #buttonSelectHalfMap1.place(relx=0.02,rely=0.923)
    #buttonSelectHalfMap2.place(relx=0.4,rely=0.923)


#%%  USe FSCdata and FractionOfZerosArray for correction

#os.chdir("/home/pbaldwin/Dropbox/GroupArticles/Salk/JPBMTiltwDmitry/Data/ribosome, April 2020/1st_preferred_orientation")
#os.chdir("/home/pbaldwin/Dropbox/GroupArticles/Salk/JPBMTiltwDmitry/Data/ribosome, April 2020/all_orientations")
#os.chdir('ResultsApr2020')



    
win.mainloop()
"""
