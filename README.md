# CommandLineSCF
Download this code underneath anaconda, and then start using this code to evaluate SCF (Baldwin, Lyumkis) on files of triples of angles (PSI THETA ROT).  



> python SCFJan2022.py Nov2021_10K_ConeAngle_30PCent10.par  --help


usage: SCFJan2022.py [-h] [--FourierRadius FOURIERRADIUS] [--NumberToUse NUMBERTOUSE] [--3DFSCMap 3DFSCMAP] [--RootOutputName ROOTOUTPUTNAME] [--TiltAngle TILTANGLE] [--Sym SYM] FileName  <br />

Calculate SCF parameter and make plots. Name of plot is based on input Angle File. The SCF is how much the SSNR has likely been attenuated due to projections not being distributed uniformly. The SCF, SCF* parameters are described here: Baldwin, P.R. and D. Lyumkis, Non-uniformity of projection distributions attenuates resolution in Cryo-EM.Prog Biophys Mol Biol, 2020. 150: p. 160-183. <br />

positional arguments:<br />
 FileName              the name of the File of Angles; Psi Theta Rot in degrees <br />

optional arguments: <br />
 -h, --help            show this help message and exit <br />
 --FourierRadius FOURIERRADIUS <br />
   &nbsp &nbsp Fourier radius (int) of the shell on which sampling is evaluated <br />
 --NumberToUse NUMBERTOUSE <br />
      &nbsp &nbsp          the number of projections to use, if you don't want to use all of them <br />
 --3DFSCMap 3DFSCMAP   the 3DFSC map, if one wants to correlate Sampling/Resolution; currently not implemented <br />
 --RootOutputName ROOTOUTPUTNAME <br />
                the root name for logging outputs. Default is SCFAnalysis <br />
 --TiltAngle TILTANGLE <br />
                       tilt angle <br />
 --Sym SYM             symmetry: Icos, Oct, Tet, Cn, or Dn. If tilt specified, then Sym =C1   <br />

----------------------------------------------------------------------------------

