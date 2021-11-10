# Microsecond_Polarimetry_R3
A collection of scripts used for the analysis and plotting in Nimmo et al. 2021a "Highly polarised microstructure from the repeating FRB 20180916B"

The data comes from SFXC (https://github.com/jive-vlbi/sfxc) in the form of filterbank files for each polarisation product (rr, ll, rl_real, rl_imaginary).
We combine these into a single filterbank file and using psrchive dspsr to create an archive file. 
The python scripts take these archive files to perform the necessary analysis and make the plots.  

The data required to run these scripts can be found here: https://zenodo.org/record/4350456#.YYuQxb3MJqs

The paper can be found here: https://ui.adsabs.harvard.edu/abs/2020arXiv201005800N/abstract


joint_fit_QU.py -- used to fit functions of Q and U spectra to the pulsar data and the burst data, to extract the rotation measure and instrumental delay

figure1.py -- plot Figure 1 in Nimmo et al. 2021a

figure2.py -- plot Figure 2 in Nimmo et al. 2021a

figure3and4.py -- plot Figures 3 and 4 in Nimmo et al. 2021a

ed-figure1.py -- plot Extended Data Figure 1 in Nimmo et al. 2021a

ed-figure2.py -- plot Extended Data Figure 2 in Nimmo et al. 2021a
