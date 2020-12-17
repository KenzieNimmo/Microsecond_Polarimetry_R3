"""To make Fig1 in Nimmo et al. 2020"""

from load_archive import load_archive
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from radiometer import radiometer
import matplotlib as mpl

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'                                                                                                                           
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'



def profile(ds):
    """
    Given a dynamic spectrum ds (with polarisations) output a Stokes I profile
    """
    print(ds.shape)
    LL = ds[0,:,:]
    RR = ds[1,:,:]
    Ids=(LL+RR)/2.
    prof=np.sum(Ids,axis=0)
    return prof

def line_up_profiles(data_16us,data_1us,b3=False):
    """
    Given the 16us and 1us data, line up the profiles by downsampling the 1us data
    """
    #define the maximum in the 16us data
    max_16=np.argmax(data_16us)
    #b3 data needs to be rolled to put the peak in the centre
    if b3==True:
        peak_1us = np.argmax(data_1us)
        newpeak=int(len(data_1us)/2.)
        data_1us=np.roll(data_1us,newpeak-peak_1us)
    #downsample the 1us data to 16us resolution
    prof16=np.add.reduceat(data_1us, np.arange(0, len(data_1us), 16))
    #define the maximum of the downsampled 16us data
    prof16max=np.argmax(prof16)
    #length of the downsampled array
    lenprof16=len(prof16)
    #crop the 16us data to have the same axis as the 1us downsampled data (and in essence the 1us data)
    new_data_16us=data_16us[max_16-prof16max:max_16+(lenprof16-prof16max)]

    if b3==True:
        return new_data_16us, data_1us
    else:
        return new_data_16us

def centre_burst(data_16us,data_1us,tcent,orig_resolution):
    """
    shifting the arrays to match polarisation plot and Marcote+2020
    tcent is offset of centre from peak bin in number of 16us bins
    give orig_resolution in us and tcent in bins
    """

    if orig_resolution!=16:
        prof_orig=np.add.reduceat(data_16us, np.arange(0, len(data_16us), int(orig_resolution/16.)))
    else:
        prof_orig = data_16us


    peakbin = np.argmax(prof_orig)
    peakbin*=orig_resolution/16.
    peakbin+=tcent
    peakbin=int(peakbin)
    newcent= int(len(data_16us)/2.)
    if peakbin > newcent:
        data_16us = np.roll(data_16us,(len(data_16us)-peakbin)+newcent)
        data_1us = np.roll(data_1us,((len(data_16us)-peakbin)+newcent)*16)
    if peakbin < newcent:
        data_16us = np.roll(data_16us,newcent-peakbin)
        data_1us = np.roll(data_1us,(newcent-peakbin)*16)

    return data_16us,data_1us

def convert_to_flux(data,datanoise,tsamp,bw,npol,SEFD):
    """
    First convert to S/N.
    Then use the radiometer equation to convert to physical units [Jy]
    """
    data-=np.mean(datanoise)
    datanoise-=np.mean(datanoise)
    data/=np.std(datanoise)

    data*=radiometer(tsamp, bw, npol, SEFD)

    return data

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.
    output:
        the smoothed signal
    example:
    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    see also:
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

ds_b1_1us = load_archive('./1us/ep114a-fullcoherence_no0204_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012)
ds_b2_1us = load_archive('./1us/ep114a-fullcoherence_no0210_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012)
ds_b3_1us = load_archive('./1us/ep114a-fullcoherence_no0224_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012)
ds_b4_1us = load_archive('./1us/ep114a-fullcoherence_no0278_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012)

prof_b1_1us= profile(ds_b1_1us)
prof_b2_1us= profile(ds_b2_1us)
prof_b3_1us= profile(ds_b3_1us)
prof_b4_1us= profile(ds_b4_1us)

#load in 16us profiles of the four bursts corrected to DM +0.012

ds_b1_16us=load_archive('./16us/burst1.fp.fc',rm=-536.18,dm=0.012)
ds_b2_16us=load_archive('./16us/burst2.fp.fc',rm=-536.18,dm=0.012)
ds_b3_16us=load_archive('./16us/burst3.fp.fc',rm=-536.18,dm=0.012)
ds_b4_16us=load_archive('./16us/burst4.fp.fc',rm=-536.18,dm=0.012)

prof_b1_16us= profile(ds_b1_16us)
prof_b2_16us= profile(ds_b2_16us)
prof_b3_16us= profile(ds_b3_16us)
prof_b4_16us= profile(ds_b4_16us)

new_prof_b1_16us = line_up_profiles(prof_b1_16us,prof_b1_1us)
new_prof_b2_16us = line_up_profiles(prof_b2_16us,prof_b2_1us)
new_prof_b3_16us, prof_b3_1us = line_up_profiles(prof_b3_16us,prof_b3_1us,b3=True)
new_prof_b4_16us = line_up_profiles(prof_b4_16us,prof_b4_1us)


#centre the bursts to match polarisation figure and Marcote+2020
#B1
oldpeakb1=7985
oldres1 = 16e-6*5
newres1 = 16e-6*1
tcent1 = (7982.6-oldpeakb1)*oldres1/newres1
width1 = 22.53*oldres1*1e3
new_prof_b1_16us,prof_b1_1us=centre_burst(new_prof_b1_16us,prof_b1_1us,tcent1,80)
#B2
oldpeakb2=598
oldres2 = 16e-6*5
newres2 = 16e-6*1
tcent2 = (597.9-oldpeakb2)*oldres2/newres2
width2 = 2.97*oldres2*1e3
new_prof_b2_16us,prof_b2_1us=centre_burst(new_prof_b2_16us,prof_b2_1us,tcent2,80)
#B3
oldpeakb3=8994
oldres3 = 16e-6*4
newres3 = 16e-6*1
tcent3 = (8987.6-oldpeakb3)*oldres3/newres3
width3 = 26.28*oldres3*1e3
new_prof_b3_16us,prof_b3_1us=centre_burst(new_prof_b3_16us,prof_b3_1us,tcent3,64)
#B4
oldpeakb4=1331
oldres4 = 16e-6*1
newres4 = 16e-6*1
tcent4 = (1363.125-oldpeakb4)*oldres4/newres4
width4 = 103.75*oldres4*1e3
new_prof_b4_16us,prof_b4_1us=centre_burst(new_prof_b4_16us,prof_b4_1us,tcent4,16)


#x-axes (time)
x1=np.arange(0,len(new_prof_b1_16us),1)
x1=x1-int(len(x1)/2.)
x1*=16 #us

x1_1us=np.arange(0,len(prof_b1_1us),1)
x1_1us=x1_1us-int(len(x1_1us)/2.)

x2=np.arange(0,len(new_prof_b2_16us),1)
x2=x2-int(len(x2)/2.)
x2*=16 #us

x2_1us=np.arange(0,len(prof_b2_1us),1)
x2_1us=x2_1us-int(len(x2_1us)/2.)

x3=np.arange(0,len(new_prof_b3_16us),1)
x3=x3-int(len(x3)/2.)
x3*=16 #us

x3_1us=np.arange(0,len(prof_b3_1us),1)
x3_1us=x3_1us-int(len(x3_1us)/2.)

x4=np.arange(0,len(new_prof_b4_16us),1)
x4=x4-int(len(x4)/2.)
x4*=16 #us

x4_1us=np.arange(0,len(prof_b4_1us),1)
x4_1us=x4_1us-int(len(x4_1us)/2.)

#convert to flux densities
b1noise_16us=prof_b1_16us[0:30000]
b2noise_16us=prof_b2_16us[0:30000]
b3noise_16us=prof_b3_16us[0:50000]
b4noise_16us=prof_b4_16us[0:8000]

new_prof_b1_16us=convert_to_flux(new_prof_b1_16us,b1noise_16us,0.016,128,2,20/1.54)
new_prof_b2_16us=convert_to_flux(new_prof_b2_16us,b2noise_16us,0.016,128,2,20/1.54)
new_prof_b3_16us=convert_to_flux(new_prof_b3_16us,b3noise_16us,0.016,128,2,20/1.54)
new_prof_b4_16us=convert_to_flux(new_prof_b4_16us,b4noise_16us,0.016,128,2,20/1.54)

b1noise_1us = profile(load_archive('./1us/ep114a-fullcoherence_no0204_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012))
b2noise_1us = profile(load_archive('./1us/ep114a-fullcoherence_no0210_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012))
b3noise_1us = profile(load_archive('./1us/ep114a-fullcoherence_no0224_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012))
b4noise_1us = profile(load_archive('./1us/ep114a-fullcoherence_no0278_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012))

prof_b1_1us=convert_to_flux(prof_b1_1us,b1noise_1us,0.001,128,2,20/1.54)
prof_b2_1us=convert_to_flux(prof_b2_1us,b2noise_1us,0.001,128,2,20/1.54)
prof_b3_1us=convert_to_flux(prof_b3_1us,b3noise_1us,0.001,128,2,20/1.54)
prof_b4_1us=convert_to_flux(prof_b4_1us,b4noise_1us,0.001,128,2,20/1.54)

b1noise_1us=convert_to_flux(b1noise_1us,b1noise_1us,0.001,128,2,20/1.54)
b2noise_1us=convert_to_flux(b2noise_1us,b2noise_1us,0.001,128,2,20/1.54)
b3noise_1us=convert_to_flux(b3noise_1us,b3noise_1us,0.001,128,2,20/1.54)
b4noise_1us=convert_to_flux(b4noise_1us,b4noise_1us,0.001,128,2,20/1.54)

#smooth 1us profile
N=19 #window length for smoothing
a=int((N-1)/2.)
prof_b1_1us_smooth = smooth(prof_b1_1us,window_len=N,window='blackman')[a-1:-(a+1)]
prof_b2_1us_smooth = smooth(prof_b2_1us,window_len=N,window='blackman')[a-1:-(a+1)]
prof_b3_1us_smooth = smooth(prof_b3_1us,window_len=N,window='blackman')[a-1:-(a+1)]
prof_b4_1us_smooth = smooth(prof_b4_1us,window_len=N,window='blackman')[a-1:-(a+1)]

#set plotting limits
x16lim = 2000 #2ms
x1lim_b1= 2000
x1lim_b2= 2000
x1lim_b3= 2000
x1lim_b4= 2000


fig = plt.figure(figsize=(9, 10))
rows=2
cols=1
widths = [1]
heights = [2, 2]
gs = gridspec.GridSpec(ncols=cols, nrows=rows, top=0.96, bottom=0.64, left=0.1, right=0.5, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0.1)
gs1 = gridspec.GridSpec(ncols=cols, nrows=rows, top=0.96, bottom=0.64,left=0.55, right=0.95, width_ratios=widths, height_ratios=heights, wspace=0, hspace=0.1)
gs2 = gridspec.GridSpec(ncols=2, nrows=3, top=0.60, bottom=0.08,left=0.1, right=0.5, width_ratios=[1,1], height_ratios=[2,2,3], wspace=0.15, hspace=0.15)
gs3 = gridspec.GridSpec(ncols=2, nrows=3, top=0.60, bottom=0.08,left=0.55, right=0.95, width_ratios=[1,1], height_ratios=[2,2,3], wspace=0.15, hspace=0.15)

ax1 = fig.add_subplot(gs[0,0]) #B1 16us
ax1.plot(x1,new_prof_b1_16us,color='k',linestyle='steps-mid')
ax1.set_xlim((-x16lim,x16lim))
ax1.set_ylim((-0.6,1))
ax1.set_yticks((-0.4,0,0.4,0.8))
plt.setp(ax1.get_xticklabels(), visible=False)

point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('B1',u'16\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.85, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)
point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


ax2 = fig.add_subplot(gs[1,0]) #B1 1us
ax2.plot(x1_1us,prof_b1_1us,color='#7f7f7f',linestyle='steps-mid')
ax2.set_xlim((-x1lim_b1,x1lim_b1))
ax2.plot(x1_1us,prof_b1_1us_smooth,color='k')
ax2.set_ylim((-3,3))
ax2.set_yticks([-2,-1,0,1,2,3])
ax2.set_xticks([-2000,-1000,0,1000,2000])

point = ax2.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('B1',u'1\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.85, 0.97), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=10)
frame = plotlabel.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
frame.set_alpha(0.6)
plt.gca().add_artist(plotlabel)
point = ax2.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=10)
frame = plotlabel.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
frame.set_alpha(0.6)
plt.gca().add_artist(plotlabel)

ax3 = fig.add_subplot(gs1[0,0]) #B2 16us
ax3.plot(x2,new_prof_b2_16us,color='k',linestyle='steps-mid')
ax3.set_xlim((-x16lim,x16lim))
ax3.set_ylim((-0.5,1.3))
ax3.set_yticks((0,0.5,1))
plt.setp(ax3.get_xticklabels(), visible=False)

point = ax3.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('B2',u'16\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.85, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)
point = ax3.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax4 = fig.add_subplot(gs1[1,0]) #B2 1us
ax4.plot(x2_1us,prof_b2_1us,color='#7f7f7f',linestyle='steps-mid')
ax4.set_xlim((-x1lim_b2,x1lim_b2))
ax4.plot(x2_1us,prof_b2_1us_smooth,color='k')
ax4.set_ylim((-3,3))
ax4.set_yticks([-2,-1,0,1,2,3])
ax4.set_xticks([-2000,-1000,0,1000,2000])

point = ax4.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('B2',u'1\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.85, 0.97), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=10)
frame = plotlabel.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
frame.set_alpha(0.6)
plt.gca().add_artist(plotlabel)
point = ax4.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=True,markerscale=0,fontsize=10)
frame = plotlabel.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('white')
frame.set_alpha(0.6)
plt.gca().add_artist(plotlabel)


ax5 = fig.add_subplot(gs2[0,0:]) #B3 16us
ax5.plot(x3,new_prof_b3_16us,color='k',linestyle='steps-mid')
ax5.set_xlim((-x16lim,x16lim))
plt.setp(ax5.get_xticklabels(), visible=False)
ax5.set_ylim((-0.5,2.5))
ax5.set_yticks((0,1,2))

point = ax5.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('B3',u'16\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.85, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)
point = ax5.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax6 = fig.add_subplot(gs2[1,0:],sharex=ax5) #B3 1us
ax6.plot(x3_1us,prof_b3_1us,color='#7f7f7f',linestyle='steps-mid')
ax6.set_xlim((-x1lim_b3,x1lim_b3))
ax6.axvspan(-375,-305,color='#ffc750')
ax6.axvspan(190,590,color='#bf5fb7')
ax6.plot(x3_1us,prof_b3_1us_smooth,color='k')
ax6.set_yticks((-2,0,2,4,6))
ax6.set_xticks([-2000,-1000,0,1000,2000])

point = ax6.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('B3',u'1\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.85, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)
point = ax6.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('g',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax7 = fig.add_subplot(gs2[2,0]) #B3 zoom 1
ax7.plot(x3_1us,prof_b3_1us,color='#7f7f7f',linestyle='steps-mid')
ax7.set_xlim((-375,-305))
ax7.axvspan(-375,-305,0,0.1,color='#ffc750')
ax7.set_ylim((-1.3,6))
ax7.set_yticks((0,1,2,3,4,5))
ax7.set_xticks((-360,-320))
point = ax7.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('i',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax8 = fig.add_subplot(gs2[2,1]) #B3 zoom 2
ax8.plot(x3_1us,prof_b3_1us,color='#7f7f7f',linestyle='steps-mid')
ax8.set_xlim((190,590))
ax8.axvspan(190,590,0,0.1,color='#bf5fb7')
ax8.set_ylim((-1.5,4))
ax8.plot(x3_1us,prof_b3_1us_smooth,color='k')
ax8.set_yticks((0,1,2,3))
ax8.set_xticks((250,450))
point = ax8.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax8.legend((point,point), ('j',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax9 = fig.add_subplot(gs3[0,0:]) #B4 16us
ax9.plot(x4,new_prof_b4_16us,color='k',linestyle='steps-mid')
ax9.set_xlim((-x16lim,x16lim))
ax9.set_ylim((-1,20))
plt.setp(ax9.get_xticklabels(), visible=False)

point = ax9.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax9.legend((point,point), ('B4',u'16\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.85, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)
point = ax9.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax9.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax10=fig.add_subplot(gs3[1,0:],sharex=ax9) #B4 1us
ax10.plot(x4_1us,prof_b4_1us,color='#7f7f7f',linestyle='steps-mid')
ax10.set_xlim((-x16lim,x16lim))
ax10.axvspan(-570,-400,color='#bf5fb7')
ax10.axvspan(-900,-560,color='#ffc750')
ax10.plot(x4_1us,prof_b4_1us_smooth,color='k')
ax10.set_ylim((-2,22))
ax10.set_xticks([-2000,-1000,0,1000,2000])

point = ax10.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('B4',u'1\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.85, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)
point = ax10.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('h',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


ax11 = fig.add_subplot(gs3[2,0]) #B4 1us sub-burst1
ax11.plot(x4_1us,prof_b4_1us,color='#7f7f7f',linestyle='steps-mid')
ax11.set_xlim((-900,-560))
ax11.axvspan(-900,-560,0,0.1,color='#ffc750')
#plt.setp(ax11.get_yticklabels(), visible=False)
ax11.set_ylim((-1.5,6))
ax11.plot(x4_1us,prof_b4_1us_smooth,color='k')
ax11.set_yticks((0,1,2,3,4,5))
ax11.set_xticks((-800,-600))
point = ax11.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax11.legend((point,point), ('k',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax12 = fig.add_subplot(gs3[2,1]) #B4 1us sub-burst 2
ax12.plot(x4_1us,prof_b4_1us,color='#7f7f7f',linestyle='steps-mid')
ax12.set_xlim((-570,-400))
ax12.axvspan(-570,-400,0,0.1,color='#bf5fb7')
#plt.setp(ax12.get_yticklabels(), visible=False)
ax12.set_ylim((np.min(prof_b4_1us[-570:-400]),24))
ax12.plot(x4_1us,prof_b4_1us_smooth,color='k')
ax12.set_xticks((-550,-450))
point = ax12.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax12.legend((point,point), ('l',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


fig.text(0.04,0.5,'Flux Density [Jy]',va='center', rotation='vertical')
fig.text(0.5,0.04,u'Time [\u03bcs]',ha='center')

plt.savefig('Nimmo_fig1.eps',format='eps',dpi=300)
plt.show()
