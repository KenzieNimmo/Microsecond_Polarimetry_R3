
"""To make Fig2 in Nimmo et al. 2020"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import transforms
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy import stats
from radiometer import radiometer
from load_archive import load_archive
from scipy.signal import correlate2d as corr


mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'                                                                                                                           
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'


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
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


if __name__=='__main__':
        timeres = 1e-6 #seconds
        SEFD = 20./1.54 #for Effelsberg

        #load in the 1us B4 archive and make a Stokes I profile
        ds_b4_1us = load_archive('./1us/ep114a-fullcoherence_no0278_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012,remove_baseline=False)
        LL_b4_1us = ds_b4_1us[0,:,:]
        RR_b4_1us = ds_b4_1us[1,:,:]
        Ids_b4_1us=(LL_b4_1us+RR_b4_1us)/2.
        I=np.sum(Ids_b4_1us,axis=0)

        #load in 1us noise file and make a noise Stokes I time series
        ds_b4_off_1us=load_archive('./1us/ep114a-fullcoherence_no0278_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012,remove_baseline=False)
        LL_b4_off_1us = ds_b4_off_1us[0,:,:]
        RR_b4_off_1us = ds_b4_off_1us[1,:,:]
        Ids_b4_off_1us=(LL_b4_off_1us+RR_b4_off_1us)/2.
        Inoise=np.sum(Ids_b4_off_1us,axis=0)

        #cut out a subset of the time bins
        peakbin = np.argmax(I)
        nbin = 100
        Inoise = Inoise[:nbin*2]
        I=I[peakbin-nbin:peakbin+nbin]

        x=np.arange(0,len(I),1)
        x-=int(len(x)/2.)
        x=np.array([float(i) for i in x])
        x*=timeres*1e6 #in us

        # smoothing the profile. We do this to define an "envelope" of the burst which we can remove
        #from the profile to leave behind any structure on top of the envelope
        N=19
        smootharr = smooth(I,window_len=N,window='blackman')
        a=int((N-1)/2.)
        smootharr=smootharr[a+1:-(a-1)]
        smoothnoisearr =smooth(Inoise,window_len=N,window='blackman')
        smoothnoisearr=smoothnoisearr[a+1:-(a-1)]

        #Convert to flux units
        noisearrflux=Inoise.copy()
        arrflux =I.copy()
        #first convert to S/N
        arrflux-=np.mean(noisearrflux)
        noisearrflux-=np.mean(noisearrflux)
        noisestd=np.std(noisearrflux)
        arrflux/=noisestd
        noisearrflux/=noisestd
        #arrSN is the S/N units Stokes I profile
        arrSN=arrflux.copy()
        arrnoiseSN=noisearrflux.copy()

        smootharrflux = smooth(arrflux,window_len=N,window='blackman')[a+1:-(a-1)]
        smoothnoisearrflux = smooth(noisearrflux,window_len=N,window='blackman')[a+1:-(a-1)]
        #now use the radiometer equation to convert from S/N to physical units
        fluxarr =arrflux*radiometer(timeres*1000,128,2, SEFD)
        smoothfluxarr=smootharrflux*radiometer(timeres*1000,128,2, SEFD)
        noiseflux = smoothnoisearrflux*radiometer(timeres*1000,128,2, SEFD)
        error_flux= np.std(noiseflux)

        #de-enveloping the burst (and off burst) data
        noisearr=Inoise
        arr=I
        smootharr = smooth(arr,window_len=N,window='blackman')[a+1:-(a-1)]
        smoothnoisearr=smooth(noisearr,window_len=N,window='blackman')[a+1:-(a-1)]
        arr/=smootharr
        noisearr/=smoothnoisearr

        #convert to S/N for histogram
        arr-=np.mean(noisearr)
        noisearr-=np.mean(noisearr)
        arr/=np.std(noisearr)
        noisearr/=np.std(noisearr)


        #correlations
        StokesI=Ids_b4_1us[:,peakbin-nbin:peakbin+nbin]
        StokesI = np.ma.masked_where(StokesI==0,StokesI)
        StokesInoise=Ids_b4_off_1us[:,:2*nbin]
        noisespec = np.mean(StokesInoise,axis=1)

        #define S/N threshold
        indices=np.where(arrSN>15.)[0]

        #correlations of individual time bin spectra with each other
        #we use the colour scheme of the geometric mean between the S/N of the two spectra we correlate
        corrcoefs=np.zeros((len(indices),len(indices)))
        positions=np.zeros((len(indices),len(indices)))
        geometric_means = np.zeros((len(indices),len(indices)))
        for i in range(len(indices)):
            for j in range(len(indices)):
                 Stokes1=StokesI[:,indices[i]]-np.mean(noisespec)
                 Stokes2=StokesI[:,indices[j]]-np.mean(noisespec)
                 newnoisespec = noisespec-np.mean(noisespec)
                 Stokes1/=np.std(newnoisespec)
                 Stokes2/=np.std(newnoisespec)

                 corrcoefs[i,j]=np.corrcoef(Stokes1,Stokes2)[0,1]
                 positions[i,j]=np.abs(indices[j]-indices[i])*1
                 geometric_means[i,j]=np.sqrt(arrSN[indices[j]]*arrSN[indices[i]])


        #get unique values of time lags in order to determine the weighted mean of the corr coeffs per delta time
        timelags=np.unique(positions)
        timelags = timelags[np.where(timelags!=0)[0]]

        corrcoefs=corrcoefs[np.where(positions!=0)]
        geometric_means=geometric_means[np.where(positions!=0)]
        positions = positions[np.where(positions!=0)]

        corrcoefs_mean = np.zeros_like(timelags)
        corrcoefs_std = np.zeros_like(timelags)
        SNscale = geometric_means/np.max(geometric_means)

        for i in range(len(timelags)):
            indices_lag = np.where(positions==timelags[i])

            corrcoefs_mean[i]=np.sum(SNscale[indices_lag]*corrcoefs[indices_lag])/np.sum(SNscale[indices_lag])#np.average(corrcoefs[indices_lag],weights=SNscale[indices_lag])
            n = len(SNscale[indices_lag])
            numerator = np.sum(n * SNscale[indices_lag] * (corrcoefs[indices_lag] - corrcoefs_mean[i]) ** 2.0)
            denominator = (n - 1) * np.sum(SNscale[indices_lag])
            weighted_std = np.sqrt(numerator / denominator)
            corrcoefs_std[i]=weighted_std


        #Figure 2 of Nimmo et al. 2020

        fig = plt.figure(figsize=(8, 10))
        cmap = plt.cm.gist_yarg
        widths = [3,3]
        heights = [3,3,3]
        outer = gridspec.GridSpec(2, 2, width_ratios = [2, 3],height_ratios=[3,2],wspace=0.3,hspace=0.2)
        gs = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=2,subplot_spec = outer[0,0], height_ratios=[1,1], wspace=0.0, hspace=0.0)
        gs1 = gridspec.GridSpecFromSubplotSpec(ncols=2, nrows=2,subplot_spec = outer[0,1],width_ratios=[2,1], height_ratios=[1,1], wspace=0.0, hspace=0.0)
        gs2 = gridspec.GridSpecFromSubplotSpec(ncols=1, nrows=1,subplot_spec = outer[1,:])

        ax1 = fig.add_subplot(gs[0,0]) #profile B4
        ax1.step(x,fluxarr,'k',alpha=1.0)
        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set_ylabel('Flux Density [Jy]')
        ax1.set_yticks((0,5,10,15,20,25))
        ax1.set_ylim(-3,30)
        ax1.set_xlim(-100,100)
        point = ax1.scatter(x[0], I[0], facecolors='none', edgecolors='none')
        burstname=ax1.legend((point,point), ('B4',u'1 \u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.86, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
        point = ax1.scatter(x[0], I[0], facecolors='none', edgecolors='none')
        ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.65), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
        plt.gca().add_artist(burstname)

        #add inset with 16us data
        ds_b4_16us=load_archive('./16us/burst4.fp.fc',rm=-536.18,dm=0.012)
        LL_b4_16us = ds_b4_16us[0,:,:]
        RR_b4_16us = ds_b4_16us[1,:,:]
        Ids_b4_16us=(LL_b4_16us+RR_b4_16us)/2.
        I_16=np.sum(Ids_b4_16us,axis=0)
        x_16=np.arange(0,len(I_16),1)

        oldpeakb4=1331
        tcent4 = (1363.125-oldpeakb4)
        peakbin_16=np.argmax(I_16)
        peakbin_16+=int(tcent4)
        cent_16 = int(len(I_16)/2.)
        if peakbin_16 > cent_16:
            I_16 = np.roll(I_16,(len(I_16)-peakbin_16)+cent_16)
        if peakbin_16 < cent_16:
            I_16 = np.roll(I_16,cent_16-peakbin_16)

        bin_16 = int(125)
        max_16 =np.max(I_16)

        axins = inset_axes(ax1, width=0.9, height=0.6, loc=2)
        axins.plot(x_16[cent_16-bin_16:cent_16+bin_16],I_16[cent_16-bin_16:cent_16+bin_16]/max_16, 'k')
        peakbin=np.argmax(I_16[cent_16-bin_16:cent_16+bin_16]/max_16)
        axins.axvspan(x_16[cent_16-bin_16+peakbin-6],x_16[cent_16-bin_16+peakbin+6],alpha=0.3,color='k')
        x1, x2, y1, y2 = 8120,8300,-0.05,1.05
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        plt.yticks(visible=False)
        plt.xticks(visible=False)

        ax5 = fig.add_subplot(gs[1,0],sharex=ax1) #dynamic spectrum B4
        ax5.imshow(StokesI,interpolation='none',origin='lower',cmap='binary',aspect='auto',vmin=np.min(StokesI),vmax=np.max(StokesI),extent=(-100,100,1.59499,1.72249))
        ax5.set_ylabel('Frequency [GHz]')
        ax5.set_xlabel(u'Time [\u03bcs]')
        point = ax5.scatter(0,1.6, facecolors='none', edgecolors='none')
        ax5.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
        ax5.set_xlim(-100,100)

        ax2 = fig.add_subplot(gs1[0,0]) #smoothed profile B4
        point = ax2.scatter(x[0], smoothfluxarr[0], facecolors='none', edgecolors='none')
        plotlabel=ax2.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
        ax2.step(x,smoothfluxarr,color='k',label='Stokes I smoothed')
        ax2.scatter(x,fluxarr,marker='o',color='k',alpha=0.2, label='Stokes I unsmoothed')
        ax2.set_ylabel('Flux Density [Jy]')
        ax2.set_xlim(-100,100)
        ax2.set_ylim(-3,30)
        ax2.legend()
        ax2.set_yticks((0,5,10,15,20,25))
        plt.setp(ax2.get_xticklabels(), visible=False)
        plt.gca().add_artist(plotlabel)


        ax3 = fig.add_subplot(gs1[1,0],sharex=ax2) #de enveloped profile and comparison with noise
        point = ax3.scatter(x[0], arr[0], facecolors='none', edgecolors='none')
        plotlabel=ax3.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
        ax3.step(x, arr,color='k',label='De-enveloped Stokes I burst')
        ax3.step(x, noisearr,color='limegreen',label='De-enveloped off-burst noise')
        ax3.legend()
        ax3.set_xlim(-100,100)
        ax3.set_yticks([-3,-2,-1,0,1,2,3,4,5])
        ax3.set_ylim(-4,6)
        ax3.set_ylabel('S/N')
        ax3.set_xlabel(u'Time [\u03bcs]')
        plt.gca().add_artist(plotlabel)


        ax7 = fig.add_subplot(gs1[1,1],sharey=ax3) #histogram of on-burst vs off-burst
        base = plt.gca().transData
        rot = transforms.Affine2D().rotate_deg(-90)
        nbin_hist = int(60/1)
        cent=int(len(arr)/2.)
        hist, bin_edges = np.histogram(arr[cent-nbin_hist-int(10/1):cent+nbin_hist],bins=np.arange(-5.5,7.5,1))
        hist_noise,b= np.histogram(noisearr[cent-nbin_hist-int(10/1):cent+nbin_hist],bins=np.arange(-5.5,7.5,1))
        ax7.fill_between(np.flip(np.arange(-5.5,6.5,1),axis=0),hist,color='k',transform= rot + base,alpha=0.3,step='post')
        ax7.fill_between(np.flip(np.arange(-5.5,6.5,1),axis=0),hist_noise,color='limegreen',transform= rot + base,alpha=0.3,step='post')
        ax7.plot(np.flip(np.arange(-5.5,6.5,1),axis=0),hist_noise, drawstyle="steps-post",color='limegreen',transform= rot + base)
        ax7.plot(np.flip(np.arange(-5.5,6.5,1),axis=0),hist, drawstyle="steps-post",color='k',transform= rot + base)
        ax7.set_ylim(-4,6)
        ax7.set_xlim(-3,np.max(hist))
        ax7.set_xticks([10,20,30,40])
        ax7.set_xlabel('Counts')
        plt.setp(ax7.get_yticklabels(), visible=False)
        ax7.set_transform(rot + base)
        point = ax7.scatter(0, 10, facecolors='none', edgecolors='none')
        plotlabel=ax7.legend((point,point), ('e',), loc='upper right',handlelength=0,bbox_to_anchor=(0.95, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
        plt.gca().add_artist(plotlabel)

        ax4 = fig.add_subplot(gs2[0,0]) #Correlation coefficient
        im=ax4.scatter(positions,corrcoefs,marker='x',c=geometric_means,cmap='Purples',alpha=1.0)
        point = ax4.scatter(x[0], I[0], facecolors='none', edgecolors='none')
        ax4.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.95, 0.97), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
        ax4.errorbar(timelags,corrcoefs_mean,yerr=corrcoefs_std,color='k')
        ax4.set_ylim(-0.2,0.5)
        ax4.set_xlim(0,65)
        ax4.set_ylabel('Correlation coefficient')
        ax4.set_xlabel(u'Time lag [\u03bcs]')
        cbar=fig.colorbar(im)
        cbar.set_label("Geometric mean S/N",rotation=-90,labelpad=20)
        plt.savefig('Nimmo_fig2.eps',format='eps',dpi=300)
        plt.show()
        exit()
