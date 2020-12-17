
"""To make extended data figure 1 in Nimmo et al. 2020"""

from load_archive import load_archive
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from radiometer import radiometer
import matplotlib as mpl
from ACF_funcs import autocorr
from scipy.signal import periodogram

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'                                                                                                                           
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'



def profile(ds):
    """
    Given a dynamic spectrum ds (with polarisations) output a Stokes I profile
    """
    LL = ds[0,:,:]
    RR = ds[1,:,:]
    Ids=(LL+RR)/2.
    prof=np.sum(Ids,axis=0)
    return prof

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

def rebin_data(x, y, dx_new, yerr=None, method='sum', dx=None):
    """
    FROM STINGRAY PACKAGE
    Rebin some data to an arbitrary new data resolution. Either sum
    the data points in the new bins or average them.

    Parameters
    ----------
    x: iterable
        The dependent variable with some resolution ``dx_old = x[1]-x[0]``

    y: iterable
        The independent variable to be binned

    dx_new: float
        The new resolution of the dependent variable ``x``

    Other parameters
    ----------------
    yerr: iterable, optional
        The uncertainties of ``y``, to be propagated during binning.

    method: {``sum`` | ``average`` | ``mean``}, optional, default ``sum``
        The method to be used in binning. Either sum the samples ``y`` in
        each new bin of ``x``, or take the arithmetic mean.

    dx: float
        The old resolution (otherwise, calculated from median diff)

    Returns
    -------
    xbin: numpy.ndarray
        The midpoints of the new bins in ``x``

    ybin: numpy.ndarray
        The binned quantity ``y``

    ybin_err: numpy.ndarray
        The uncertainties of the binned values of ``y``.

    step_size: float
        The size of the binning step
    """

    y = np.asarray(y)
    yerr = np.asarray(apply_function_if_none(yerr, y, np.zeros_like))

    dx_old = apply_function_if_none(dx, np.diff(x), np.median)

    if dx_new < dx_old:
        raise ValueError("New frequency resolution must be larger than "
                         "old frequency resolution.")

    step_size = dx_new / dx_old

    output = []
    outputerr = []
    for i in np.arange(0, y.shape[0], step_size):
        total = 0
        totalerr = 0

        int_i = int(i)
        prev_frac = int_i + 1 - i
        prev_bin = int_i
        total += prev_frac * y[prev_bin]
        totalerr += prev_frac * (yerr[prev_bin] ** 2)

        if i + step_size < len(x):
            # Fractional part of next bin:
            next_frac = i + step_size - int(i + step_size)
            next_bin = int(i + step_size)
            total += next_frac * y[next_bin]
            totalerr += next_frac * (yerr[next_bin] ** 2)

        total += sum(y[int(i + 1):int(i + step_size)])
        totalerr += sum(yerr[int(i + 1):int(step_size)] ** 2)
        output.append(total)
        outputerr.append(np.sqrt(totalerr))

    output = np.asarray(output)
    outputerr = np.asarray(outputerr)

    if method in ['mean', 'avg', 'average', 'arithmetic mean']:
        ybin = output / np.float(step_size)
        ybinerr = outputerr / np.sqrt(np.float(step_size))

    elif method == "sum":
        ybin = output
        ybinerr = outputerr

    else:
        raise ValueError("Method for summing or averaging not recognized. "
                         "Please enter either 'sum' or 'mean'.")

    tseg = x[-1] - x[0] + dx_old

    if (tseg / dx_new % 1) > 0:
        ybin = ybin[:-1]
        ybinerr = ybinerr[:-1]

    new_x0 = (x[0] - (0.5 * dx_old)) + (0.5 * dx_new)
    xbin = np.arange(ybin.shape[0]) * dx_new + new_x0

    return xbin, ybin, ybinerr, step_size

def apply_function_if_none(variable, value, func):
    """
    Assign a function value to a variable if that variable has value ``None`` on input.

    Parameters
    ----------
    variable : object
        A variable with either some assigned value, or ``None``

    value : object
        A variable to go into the function

    func : function
    Function to apply to ``value``. Result is assigned to ``variable``

    Returns
    -------
        new_value : object
            The new value of ``variable``

    Examples
    --------
    >>> var = 4
    >>> value = np.zeros(10)
    >>> apply_function_if_none(var, value, np.mean)
    4
    >>> var = None
    >>> apply_function_if_none(var, value, lambda y: np.mean(y))
    0.0
    """
    if variable is None:
        return func(value)
    else:
        return variable


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

#Load in data for B3 and B4 at 1us resolution
ds_b3_1us = load_archive('./1us/ep114a-fullcoherence_no0224_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012)
ds_b4_1us = load_archive('./1us/ep114a-fullcoherence_no0278_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012)
prof_b3_1us= profile(ds_b3_1us)
prof_b4_1us= profile(ds_b4_1us)

oldpeakb41us=1331
oldres41us = 16e-6*1
newres41us = 1e-6*1
tcent41us = (1363.125-oldpeakb41us)*oldres41us/newres41us

oldpeakb31us=8994
oldres31us = 16e-6*4
newres31us = 1e-6*1
tcent31us = (8987.6-oldpeakb31us)*oldres31us/newres31us

peakbin_3 = int(np.argmax(prof_b3_1us)+tcent31us)
peakbin_4 = int(np.argmax(prof_b4_1us)+tcent41us)
newpeak_3 = int(len(prof_b3_1us)/2.)
newpeak_4 = int(len(prof_b4_1us)/2.)
if peakbin_3 > newpeak_3:
    prof_b3_1us = np.roll(prof_b3_1us,(len(prof_b3_1us)-peakbin_3)+newpeak_3)
if peakbin_3 < newpeak_3:
    prof_b3_1us = np.roll(prof_b3_1us,newpeak_3-peakbin_3)
if peakbin_4 > newpeak_4:
    prof_b4_1us = np.roll(prof_b4_1us,(len(prof_b4_1us)-peakbin_4)+newpeak_4)
if peakbin_4 < newpeak_4:
    prof_b4_1us = np.roll(prof_b4_1us,newpeak_4-peakbin_4)

#noise data
ds_b3_off_1us=load_archive('./1us/ep114a-fullcoherence_no0224_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012)
ds_b4_off_1us=load_archive('./1us/ep114a-fullcoherence_no0278_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012)
b3noise = profile(ds_b3_off_1us)
b4noise = profile(ds_b4_off_1us)

b3_1us=convert_to_flux(prof_b3_1us,b3noise,0.001,128,2,20/1.54)
b4_1us=convert_to_flux(prof_b4_1us,b4noise,0.001,128,2,20/1.54)
b3noise=convert_to_flux(b3noise,b3noise,0.001,128,2,20/1.54)
b4noise=convert_to_flux(b4noise,b4noise,0.001,128,2,20/1.54)


x3 = np.arange(0,len(b3_1us),1)
x3 = x3-int(len(b3_1us)/2.)
x4 = np.arange(0,len(b4_1us),1)
x4 = x4-int(len(b4_1us)/2.)

#crop out the relevant data B3 and B4-sb1 (basically structure where we don't need to remove an envelope)
cent_b3=int(len(b3_1us)/2.)
cent_b4=int(len(b4_1us)/2.)
b3_1us = b3_1us[cent_b3+350:cent_b3+1315]
b4_1us = b4_1us[cent_b4-900:cent_b4-560]
cent_b3=int(len(x3)/2.)
cent_b4=int(len(x4)/2.)
x3 = x3[cent_b3+350:cent_b3+1315]
x4 = x4[cent_b4-900:cent_b4-560]

#crop out the right amount of noise data                                                                                                                              
b3noise=b3noise[0:len(b3_1us)]
b4noise=b4noise[0:len(b4_1us)]

#smoothed profile
N=19 #window length for smoothing
a=int((N-1)/2.)
b3_smooth = smooth(b3_1us,window_len=N,window='blackman')[a-1:-(a+1)]
b4_smooth = smooth(b4_1us,window_len=N,window='blackman')[a-1:-(a+1)]

#ACFs #using Marcote et al. 2020 eq 1
ACF_b3 = autocorr(b3_1us, len(b3_1us))
ACF_b4 = autocorr(b4_1us, len(b4_1us))
ACF_x_b3 = np.arange(1,len(ACF_b3)+1,1)
ACF_x_b4 = np.arange(1,len(ACF_b4)+1,1)

#Power spectra
#compute the power spectra by taking the square of the fourier transform of the time series
freq_b4,ps_b4 = periodogram(b4_1us, fs=1.0/1.0e-6, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='spectrum')
freq_b4=freq_b4[1:]
ps_b4=ps_b4[1:]
freq_b3,ps_b3 = periodogram(b3_1us, fs=1.0/1.0e-6, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='spectrum')
freq_b3=freq_b3[1:]
ps_b3=ps_b3[1:]


new_res = (freq_b4[1]-freq_b4[0])*6

freq_b4_bin3, ps_b4_bin3, ybinerr, step_size = rebin_data(freq_b4, ps_b4, new_res, yerr=None, method='average', dx=None)
freq_b3_bin3, ps_b3_bin3, ybinerr, step_size = rebin_data(freq_b3, ps_b3, new_res, yerr=None, method='average', dx=None)

#We read in the fits that we determined using the STINGRAY package as described in the text of Nimmo et al. 2020
fit_b3 = np.load('B3-powerlaw-fit-fullres.npy')
fit_b4 = np.load('B4-powerlaw-fit-fullres.npy')

#repeat the analysis for off burst data to compare
#noise ACF
ACF_b3_noise = autocorr(b3noise, len(b3noise))
ACF_b4_noise = autocorr(b4noise, len(b4noise))


#noise power spectrum
freq_b4n,ps_b4n = periodogram(b4noise, fs=1.0/1.0e-6, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='spectrum')
freq_b4n=freq_b4n[1:]
ps_b4n=ps_b4n[1:]
freq_b3n,ps_b3n = periodogram(b3noise, fs=1.0/1.0e-6, window='boxcar', nfft=None, detrend='constant', return_onesided=True, scaling='spectrum')
freq_b3n=freq_b3n[1:]
ps_b3n=ps_b3n[1:]

fig = plt.figure(figsize=(9, 12))
rows=2
cols=1
widths = [1]
heights = [2, 2]

#outer boxes
outergs1 = gridspec.GridSpec(1, 1)
outergs1.update(bottom=.5,top=0.97,left=0.02,right=0.98)
outerax1 = fig.add_subplot(outergs1[0])
outerax1.tick_params(axis='both',which='both',bottom=0,left=0,labelbottom=0, labelleft=0,labeltop=0,labelright=0,right=0,top=0)
outergs2 = gridspec.GridSpec(1, 1)
outergs2.update(bottom=0.02,top=0.49,left=0.02,right=0.98)
outerax2 = fig.add_subplot(outergs2[0])
outerax2.tick_params(axis='both',which='both',bottom=0,left=0,labelbottom=0, labelleft=0,labeltop=0,labelright=0,right=0,top=0)

#inner plots
gs = gridspec.GridSpec(ncols=cols, nrows=rows, top=0.94, bottom=0.54, left=0.09,right=0.49, width_ratios=widths, height_ratios=heights, wspace=0.15, hspace=0.2)
gs1 = gridspec.GridSpec(ncols=cols, nrows=rows, top=0.94, bottom=0.54, left=0.57,right=0.97,width_ratios=widths, height_ratios=heights, wspace=0.15, hspace=0)
gs2 = gridspec.GridSpec(ncols=cols, nrows=rows, top=0.46, bottom=0.06,left=0.09,right=0.49, width_ratios=widths, height_ratios=heights, wspace=0.15, hspace=0.2)
gs3 = gridspec.GridSpec(ncols=cols, nrows=rows, top=0.46, bottom=0.06,left=0.57,right=0.97, width_ratios=widths, height_ratios=heights, wspace=0.15, hspace=0)

ax1 = fig.add_subplot(gs[0,0]) #B4-sb1 1us
ax1.plot(x4,b4_1us,color='#7f7f7f',linestyle='steps-mid',label='Profile')
ax1.set_ylabel('Flux Density [Jy]')
ax1.set_yticks((-1,0,1,2,3,4,5))
ax1.set_xticks((-900,-800,-700,-600))
ax1.set_xlabel(u'Time [\u03bcs]')
ax1.plot(x4,b4_smooth,color='k',label='Profile smoothed')
plotlabel=ax1.legend()
plt.gca().add_artist(plotlabel)
ax1.set_ylim((-1.5,5.5))
ax1.set_xlim((x4[0],x4[-1]))
point = ax1.scatter(x4[0], b4_1us[0], facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax2 = fig.add_subplot(gs[1,0]) #B4-sb1 ACF
ax2.plot(ACF_x_b4,ACF_b4_noise,color='#b2b2b2',label='Off burst ACF')
ax2.plot(ACF_x_b4,ACF_b4,color='k',label='ACF')
ax2.set_ylim((-0.4,0.6))
ax2.set_xlim((ACF_x_b4[0],ACF_x_b4[-1]))
ax2.set_xticks((0,100,200,300))
ax2.set_ylabel('ACF')
ax2.set_xlabel(u'Time Lag [\u03bcs]')
plotlabel=ax2.legend(loc='upper center')
plt.gca().add_artist(plotlabel)
point = ax2.scatter(100, 0.2, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax3 = fig.add_subplot(gs1[0,0]) #B4-sb1 ps
ax3.loglog(freq_b4n,ps_b4n,color='#b2b2b2',linestyle='steps-mid',label='Off burst power spectrum')
ax3.loglog(freq_b4,ps_b4,color='k',linestyle='steps-mid',label='Power spectrum')
ax3.loglog(freq_b4_bin3,ps_b4_bin3,color='orange',linestyle='steps-mid',label='Power spectrum binned',lw=2)
ax3.loglog(freq_b4,fit_b4,color='mediumvioletred',label='Power law fit',lw=3)
plt.setp(ax3.get_xticklabels(), visible=False)
ax3.set_xlim((6e3,5e5))
ax3.set_ylim((2e-5,4))
plotlabel=ax3.legend()
plt.gca().add_artist(plotlabel)
ax3.set_ylabel('Power [Jy$^2$]',labelpad=-0.1)
point = ax3.scatter(freq_b4[0], ps_b4[0], facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


ax4 = fig.add_subplot(gs1[1,0]) #B4-sb1 ps resid
ax4.loglog(freq_b4,2*ps_b4/fit_b4,linestyle='steps-mid',color='k',label='Residuals')
ax4.set_xlim((6e3,5e5))
ax4.set_ylim((1e-2,3e1))
ax4.legend(loc='lower left')
ax4.set_xlabel('Frequency [Hz]',labelpad=-0.05)
ax4.set_ylabel('Power [Jy$^2$]',labelpad=-0.1)
point = ax4.scatter(freq_b4[0], 1, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


ax5 = fig.add_subplot(gs2[0,0]) #B3 1us
ax5.plot(x3,b3_1us,color='#7f7f7f',linestyle='steps-mid',label='Profile')
ax5.set_xlim((350,x3[-1]))
ax5.set_ylabel('Flux Density [Jy]')
ax5.set_yticks((-2,-1,0,1,2,3,4,5))
ax5.set_xlabel(u'Time [\u03bcs]')
ax5.plot(x3,b3_smooth,color='k',label='Profile smoothed')
plotlabel=ax5.legend()
plt.gca().add_artist(plotlabel)
point = ax5.scatter(x3[0], b3_1us[0], facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.07, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax6 = fig.add_subplot(gs2[1,0]) #B3 ACF
ax6.plot(ACF_x_b3,ACF_b3_noise,color='#b2b2b2',label='Off burst ACF')
ax6.plot(ACF_x_b3,ACF_b3,color='k',label='ACF')
ax6.set_ylim((-0.2,0.4))
ax6.set_xlim((ACF_x_b3[0],ACF_x_b3[-1]))
ax6.set_ylabel('ACF')
ax6.set_xlabel(u'Time Lag [\u03bcs]')
plotlabel=ax6.legend(loc='upper center')
plt.gca().add_artist(plotlabel)
point = ax6.scatter(100, 0.2, facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax7 = fig.add_subplot(gs3[0,0]) #B3 ps
ax7.loglog(freq_b3,ps_b3n,color='#b2b2b2',linestyle='steps-mid',label='Off burst power spectrum')
ax7.loglog(freq_b3,ps_b3,color='k',linestyle='steps-mid',label='Power spectrum')
ax7.loglog(freq_b3_bin3,ps_b3_bin3,color='orange',linestyle='steps-mid',label='Power spectrum binned',lw=2)
ax7.loglog(freq_b3,fit_b3,color='mediumvioletred',label='Power law fit',lw=3)
plt.setp(ax7.get_xticklabels(), visible=False)
ax7.set_xlim((6e3,5e5))
ax7.set_ylim((1e-5,6e-1))
leg=ax7.legend()
frame = leg.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')
frame.set_alpha(0.6)
plt.gca().add_artist(leg)
ax7.set_ylabel('Power [Jy$^2$]',labelpad=-0.1)
point = ax7.scatter(freq_b3[0], ps_b3[0], facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('g',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax8 = fig.add_subplot(gs3[1,0]) #B3 ps resid
ax8.loglog(freq_b3,2*ps_b3/fit_b3,linestyle='steps-mid',color='k',label='Residuals')
ax8.set_xlim((6e3,5e5))
ax8.set_ylim((1e-2,3e1))
ax8.legend(loc='lower left')
ax8.set_xlabel('Frequency [Hz]',labelpad=-0.05)
ax8.set_ylabel('Power [Jy$^2$]',labelpad=-0.1)
point = ax8.scatter(freq_b3[0], 1, facecolors='none', edgecolors='none')
plotlabel=ax8.legend((point,point), ('h',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

fig.text(0.025,0.95,'B4-sb1',fontsize=12)
fig.text(0.025,0.47,'B3',fontsize=12)
plt.savefig('Nimmo_ed-figure1.eps',format='eps',dpi=300)
plt.show()
