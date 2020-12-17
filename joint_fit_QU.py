#from lmfit import minimize, Parameters, fit_report
import numpy as np
import matplotlib.pyplot as plt
from load_archive import load_archive
from pol_prof import get_profile

def linear_pol_w_psr(params,freq=None,Qdata=None,Udata=None,freq_psr=None,Qdata_psr=None,Udata_psr=None):
    """
    function to fit Q and U as a function of frequency for both the pulsar and the burst
    Give as input

    """
    #parameters for this model
    RM = params['RM']
    delay=params['delay']
    offset=params['offset']
    offsetp=params['offsetp']
    A = params['A']
    Ap=params['Ap']
    RMp = params['RMp']

    #burst Q and U
    Q = A*np.cos(2*(((3e8)**2/(freq*1e6)**2)*(RM) + (freq*1e6)*delay*np.pi + offset))
    U = A*np.sin(2*(((3e8)**2/(freq*1e6)**2)*(RM) + (freq*1e6)*delay*np.pi + offset))
    #pulsar Q and U
    Qp = Ap*np.cos(2*(((3e8)**2/(freq_psr*1e6)**2)*(RMp) + (freq_psr*1e6)*delay*np.pi + offsetp))
    Up = Ap*np.sin(2*(((3e8)**2/(freq_psr*1e6)**2)*(RMp) + (freq_psr*1e6)*delay*np.pi + offsetp))
    #residuals
    residQ = Qdata - Q
    residU = Udata - U
    residQ_psr = Qdata_psr - Qp
    residU_psr = Udata_psr - Up

    return np.concatenate((residQ,residU,residQ_psr,residU_psr))

def Q_func(params,freq):
    RM = params['RM']
    delay=params['delay']
    offset=params['offset']
    A = params['A']
    return A*np.cos(2*(((3e8)**2/(freq*1e6)**2)*(RM) + (freq*1e6)*delay*np.pi + offset))

def U_func(params,freq):
    RM =params['RM']
    delay=params['delay']
    offset=params['offset']
    A = params['A']
    return A*np.sin(2*(((3e8)**2/(freq*1e6)**2)*(RM) + (freq*1e6)*delay*np.pi + offset))

def f_scrunch(data,newnchan):
    return np.array(np.row_stack([np.mean(subint, axis=0) for subint in np.vsplit(data,newnchan)]))

def QU_data(ds,freqs,offbegin,offend,burstbegin,burstend,downsamp_freq=1):
    """
    Function to read in dynamic spectrum and output the on burst Q and U as a function of frequency.

    -  ds is the dynamic spectrum output from load_archive
    -  offbegin and offend define the offburst time range in bins
    -  burstbegin and burstend define the burst time range in bins
    -  downsamp_freq is the downsample factor in frequency (might help with fit)
    """
    LL = ds[0,:,:]
    RR = ds[1,:,:]
    RLr = ds[2,:,:]
    RLi = ds[3,:,:]
    offRR = RR[:,offbegin:offend]
    offLL = LL[:,offbegin:offend]
    offRLr = RLr[:,offbegin:offend]
    offRLi = RLi[:,offbegin:offend]

    if downsamp_freq>1:
        favg = int(downsamp_freq)
        nchan_tot = np.shape(RR)[0]
        if nchan_tot % favg != 0:
            print("The total number of channels is %s, please choose an fscrunch value that divides the total number of channels."%nchan_tot)
            sys.exit()
        else:
            newnchan=nchan_tot/favg
            RR = f_scrunch(RR,newnchan)
            LL = f_scrunch(LL,newnchan)
            RLr = f_scrunch(RLr,newnchan)
            RLi = f_scrunch(RLi,newnchan)
            offRR = f_scrunch(offRR,newnchan)
            offLL = f_scrunch(offLL,newnchan)
            offRLr = f_scrunch(offRLr,newnchan)
            offRLi = f_scrunch(offRLi,newnchan)

    RRprofile = np.mean(RR,axis=0)
    LLprofile = np.mean(LL,axis=0)
    #put the pulse in the centre
    #not super crucial but done it anyway
    peakbin = np.where((RRprofile+LLprofile)==np.max(RRprofile+LLprofile))[0][0]
    newpeak = int(len(RRprofile)/2.)
    for i in range(np.shape(RR)[0]):
        if peakbin > newpeak:
            RR[i,:] = np.roll(RR[i,:],(len(RRprofile)-peakbin)+newpeak)
            LL[i,:] = np.roll(LL[i,:],(len(LLprofile)-peakbin)+newpeak)
            RLr[i,:] = np.roll(RLr[i,:],(len(RRprofile)-peakbin)+newpeak)
            RLi[i,:] = np.roll(RLi[i,:],(len(LLprofile)-peakbin)+newpeak)
        if peakbin < newpeak:
            RR[i,:] = np.roll(RR[i,:],newpeak-peakbin)
            LL[i,:] = np.roll(LL[i,:],newpeak-peakbin)
            RLr[i,:] = np.roll(RLr[i,:],newpeak-peakbin)
            RLi[i,:] = np.roll(RLi[i,:],newpeak-peakbin)
        if peakbin == newpeak:
            break

    RR = RR[:,burstbegin:burstend]
    LL = LL[:,burstbegin:burstend]
    RLr = RLr[:,burstbegin:burstend]
    RLi = RLi[:,burstbegin:burstend]

    offRRspec = np.sum(offRR,axis=1)
    offLLspec = np.sum(offLL,axis=1)
    offRLrspec = np.sum(offRLr,axis=1)
    offRLispec = np.sum(offRLi,axis=1)
    RRspec = np.sum(RR,axis=1)
    LLspec = np.sum(LL,axis=1)
    RLrspec = np.sum(RLr,axis=1)
    RLispec = np.sum(RLi,axis=1)


    I = RRspec+LLspec
    Q = 2*RLrspec
    U = 2*RLispec
    V = LLspec - RRspec

    polyIcoeff = np.polyfit(freqs,I,2)
    polyI = np.poly1d(polyIcoeff)

    #since we're dividing by I to determine Q/I and U/I we mask where I==0
    I = np.ma.masked_where(I==0,I)
    Q = np.ma.masked_where(I==0,Q)
    U = np.ma.masked_where(I==0,U)

    QI=Q/polyI(freqs)
    UI=U/polyI(freqs)

    return QI, UI

params = Parameters()
params.add('delay', value=5.5e-9,min=4.7e-9,max=6.2e-9)
params.add('offset', value=0,min=-50, max=50)
params.add('A', value=-1.03,min=-1.5,max=-0.9)
params.add('RM', value=-114,min=-150,max=-90)
params.add('offsetp', value=1,min=-10, max=10)
params.add('Ap', value=1,min=-1.5,max=1.5)
params.add('RMp', value=-218.7)
params['RMp'].vary = False

#archive = 'burst1.b4.fp.fc'
archive = 'burst4.fp.fc'

if archive == 'burst1.b4.fp.fc':
    ds, extent = load_archive('./16us/'+archive,rm=0,dm=0.012,extent=True)
    favg = 1
    freqs=np.linspace(extent[2],extent[3],np.shape(ds)[1]/favg)
    Qdata, Udata = QU_data(ds,freqs,8000,16000,7977,8020,downsamp_freq=favg)


if archive == 'burst4.fp.fc':
    ds, extent = load_archive('./16us/'+archive,rm=0,dm=0.012,extent=True)
    favg = 4
    freqs=np.linspace(extent[2],extent[3],np.shape(ds)[1]/favg)
    Qdata,Udata=QU_data(ds,freqs,0,9000,8165,8296,downsamp_freq=favg)


#pulsar data
ds_psr, ext_psr = load_archive('./pulsar/ep114a-fullcoherence_no0201_EfEf.ar.fp',extent=True)
favg = 1
freqs_psr=np.linspace(ext_psr[2],ext_psr[3],np.shape(ds_psr)[1]/favg)
Qdata_psr,Udata_psr = QU_data(ds_psr,freqs_psr,0,200,230,339,downsamp_freq=favg)

#perform the QU fit
out = minimize(linear_pol_w_psr, params, kws={"freq": freqs, "Qdata": Qdata, "Udata": Udata, "freq_psr":freqs_psr,"Qdata_psr": Qdata_psr, "Udata_psr": Udata_psr})
print(fit_report(out))

fig,axes=plt.subplots(2,figsize=[8,8],sharex=True)

axes[0].plot(freqs,Qdata,'o')
axes[0].plot(freqs,Q_func(out.params,freqs),'r')
axes[1].plot(freqs,Udata,'o')
axes[1].plot(freqs,U_func(out.params,freqs),'r')
plt.text(0.16,1.1,"Fit RM %s, fit delay %s seconds"%( out.params['RM'].value,out.params['delay'].value),transform=axes[0].transAxes)
plt.show()
