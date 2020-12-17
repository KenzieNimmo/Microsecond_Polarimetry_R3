import numpy as np
from scipy.interpolate import interp1d
import PA_errors


def get_profile(ds,offbegin,offend,pa_deg,PAoffset=None, burstbeg = None, burstend=None,tcent=None,name=None,Lbias=False,offpulsedata=np.array([None])):
    """
    ds is with dimensions polarisation, channel, profile bin.
    offbegin and offend are the time bins of the off burst times.
    pa_deg is the parallactic angle in degrees
    Polarisation is in the form RR, LL, RL_real, RL_imaginary

    Optional arguments: PAoffset to shift the polarisation position angle by some angle
                        (for this paper (Nimmo et al. 2020) we use the weighted mean PPA of the four bursts).
                        burstbeg burstend are the beginning and end times of the burst to compute the polarisation fractions
                        tcent is the centre time (relative to the peak time, in bins) of the burst (to put at time=0 -- matches what we did previously),
                        if not given, the peak of the burst is taken as tcent.
                        name is for saving numpy arrays with the filename name_*.npy. If name=None, no files will be saved

    Stokes params are defined as (PSR/IEEE convention)
    I = RR + LL
    Q = 2*RL_real
    U = 2*i*RL_imaginary
    V = LL - RR

    linear is sqrt(Q^2+U^2) which is equivalent to sqrt(Q^2-abs(U^2))
    """
    #read in data
    LL = ds[0,:,:]
    RR = ds[1,:,:]
    RLr = ds[2,:,:]
    RLi = ds[3,:,:]
    #read in off burst data
    if (offpulsedata!=None).all():
        offLL = offpulsedata[0,:,:]
        offRR = offpulsedata[1,:,:]
        offRLr = offpulsedata[2,:,:]
        offRLi = offpulsedata[3,:,:]
    else:
        offRR = RR[:,offbegin:offend]
        offLL = LL[:,offbegin:offend]
        offRLr = RLr[:,offbegin:offend]
        offRLi = RLi[:,offbegin:offend]

    RRprofile = np.sum(RR,axis=0)
    RRoffprof = np.sum(offRR,axis=0)

    LLprofile = np.sum(LL,axis=0)
    LLoffprof = np.sum(offLL,axis=0)

    RLrprofile = np.sum(RLr,axis=0)
    RLroffprof = np.sum(offRLr,axis=0)

    RLiprofile = np.sum(RLi,axis=0)
    RLioffprof = np.sum(offRLi,axis=0)

    #create Stokes parameters with correct statistics
    I=RRprofile+LLprofile
    Q = 2*RLrprofile
    U = 2*RLiprofile
    V=LLprofile-RRprofile

    Ioff = RRoffprof+LLoffprof
    Qoff = 2*RLroffprof
    Uoff = 2*RLioffprof
    Voff = LLoffprof-RRoffprof

    #statistics should be ~ the same for all four Stokes parameters
    #use 1% as limit within which the statistics should agree
    Ivar=np.var(Ioff)
    Ivar_var = 0.01*Ivar
    sigmaQ = np.std(Qoff)
    sigmaU = np.std(Uoff)
    sigmaV = np.std(Voff)
    Ioffstd = np.std(Ioff)
    if ((Ivar+Ivar_var)>np.var(Qoff)>(Ivar-Ivar_var)) and ((Ivar+Ivar_var)>np.var(Uoff)>(Ivar-Ivar_var)) and ((Ivar+Ivar_var)>np.var(Voff)>(Ivar-Ivar_var)):
        print("Ivar", np.var(Ioff))
        print("Qvar", np.var(Qoff))
        print("Uvar", np.var(Uoff))
        print("Vvar", np.var(Voff))
    else:
        I/=Ioffstd
        Ioff/=Ioffstd
        Q/=sigmaQ
        Qoff/=sigmaQ
        U/=sigmaU
        Uoff/=sigmaU
        V/=sigmaV
        Voff/=sigmaV
        #Note we remove the baseline already in the load_archive function
        print("Ivar", np.var(Ioff))
        print("Qvar", np.var(Qoff))
        print("Uvar", np.var(Uoff))
        print("Vvar", np.var(Voff))
        sigmaQ = np.std(Qoff)
        sigmaU = np.std(Uoff)
        sigmaV = np.std(Voff)
        Ioffstd = np.std(Ioff)


    linear = np.sqrt(Q**2+U**2)
    linearoff = np.sqrt(Qoff**2+Uoff**2)

    Ioffstd = np.std(Ioff)

    print("Ioffstd", Ioffstd)

    peakbin = np.where((I)==np.max(I))[0][0]
    if tcent==None:
        peakbin = peakbin
    else: peakbin+=int(tcent)
    newpeak = int(len(I)/2.)
    if peakbin > newpeak:
        RLrprofile = np.roll(RLrprofile,(len(I)-peakbin)+newpeak)
        RLiprofile = np.roll(RLiprofile,(len(I)-peakbin)+newpeak)
        linear = np.roll(linear,(len(I)-peakbin)+newpeak)
        I = np.roll(I,(len(I)-peakbin)+newpeak)
        V = np.roll(V,(len(I)-peakbin)+newpeak)
        Q = np.roll(Q,(len(I)-peakbin)+newpeak)
        U = np.roll(U,(len(I)-peakbin)+newpeak)
    if peakbin < newpeak:
        RLrprofile = np.roll(RLrprofile,newpeak-peakbin)
        RLiprofile = np.roll(RLiprofile,newpeak-peakbin)
        linear = np.roll(linear,newpeak-peakbin)
        I = np.roll(I,newpeak-peakbin)
        V = np.roll(V,newpeak-peakbin)
        Q = np.roll(Q,newpeak-peakbin)
        U = np.roll(U,newpeak-peakbin)
    if peakbin==newpeak:
        I=I


    #debiasing following Everett and Weisberg 2001
    Ltrue = np.zeros_like(linear)
    for i in range(len(linear)):
        if linear[i]/Ioffstd > 1.57:
            Ltrue[i] = Ioffstd*np.sqrt((linear[i]/Ioffstd)**2-1)


    #errors on Ltrue
    sigmaLtrue = np.zeros_like(Ltrue)
    for i in range(len(Ltrue)):
        if Ltrue[i]>0:
            sigmaL = 0.5*np.sqrt(2*(Q[i]*sigmaQ)**2+2*(U[i]*sigmaU)**2)*(1/linear[i])
            sigmaLtrue2 = 2*sigmaL*Ltrue[i]**2/linear[i]
            sigmaLtrue[i] = 0.5*sigmaLtrue2/Ltrue[i]


    #polarisation position angle (PPA)
    PA = np.zeros_like(I)
    for i in range(len(I)):
        PA[i]=(0.5*np.arctan2(U[i],(Q[i])))
    PA*=(180./np.pi) #converting to degrees
    PA+=pa_deg #rotating the linear polarisation vector by the parallactic angle
    if PAoffset!=None:
        PA+=PAoffset #rotating by some offset (for Nimmo et al. 2020 we use the weighted average of the four bursts)

    PAmask = np.ones_like(RRprofile)
    ind = np.where((Ltrue/Ioffstd>3)) #setting a sigma threshold of 3 (below which we don't plot the PPA)
    PAmask[ind] = 0
    PAnomask = PA
    PA = np.ma.masked_where(PAmask==True,PA)
    x = np.linspace(0,len(PA)+1,len(PA)) #an x array for the PPA

    #errors on PPA
    #load in an array to interpolate to find the errors for low linear S/N (following Everett and Weisberg 2001)
    #for high S/N (anything above sigma = 10) use 28.65*sigmaI/unbiased_linear (see Everett and Weisberg 2001)
    lowSN = np.load("lowSN_PAerror.npy")
    SNs = np.arange(3.0,10.0,0.01)
    SNsfunc = interp1d(SNs,lowSN)
    PAerror=[]
    weights = []
    for i in range(len(Ltrue)):
        if Ltrue[i]/Ioffstd >= 10.:
            PAerror.append(28.65*Ioffstd/Ltrue[i])
            weights.append(1./(28.65*Ioffstd/Ltrue[i]))
        if Ltrue[i]/Ioffstd>=3 and Ltrue[i]/Ioffstd < 10.:
            PAerror.append(SNsfunc(Ltrue[i]/Ioffstd))
            weights.append(1./(SNsfunc(Ltrue[i]/Ioffstd)**2))
        if Ltrue[i]/Ioffstd<3.:
            PAerror.append(0)
            weights.append(0)

    #pdf of PA (following Everett and Weisberg 2001)
    pdf = []
    for i in range(len(Ltrue)):
        pdf.append(PA_errors.PApdf(PAnomask[i]*np.pi/180., Ltrue[i], Ioffstd,thresh=3))

    pdf = np.array(pdf)
    pdf = pdf.T


    #fit a straight horizontal line to PA using the correct errors
    # to test the hypothesis that the PPA is flat across the band
    
    fit=np.polyfit(x[burstbeg:burstend],PA[burstbeg:burstend],0,w=weights[burstbeg:burstend],full=True)
    PAmean=fit[0][0]
    burstweights=np.array(weights[burstbeg:burstend])
    burstPAs = np.array(PA[burstbeg:burstend])
    dof = len(np.where(burstweights!=0)[0])-1
    print('dof', dof)
    #calculate chisq
    chisq_k = np.sum((burstPAs-PAmean)**2*burstweights)
    redchisq=chisq_k/dof
    print("chisq",chisq_k)
    print("reduced chisq",redchisq)
    print("mean offset PPA",PAmean)

    totL=np.sum(Ltrue[burstbeg:burstend])
    #errors on fractional polarisation
    sigmaLtot=np.sqrt(np.sum(np.array(sigmaLtrue[burstbeg:burstend])**2))
    sigmaItot= np.sqrt(np.sum((np.zeros_like(sigmaLtrue[burstbeg:burstend])+Ioffstd)**2))
    sigmaVtot= np.sqrt(np.sum((np.zeros_like(sigmaLtrue[burstbeg:burstend])+sigmaV)**2))

    totI=np.sum(I[burstbeg:burstend])
    sigmaLI = totL/totI*np.sqrt((sigmaLtot/totL)**2+(sigmaItot/totI)**2)

    totVabs=np.sum(np.abs(V[burstbeg:burstend]))
    totV=np.sum(V[burstbeg:burstend])
    sigmaVI = totV/totI*np.sqrt((sigmaVtot/totV)**2+(sigmaItot/totI)**2)
    print("linear frac (L/I)",totL/totI,"+-",sigmaLI)
    #print("abs V frac (V/I)",totVabs/totI)
    print("circular frac (V/I)", totV/totI,"+-",sigmaVI)

    if name!=None:
        np.save('%s_I.npy'%name,I)
        np.save('%s_V.npy'%name,V)
        np.save('%s_linear.npy'%name,Ltrue)

    if (offpulsedata!=None).all():
        return I, Ltrue, V, PA, x, PAmean, PAerror, pdf, weights, Ioffstd, sigmaV, np.std(linearoff)
    else:
        return I, Ltrue, V, PA, x, PAmean, PAerror, pdf, weights
