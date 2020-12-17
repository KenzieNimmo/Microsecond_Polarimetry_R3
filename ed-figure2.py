""" Kenzie Nimmo 2020 """
""" Script to produce the extended data figure 1 from Nimmo et al. 2020"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import psrchive
import matplotlib.gridspec as gridspec
from load_archive import load_archive

mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10
mpl.rcParams['text.latex.preamble'] = r'\usepackage[T1]{fontenc}'

#load in the data
model_archive = 'gl98_1642.fits'#this archive is from the EPN database
ds_trueRM = load_archive('./pulsar/B2111+46_Merged_morezap.f',dm=141.259994506836,rm=-218.7)
ds_rmfitRM = load_archive('./pulsar/B2111+46_Merged_morezap.f',dm=141.259994506836,rm=-679)

def Stokes(ds,offbegin,offend):
    LL = ds[0,:,:]
    RR = ds[1,:,:]
    RLr = ds[2,:,:]
    RLi = ds[3,:,:]

    LLoff = ds[0,:,offbegin:offend]
    RRoff = ds[1,:,offbegin:offend]

    Ioff = LLoff + RRoff
    Ioffprof = np.sum(Ioff,axis=0)
    Ioffstd = np.std(Ioffprof)

    I = LL + RR
    V = LL - RR
    Q = 2*RLr
    U = 2*RLi

    I_prof = np.sum(I,axis=0)
    Q_prof = np.sum(Q,axis=0)
    U_prof = np.sum(U,axis=0)
    V_prof = np.sum(V,axis=0)

    peakbin = np.argmax(I_prof)
    newpeak = int(len(I_prof)/2.)
    if peakbin > newpeak:
        I_prof = np.roll(I_prof,(len(I_prof)-peakbin)+newpeak)
        V_prof = np.roll(V_prof,(len(I_prof)-peakbin)+newpeak)
        Q_prof = np.roll(Q_prof,(len(I_prof)-peakbin)+newpeak)
        U_prof = np.roll(U_prof,(len(I_prof)-peakbin)+newpeak)
    if peakbin < newpeak:
        I_prof = np.roll(I_prof,newpeak-peakbin)
        V_prof = np.roll(V_prof,newpeak-peakbin)
        Q_prof = np.roll(Q_prof,newpeak-peakbin)
        U_prof = np.roll(U_prof,newpeak-peakbin)


    linear=np.sqrt(Q_prof**2+U_prof**2)
    Lunbias = np.zeros_like(linear)
    for i in range(len(linear)):
        if linear[i]/Ioffstd > 1.57:
            Lunbias[i] = np.sqrt(((linear[i]/Ioffstd)**2-1))*Ioffstd

    PA = np.zeros_like(I_prof)
    for i in range(len(I_prof)):
        PA[i]=(0.5*np.arctan2(U_prof[i],(Q_prof[i])))
    PA*=(180./np.pi)

    PAmask = np.ones_like(I_prof)
    ind = np.where((Lunbias/Ioffstd)>3)
    PAmask[ind] = 0
    PA = np.ma.masked_where(PAmask==True,PA)

    return I_prof, Q_prof, U_prof, V_prof, linear, PA


data_trueRM_I, data_trueRM_Q, data_trueRM_U, data_trueRM_V, data_trueRM_L, PA_trueRM = Stokes(ds_trueRM,0,400)
data_rmfitRM_I, data_rmfitRM_Q, data_rmfitRM_U, data_rmfitRM_V, data_rmfitRM_L, PA_rmfitRM = Stokes(ds_rmfitRM,0,400)

#model
ds_model = load_archive('./pulsar/'+model_archive,model=True)

StokesI_model = ds_model[0,:]/np.max(ds_model[0,:])
StokesQ_model = ds_model[1,:]/np.max(ds_model[0,:])
StokesU_model = ds_model[2,:]/np.max(ds_model[0,:])
StokesV_model = ds_model[3,:]/np.max(ds_model[0,:])


peakbin = 217#np.argmax(StokesI_model)
centrebin = len(StokesI_model)/2.
shiftbin=int(-peakbin+centrebin)+3
StokesI_model = np.roll(StokesI_model,shiftbin)
stdI = np.std(StokesI_model[0:120])
StokesQ_model = np.roll(StokesQ_model,shiftbin)
StokesU_model = np.roll(StokesU_model,shiftbin)
StokesV_model = np.roll(StokesV_model,shiftbin)
Linear_model = np.sqrt(StokesQ_model**2+StokesU_model**2)

#unbias L model
Lunbias_model = np.zeros_like(Linear_model)
for i in range(len(Linear_model)):
    if Linear_model[i]/stdI >=1.57:
        Lunbias_model[i]=stdI*np.sqrt((Linear_model[i]/stdI)**2-1)

Linear_model = Lunbias_model
print("model L/I", np.sum(Linear_model[150:240])/np.sum(StokesI_model[150:240]))
print("model V/I", np.sum(np.abs(StokesV_model[150:240]))/np.sum(StokesI_model[150:240]))

PA = np.zeros_like(StokesI_model)
for i in range(len(StokesI_model)):
    PA[i]=(0.5*np.arctan2(StokesU_model[i],StokesQ_model[i]))
PA*=(180./np.pi)
PAmask = np.ones_like(StokesI_model)
ind = np.where(Linear_model > 0.1)
PAmask[ind] =0
PA = np.ma.masked_where(PAmask==True,PA)
x = np.linspace(0,len(PA)+1,len(PA))/len(PA)
PA-=np.mean(PA)
if (PA<-90).any:
    PA[np.where(PA<-90)[0]]+=180

#true RM
centre=int(len(data_trueRM_I)/2.)
meanval_true=np.mean(data_trueRM_I)
div=np.max(data_trueRM_I)
data_trueRM_I=data_trueRM_I/div
data_trueRM_Q=data_trueRM_Q/div
data_trueRM_U=data_trueRM_U/div
data_trueRM_V=data_trueRM_V/div
data_trueRM_L = data_trueRM_L/div
xtrue = np.linspace(0,len(PA_trueRM)+1,len(PA_trueRM))/len(PA_trueRM)

#rmfit RM
meanval_rmfit=np.mean(data_rmfitRM_I)
#data_rmfitRM_I-=meanval_rmfit
div = np.max(data_rmfitRM_I)
data_rmfitRM_I=data_rmfitRM_I/div
data_rmfitRM_Q=data_rmfitRM_Q/div
data_rmfitRM_U=data_rmfitRM_U/div
data_rmfitRM_V=data_rmfitRM_V/div
data_rmfitRM_L = data_rmfitRM_L/div

meanpa=np.mean(PA_rmfitRM)
PA_rmfitRM-=meanpa
PA_trueRM-=meanpa
if (PA_rmfitRM<-90).any:
    PA_rmfitRM[np.where(PA_rmfitRM<-90)[0]]+=180
if (PA_trueRM<-90).any:
    PA_trueRM[np.where(PA_trueRM<-90)[0]]+=180

fig = plt.figure(figsize=(8, 5))
rows=2
cols=2
widths = [1, 1]
heights = [1, 3]
gs = gridspec.GridSpec(ncols=cols, nrows=rows, width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)


ax1 = fig.add_subplot(gs[0,0])
a=0
b=len(x)-1
bt=len(xtrue)
ax1.scatter(x[a:b],PA[a:b], c="#8b8b8b", s=10)
ax1.scatter(xtrue[a:bt],PA_trueRM, c="k", alpha=1.0, s=10)
ax1.set_ylim(-90,90)
ax1.set_ylabel(r'${\rm PA\ (deg)}$')
ax1.set_xticklabels([])
ax1.set_yticks([-45,0,45])
plt.setp(ax1.get_xticklabels(), visible=False)
point = ax1.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax2 = fig.add_subplot(gs[0,1])
ax2.scatter(x[a:b],PA[a:b], c='#8b8b8b', s=10)
ax2.scatter(xtrue[a:bt],PA_rmfitRM, c="k", s=10)
ax2.set_ylim(-90,90)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_yticks([-45,0,45])
plt.setp(ax2.get_xticklabels(), visible=False)
point = ax2.scatter(0, 0, facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax3 = fig.add_subplot(gs[1,0],sharex=ax1)
point = ax3.scatter(0.5, 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)
ax3.plot(x[a:b],StokesI_model[a:b],'#8b8b8b')
ax3.plot(x[a:b],StokesV_model[a:b],'#9696ff')
ax3.plot(x[a:b],Linear_model[a:b],'#ff9797')
ax3.plot(xtrue[a:bt],data_trueRM_I,'k',label="I")
ax3.plot(xtrue[a:bt],data_trueRM_V,'b',label="V")
ax3.plot(xtrue[a:bt],data_trueRM_L,'r',label="L")
ax3.set_ylabel(r'${\rm Normalised\ Flux}$')
ax3.set_xticks([0.4,0.5,0.6])
ax3.set_xlim(0.3,0.7)
ax3.set_xticklabels([r'$0.4$',r'$0.5$',r'$0.6$'])
ax3.set_xlabel(r'${\rm Pulse\ phase}$')
ax3.set_yticks([0,0.5,1.0])
ax3.set_yticklabels([r'$0.0$',r'$0.5$',r'$1.0$'])
ax3.set_ylim(-0.2,1.1)
ax3.legend()

ax4 = fig.add_subplot(gs[1,1],sharex=ax2)
point = ax4.scatter(0.5, 0, facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.05, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)
ax4.plot(x[a:b],StokesI_model[a:b],'#8b8b8b')
ax4.plot(x[a:b],StokesV_model[a:b],'#9696ff')
ax4.plot(x[a:b],Linear_model[a:b],'#ff9797')
ax4.plot(xtrue[a:bt],data_rmfitRM_I,'k',label="I")
ax4.plot(xtrue[a:bt],data_rmfitRM_V,'b',label="V")
ax4.plot(xtrue[a:bt],data_rmfitRM_L,'r',label="L")
ax4.set_yticklabels([])
ax4.set_yticks([0,0.5,1.0])

ax4.set_xticks([0.4,0.5,0.6])
ax4.set_xlim(0.3,0.7)
ax4.set_ylim(-0.2,1.1)
ax4.set_xticklabels([r'$0.4$',r'$0.5$',r'$0.6$'])
ax4.set_xlabel(r'${\rm Pulse\ phase}$')
ax4.legend()
plt.savefig("Nimmo_ed-figure2.eps", format = 'eps', dpi=300)
plt.show()


bb=a+365
eb=a+615


totI=np.sum(data_rmfitRM_I[bb:eb])
totL=np.sum(np.abs(data_rmfitRM_L[bb:eb]))
totV=np.sum(np.abs(data_rmfitRM_V[bb:eb]))

print(totI)
print("L/I", totL/totI)
print("V/I", totV/totI)
