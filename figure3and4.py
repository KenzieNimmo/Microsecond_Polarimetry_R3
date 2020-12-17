"""Script to plot Figures 3 and 4 in Nimmo et al. 2020"""
""" Kenzie Nimmo 2020 """

import psrchive
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import astropy
from astropy.coordinates import (EarthLocation, SkyCoord,
                                     AltAz, CIRS, Angle)
from astropy.time import Time
from astropy import units as u
from parallactic_angle import parangle as pa
import matplotlib  as mpl
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import PA_errors
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pol_prof import get_profile

from load_archive import load_archive

mpl.rcParams['font.size'] = 8
mpl.rcParams['font.family'] = 'sans-serif'                                                                                                                           
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['legend.fontsize'] = 10
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['xtick.labelsize'] = 8
mpl.rcParams['ytick.labelsize'] = 8
mpl.rcParams['xtick.major.pad']='6'
mpl.rcParams['ytick.major.pad']='6'


#the rotation measure and dispersion measure values here are determined using rmfit and pdmp respectively
#note the rotation measure is not the true rotation measure since there is a delay term that has not been corrected for
#see Nimmo et al. 2020 for more details.

#16us data
ds_b1_16us=load_archive('./16us/burst1.b4.fp.fc',rm=-536.18,dm=0.012) #downsampled in time by a factor of 4
ds_b2_16us=load_archive('./16us/burst2.b8.fp.fc',rm=-536.18,dm=0.012) #downsampled in time by a factor of 8
ds_b3_16us=load_archive('./16us/burst3.b4.fp.fc',rm=-536.18,dm=0.012) #downsampled in time by a factor of 4
ds_b4_16us=load_archive('./16us/burst4.fp.fc',rm=-536.18,dm=0.012)
#1us data
ds_b4_1us=load_archive('./1us/ep114a-fullcoherence_no0278_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012)
ds_b3_1us=load_archive('./1us/ep114a-fullcoherence_no0224_EfEf_1us_nodelay.ar.fp',rm=-536.18,dm=0.012)
#1us off burst data
ds_b3_off_1us=load_archive('./1us/ep114a-fullcoherence_no0224_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012)
ds_b4_off_1us=load_archive('./1us/ep114a-fullcoherence_no0278_EfEf_1us_nodelay_NOISE.ar.fp',rm=-536.18,dm=0.012)

#compute the parallactic angle correction
b1_parallactic=(pa(Angle('20h33m39s'),Angle('01h58m00.7502s'),Angle('65d43m0.3152s'),Angle('50d31m29.39459s')))
b2_parallactic=(pa(Angle('20h55m42s'),Angle('01h58m00.7502s'),Angle('65d43m0.3152s'),Angle('50d31m29.39459s')))
b3_parallactic=(pa(Angle('21h46m51s'),Angle('01h58m00.7502s'),Angle('65d43m0.3152s'),Angle('50d31m29.39459s')))
b4_parallactic=(pa(Angle('00h57m22s'),Angle('01h58m00.7502s'),Angle('65d43m0.3152s'),Angle('50d31m29.39459s')))

conv = 2.355 #conversion from FWHM to sigma

#burst widths like in Marcote+2020
#B1
oldpeakb1=7985
oldres1 = 16e-6*5
newres1 = 16e-6*4
tcent1 = (7982.6-oldpeakb1)*oldres1/newres1
width1 = 22.53*oldres1*1e3
#B2
oldpeakb2=598
oldres2 = 16e-6*5
newres2 = 16e-6*8
tcent2 = (597.9-oldpeakb2)*oldres2/newres2
width2 = 2.97*oldres2*1e3
#B3
oldpeakb3=8994
oldres3 = 16e-6*4
newres3 = 16e-6*4
tcent3 = (8987.6-oldpeakb3)*oldres3/newres3
width3 = 26.28*oldres3*1e3
#B3 sub-bursts
centb31 = (8981.5-8987.6)*oldres3*1e3
widthb31 = 2.08*oldres3*1e3
centb32 = (8994.2-8987.6)*oldres3*1e3
widthb32 = 1.77*oldres3*1e3
#B4
oldpeakb4=1331
oldres4 = 16e-6*1
newres4 = 16e-6*1
tcent4 = (1363.125-oldpeakb4)*oldres4/newres4
width4 = 103.75*oldres4*1e3

#B4 1us
oldpeakb41us=1331
oldres41us = 16e-6*1
newres41us = 1e-6*1
tcent41us = (1363.125-oldpeakb41us)*oldres41us/newres41us
width41us = 103.75*oldres41us*1e3

#B4 sub-bursts
centb41 = (1318.74 - 1363.125)*oldres4*1e3
widthb41 = 15*oldres4*1e3
centb42 = (1331.4 - 1363.125)*oldres4*1e3
widthb42 = 3.75*oldres4*1e3
centb43 = (1393.75 - 1363.125)*oldres4*1e3
widthb43 = 42.5*oldres4*1e3

#B3 1us
oldpeakb31us=8994
oldres31us = 16e-6*4
newres31us = 1e-6*1
tcent31us = (8987.6-oldpeakb31us)*oldres31us/newres31us
width31us = 26.28*oldres31us*1e3


I_b1, linear_b1, V_b1, PA_b1, x_b1, PA_mean_b1, PAerror_b1, pdf1, w1 = get_profile(ds_b1_16us,8000,14000,b1_parallactic,PAoffset=89.1886036595, burstbeg = 7976, burstend=8024,tcent=tcent1,name='b1_64us')
I_b2, linear_b2, V_b2, PA_b2, x_b2, PA_mean_b2, PAerror_b2, pdf2, w2  = get_profile(ds_b2_16us,2000,7000,b2_parallactic,PAoffset=89.1886036595,burstbeg=3998,burstend=4002,tcent=tcent2,name='b2_128us')
I_b3, linear_b3, V_b3, PA_b3, x_b3, PA_mean_b3, PAerror_b3, pdf3, w3  = get_profile(ds_b3_16us,0,12000,b3_parallactic,PAoffset=89.1886036595,burstbeg=7978,burstend=8022,tcent=tcent3,name='b3_64us')
I_b4, linear_b4, V_b4, PA_b4, x_b4, PA_mean_b4, PAerror_b4, pdf4, w4  = get_profile(ds_b4_16us,0,8000,b4_parallactic,PAoffset=89.1886036595,burstbeg=8104,burstend=8280,tcent=tcent4,name='b4_16us')

I_b3_1us, linear_b3_1us, V_b3_1us, PA_b3_1us, x_b3_1us, PA_mean_b3_1us, PAerror_b3_1us, pdf3_1us, w3_1us, Istd_b3_1us, Vstd_b3_1us, Lstd_b3_1us  = get_profile(ds_b3_1us,0,1500,b3_parallactic,PAoffset=89.1886036595,burstbeg=1977,burstend=2173,name='b3_1us',Lbias=True,offpulsedata=ds_b3_off_1us)

I_b4_1us, linear_b4_1us, V_b4_1us, PA_b4_1us, x_b4_1us, PA_mean_b4_1us, PAerror_b4_1us, pdf4_1us, w4_1us, Istd_b4_1us, Vstd_b4_1us, Lstd_b4_1us  = get_profile(ds_b4_1us,0,1500,b4_parallactic,PAoffset=89.1886036595,burstbeg=1977,burstend=2173,name='b4_1us',Lbias=True,offpulsedata=ds_b4_off_1us)

print("Means", [PA_mean_b1,PA_mean_b2,PA_mean_b3,PA_mean_b4])

#global fit of the PPA
xs1=np.concatenate((x_b1[7976:8024],x_b2[3998:4002]))
xs2=np.concatenate((x_b3[7978:8022],x_b4[8104:8280]))
xs = np.concatenate((xs1,xs2))
ys1=np.concatenate((PA_b1[7976:8024],PA_b2[3998:4002]))
ys2=np.concatenate((PA_b3[7978:8022],PA_b4[8104:8280]))
ys=np.concatenate((ys1,ys2))
ws1 = np.concatenate((w1[7976:8024],w2[3998:4002]))
ws2=np.concatenate((w3[7978:8022],w4[8104:8280]))
ws=np.concatenate((ws1,ws2))

fit=np.polyfit(xs,ys,0,w=ws,full=True)
globalmean=fit[0][0]
dof = len(np.where(ws!=0)[0])-1
chisq = np.sum((ys-globalmean)**2*ws)
redchisq=chisq/dof

print("Global linear fit to the PPA")
print("global",globalmean)
print("chisq_global",chisq)
print("dof",dof)
print("reduced chisq",redchisq)

#bins for plotting
b1_bin = 63
b2_bin = 31
b3_bin = 63
b4_bin = 125
b3_bin_1us = 2000
b4_bin_1us = 2000

#centre bin -- we have centred the burst in the array -- this is to match the times in Marcote et al. 2020
b1_centre=int(len(I_b1)/2.)
b2_centre=int(len(I_b2)/2.)
b3_centre=int(len(I_b3)/2.)
b4_centre=int(len(I_b4)/2.)
b3_centre_1us=int(len(I_b3_1us)/2.)
b4_centre_1us=int(len(I_b4_1us)/2.)

width_1=22.53
width_2=2.97
width_3=26.28
width_4=103.75
width_3_1us=width_3*16
width_4_1us = width_4*16

x_b1-=(x_b1[b1_centre])
x_b1*=4*16e-3
x_b2-=(x_b2[b2_centre])
x_b2*=8*16e-3
x_b3-=(x_b3[b3_centre])
x_b3*=4*16e-3
x_b4-=(x_b4[b4_centre])
x_b4*=16e-3
x_b3_1us-=(x_b3_1us[b3_centre_1us])
#x_b3_1us*=1e-3
x_b4_1us-=(x_b4_1us[b4_centre_1us])
#x_b4_1us*=1e-3 #commented out so that the units of the 1us plot are microseconds

#Now make the figure
#Figure 3 in Nimmo et al. 2020
fig = plt.figure(figsize=(8, 8))
rows=2
cols=2
widths = [1, 1]
heights = [1, 3]
gs = gridspec.GridSpec(ncols=cols, nrows=rows, top=0.96, bottom=0.58, width_ratios=widths, height_ratios=heights, wspace=0.1, hspace=0.0)
gs2 = gridspec.GridSpec(ncols=cols, nrows=rows, top=0.52, bottom=0.08, width_ratios=widths, height_ratios=heights, wspace=0.1, hspace=0.0)

ax1 = fig.add_subplot(gs[0,0]) #PA B1
ax1.plot(x_b1[b1_centre-b1_bin:b1_centre+b1_bin], np.zeros(len(x_b1[b1_centre-b1_bin:b1_centre+b1_bin])),'g')
cmap = plt.cm.gist_yarg
vmin=np.min(pdf1[:,b1_centre-b1_bin:b1_centre+b1_bin])
vmax=np.max(pdf1[:,b1_centre-b1_bin:b1_centre+b1_bin])
ax1.imshow(pdf1[:,b1_centre-b1_bin:b1_centre+b1_bin],cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b1[b1_centre-b1_bin],x_b1[b1_centre+b1_bin],-90,90])
ax1.set_ylim(-20,20)
plt.setp(ax1.get_xticklabels(), visible=False)
ax1.set_yticks([-10,0,10])
point = ax1.scatter(x_b1[0], 0, facecolors='none', edgecolors='none')
plotlabel=ax1.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax2 = fig.add_subplot(gs[1,0],sharex=ax1) #profile B1
max1=np.max(I_b1[b1_centre-b1_bin:b1_centre+b1_bin])
ax2.step(x_b1[b1_centre-b1_bin:b1_centre+b1_bin],I_b1[b1_centre-b1_bin:b1_centre+b1_bin]/max1,'k',alpha=1.0,where='post')
ax2.step(x_b1[b1_centre-b1_bin:b1_centre+b1_bin],V_b1[b1_centre-b1_bin:b1_centre+b1_bin]/max1,'b',alpha=1.0,where='post')
ax2.step(x_b1[b1_centre-b1_bin:b1_centre+b1_bin],linear_b1[b1_centre-b1_bin:b1_centre+b1_bin]/max1,'r',alpha=1.0,where='post')
#ax2.set_yticklabels([])
ax2.set_xlim(-4,4)
ax2.set_ylabel('Normalised Flux')
ax2.set_yticks([0,1])
ax2.set_ylim(-0.45,1.1)
ax2.hlines(y=-0.4,xmin=(-width1*2./conv),xmax=(width1*2./conv), lw=6,color='#d6ffff',zorder=0.8)
ax2.hlines(y=-0.4, xmin=(-width1/2.),xmax=(width1/2.), lw=6, color='#48D1CC',zorder=0.9)
point = ax2.scatter(x_b1[0], I_b1[0], facecolors='none', edgecolors='none')
plotlabel=ax2.legend((point,point), ('B1',u'64\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
plotlabel=ax2.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


ax3 = fig.add_subplot(gs2[0,0]) #PA B3
ax3.set_ylim(-20,20)
vmin=np.min(pdf3[:,b3_centre-b3_bin:b3_centre+b3_bin])
vmax=np.max(pdf3[:,b3_centre-b3_bin:b3_centre+b3_bin])
ax3.imshow(pdf3[:,b3_centre-b3_bin:b3_centre+b3_bin],cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b3[b3_centre-b3_bin],x_b3[b3_centre+b3_bin],-90,90])
ax3.plot(x_b3[b3_centre-b3_bin:b3_centre+b3_bin], np.zeros(len(x_b3[b3_centre-b3_bin:b3_centre+b3_bin])),'g')
ax3.set_yticks([-10,0,10])
plt.setp(ax3.get_xticklabels(), visible=False)
point = ax3.scatter(x_b3[0], 0, facecolors='none', edgecolors='none')
plotlabel=ax3.legend((point,point), ('e',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax4 = fig.add_subplot(gs2[1,0],sharex=ax3) #profile B3
max3=np.max(I_b3[b3_centre-b3_bin:b3_centre+b3_bin])
ax4.axvspan(x_b3[b3_centre-b3_bin+57],x_b3[b3_centre-b3_bin+58],color='#b2b2b2',zorder=0.1)
ax4.step(x_b3[b3_centre-b3_bin:b3_centre+b3_bin],I_b3[b3_centre-b3_bin:b3_centre+b3_bin]/max3,'k',alpha=1.0,where='post')
ax4.step(x_b3[b3_centre-b3_bin:b3_centre+b3_bin],V_b3[b3_centre-b3_bin:b3_centre+b3_bin]/max3,'b',alpha=1.0,where='post')
ax4.step(x_b3[b3_centre-b3_bin:b3_centre+b3_bin],linear_b3[b3_centre-b3_bin:b3_centre+b3_bin]/max3,'r',alpha=1.0,where='post')
ax4.set_xlim(-4,4)
ax4.set_ylim(-0.45,1.1)
ax4.set_yticks([0,1])
ax4.hlines(y=-0.4,xmin=(-width3*2./conv),xmax=(width3*2./conv), lw=6,color='#d6ffff',zorder=0.8)
ax4.hlines(y=-0.4, xmin=(-width3/2.),xmax=(width3/2.), lw=6, color='#48D1CC',zorder=0.9)
ax4.set_ylabel('Normalised Flux')
point = ax4.scatter(x_b3[0], I_b3[0], facecolors='none', edgecolors='none')
plotlabel=ax4.legend((point,point), ('B3',u'64\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax4.hlines(y=-0.35, xmin=(((centb31-(widthb31/2.)))),xmax=((centb31+(widthb31/2.))), lw=7, color='#FFA500',zorder=0.7,alpha=1.0)
ax4.hlines(y=-0.35, xmin=(centb31-(widthb31*2./conv)),xmax=((centb31+(widthb31*2./conv))), lw=7, color='#ffc251',zorder=0.6)
ax4.hlines(y=-0.35, xmin=(((centb32-(widthb32/2.)))),xmax=((centb32+(widthb32/2.))), lw=7, color='#8D32C1',zorder=0.7)
ax4.hlines(y=-0.35, xmin=(centb32-(widthb32*2./conv)),xmax=((centb32+(widthb32*2./conv))), lw=7, color='#ae76b4',zorder=0.6)
plotlabel=ax4.legend((point,point), ('f',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


ax5 = fig.add_subplot(gs[0,1]) #PA B2
ax5.set_yticks([-10,0,10])
vmin=np.min(pdf2[:,b2_centre-b2_bin:b2_centre+b2_bin])
vmax=np.max(pdf2[:,b2_centre-b2_bin:b2_centre+b2_bin])
ax5.imshow(pdf2[:,b2_centre-b2_bin:b2_centre+b2_bin],cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b2[b2_centre-b2_bin]-0.128*0.5,x_b2[b2_centre+b2_bin]-0.128*0.5,-90,90])
ax5.plot(x_b2[b2_centre-b2_bin:b2_centre+b2_bin], np.zeros(len(x_b2[b2_centre-b2_bin:b2_centre+b2_bin])),'g')
plt.setp(ax5.get_xticklabels(), visible=False)
ax5.set_ylim(-20,20)
point = ax5.scatter(x_b2[0], 0, facecolors='none', edgecolors='none')
plotlabel=ax5.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


ax6 = fig.add_subplot(gs[1,1],sharex=ax5) #profile B2
max2=np.max(I_b2[b2_centre-b2_bin:b2_centre+b2_bin])
ax6.step(x_b2[b2_centre-b2_bin:b2_centre+b2_bin],I_b2[b2_centre-b2_bin:b2_centre+b2_bin]/max2,'k',alpha=1.0,where='mid')
ax6.step(x_b2[b2_centre-b2_bin:b2_centre+b2_bin],V_b2[b2_centre-b2_bin:b2_centre+b2_bin]/max2,'b',alpha=1.0,where='mid')
ax6.step(x_b2[b2_centre-b2_bin:b2_centre+b2_bin],linear_b2[b2_centre-b2_bin:b2_centre+b2_bin]/max2,'r',alpha=1.0,where='mid')
ax6.set_xlim(-4,4)
ax6.set_ylim(-0.45,1.1)
ax6.set_yticks([0,1])
ax6.hlines(y=-0.4,xmin=(-width2*2./conv),xmax=(width2*2./conv), lw=6,color='#d6ffff',zorder=0.8)
ax6.hlines(y=-0.4, xmin=(-width2/2.),xmax=(width2/2.), lw=6, color='#48D1CC',zorder=0.9)
point = ax6.scatter(x_b3[0], I_b3[0], facecolors='none', edgecolors='none')
plotlabel=ax6.legend((point,point), ('B2',u'128\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
plotlabel=ax6.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax7 = fig.add_subplot(gs2[0,1]) #PA B4
ax7.set_ylim(-20,20)
ax7.set_yticks([-10,0,10])
plt.setp(ax7.get_xticklabels(), visible=False)
vmin=np.min(pdf4[:,b4_centre-b4_bin:b4_centre+b4_bin])
vmax=np.max(pdf4[:,b4_centre-b4_bin:b4_centre+b4_bin])
ax7.imshow(pdf4[:,b4_centre-b4_bin:b4_centre+b4_bin],cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b4[b4_centre-b4_bin],x_b4[b4_centre+b4_bin],-90,90])
ax7.plot(x_b4[b4_centre-b4_bin:b4_centre+b4_bin], np.zeros(len(x_b4[b4_centre-b4_bin:b4_centre+b4_bin])),'g')#np.mean(PA_b4[b4_centre-b4_bin:b4_centre+b4_bin]))
point = ax7.scatter(x_b4[0], 0, facecolors='none', edgecolors='none')
plotlabel=ax7.legend((point,point), ('g',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax8 = fig.add_subplot(gs2[1,1],sharex=ax7) #profile B4
max4=np.max(I_b4[b4_centre-b4_bin:b4_centre+b4_bin])
peakbin=np.argmax(I_b4)
ax8.axvspan(x_b4[peakbin-4],x_b4[peakbin+4],color='#b2b2b2',zorder=0.1)
ax8.step(x_b4[b4_centre-b4_bin:b4_centre+b4_bin],I_b4[b4_centre-b4_bin:b4_centre+b4_bin]/max4,'k',alpha=1.0,where='post')
ax8.step(x_b4[b4_centre-b4_bin:b4_centre+b4_bin],V_b4[b4_centre-b4_bin:b4_centre+b4_bin]/max4,'b',alpha=1.0,where='post')
ax8.step(x_b4[b4_centre-b4_bin:b4_centre+b4_bin],linear_b4[b4_centre-b4_bin:b4_centre+b4_bin]/max4,'r',alpha=1.0,where='post')
ax8.arrow(-0.860269,0.325771,0,-0.07,color='limegreen',head_width=0.1,head_length=0.05)
ax8.set_xlim(-2,2)
ax8.set_ylim(-0.2,1.1)
ax8.set_yticks([0,1])
ax8.hlines(y=-0.15,xmin=(-width4*2./conv),xmax=(width4*2./conv), lw=6,color='#d6ffff',zorder=0.8)
ax8.hlines(y=-0.15, xmin=(-width4/2.),xmax=(width4/2.), lw=6, color='#48D1CC',zorder=0.9)
point = ax8.scatter(x_b4[0], I_b4[0], facecolors='none', edgecolors='none')
plotlabel=ax8.legend((point,point), ('B4',u'16\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.02, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
ax8.hlines(y=-0.1, xmin=(((centb41-(widthb41/2.)))),xmax=((centb41+(widthb41/2.))), lw=7, color='#FFA500',zorder=0.7,alpha=1.0)
ax8.hlines(y=-0.1, xmin=(centb41-(widthb41*2./conv)),xmax=((centb41+(widthb41*2./conv))), lw=7, color='#ffc251',zorder=0.6)
ax8.hlines(y=-0.1, xmin=(((centb42-(widthb42/2.)))),xmax=((centb42+(widthb42/2.))), lw=7, color='#8D32C1',zorder=0.7)
ax8.hlines(y=-0.1, xmin=(centb42-(widthb42*2./conv)),xmax=((centb42+(widthb42*2./conv))), lw=7, color='#ae76b4',zorder=0.6)
ax8.hlines(y=-0.1, xmin=(((centb43-(widthb43/2.)))),xmax=((centb43+(widthb43/2.))), lw=7, color='#008000',zorder=0.7)
ax8.hlines(y=-0.1, xmin=(centb43-(widthb43*2./conv)),xmax=((centb43+(widthb43*2./conv))), lw=7, color='#4db056',zorder=0.6)
ax8.axvline((centb41-(widthb41*2./conv)),linestyle='--',color='k')
ax8.axvline((centb42-(widthb42*2./conv)),linestyle='--',color='k')
ax8.axvline(((centb42+(widthb42*2./conv))),linestyle='--',color='k')
ax8.axvline((centb43-(widthb43*2./conv)),linestyle='--',color='k')
ax8.axvline(((centb43+(widthb43*2./conv))),linestyle='--',color='k')
plotlabel=ax8.legend((point,point), ('h',), loc='upper left',handlelength=0,bbox_to_anchor=(0.9, 0.35), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

axins = zoomed_inset_axes(ax8, 2.5, loc=1)
axins.step(x_b4[b4_centre-b4_bin:b4_centre+b4_bin],I_b4[b4_centre-b4_bin:b4_centre+b4_bin]/max4, 'k',where='post')
axins.step(x_b4[b4_centre-b4_bin:b4_centre+b4_bin],linear_b4[b4_centre-b4_bin:b4_centre+b4_bin]/max4, 'r',where='post')
axins.step(x_b4[b4_centre-b4_bin:b4_centre+b4_bin],V_b4[b4_centre-b4_bin:b4_centre+b4_bin]/max4, 'b',where='post')
x1, x2, y1, y2 = -1,-0.45,-0.05,0.25
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.yticks(visible=False)
plt.xticks(visible=False)

ax8.set_xlabel('Time [ms]')
ax4.set_xlabel('Time [ms]')
ax1.set_ylabel('PPA [deg]')
ax3.set_ylabel('PPA [deg]')

plt.savefig("Nimmo_fig4.eps",  format='eps', dpi=300)
plt.show()

#high time resolution polarisation plot
#Figure 4 in Nimmo et al. 2020
fig = plt.figure(figsize=(8, 8))
rows=2
cols=1
widths = [1]
heights = [1, 3]
gs3 = gridspec.GridSpec(ncols=cols, nrows=rows, bottom=0.58,top=0.96,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)
gs4 = gridspec.GridSpec(ncols=cols, nrows=rows, bottom=0.08,top=0.52,width_ratios=widths, height_ratios=heights, wspace=0.0, hspace=0.0)

ax9 = fig.add_subplot(gs4[0,0]) #PA B4 high time res
ax9.set_ylim(-20,20)
ax9.set_yticks([-10,0,10])
plt.setp(ax9.get_xticklabels(), visible=False)
vmin=np.min(pdf4_1us[:,b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us])
vmax=np.max(pdf4_1us[:,b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us])
ax9.imshow(pdf4_1us[:,b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us],cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b4_1us[b4_centre_1us-b4_bin_1us],x_b4_1us[b4_centre_1us+b4_bin_1us],-90,90])
ax9.plot(x_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us], np.zeros(len(x_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us])),'g')
ax9.set_ylabel('PPA [deg]')
point = ax9.scatter(x_b4_1us[0], 0, facecolors='none', edgecolors='none')
plotlabel=ax9.legend((point,point), ('c',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax10 = fig.add_subplot(gs4[1,0],sharex=ax9) #Profile B4 high time res
max4=np.max(I_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us])
ax10.step(x_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us],I_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us]/max4,'k',alpha=1.0,where='post')
ax10.step(x_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us],V_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us]/max4,'b',alpha=1.0,where='post')
ax10.step(x_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us],linear_b4_1us[b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us]/max4,'r',alpha=1.0,where='post')
Ioffstd=Istd_b4_1us/max4
Voffstd=Vstd_b4_1us/max4
Loffstd=Lstd_b4_1us/max4
ax10.errorbar(0.045e3,0.6,yerr=Ioffstd,ecolor='k')
ax10.errorbar(0.05e3,0.6,yerr=Voffstd,ecolor='b')
ax10.errorbar(0.055e3,0.6,yerr=Loffstd,ecolor='r')
ax10.set_xlim(-0.06e3,0.06e3)
ax10.set_ylim(-0.2,1.2)
ax10.set_ylabel('Normalised Flux')
ax10.set_xlabel(u'Time [\u03bcs]')
ax10.set_yticks([0,1])
point = ax10.scatter(x_b4_1us[0], I_b4_1us[0], facecolors='none', edgecolors='none')
plotlabel=ax10.legend((point,point), ('B4-sb2',u'1\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.06, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
plotlabel=ax10.legend((point,point), ('d',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

ax11 = fig.add_subplot(gs3[0,0]) #PA B4 high time res
ax11.set_ylim(-20,20)
ax11.set_yticks([-10,0,10])
plt.setp(ax11.get_xticklabels(), visible=False)
vmin=np.min(pdf3_1us[:,b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us])
vmax=np.max(pdf3_1us[:,b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us])
ax11.imshow(pdf3_1us[:,b4_centre_1us-b4_bin_1us:b4_centre_1us+b4_bin_1us],cmap=cmap,origin='lower', aspect='auto', interpolation='nearest', extent=[x_b3_1us[b3_centre_1us-b3_bin_1us],x_b3_1us[b3_centre_1us+b3_bin_1us],-90,90])
ax11.plot(x_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us], np.zeros(len(x_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us])),'g')
ax11.set_ylabel('PPA [deg]')
point = ax11.scatter(x_b3_1us[0], 0, facecolors='none', edgecolors='none')
plotlabel=ax11.legend((point,point), ('a',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)


ax12 = fig.add_subplot(gs3[1,0],sharex=ax11) #Profile B4 high time res
max3=np.max(I_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us])
ax12.step(x_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us],I_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us]/max3,'k',alpha=1.0,where='post')
ax12.step(x_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us],V_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us]/max3,'b',alpha=1.0,where='post')
ax12.step(x_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us],linear_b3_1us[b3_centre_1us-b3_bin_1us:b3_centre_1us+b3_bin_1us]/max3,'r',alpha=1.0,where='post')
Ioffstd=Istd_b3_1us/max3
Voffstd=Vstd_b3_1us/max3
Loffstd=Lstd_b3_1us/max3
ax12.errorbar(0.025e3,0.7,yerr=Ioffstd,ecolor='k')
ax12.errorbar(0.027e3,0.7,yerr=Voffstd,ecolor='b')
ax12.errorbar(0.029e3,0.7,yerr=Loffstd,ecolor='r')
ax12.set_xlim(-0.02e3,0.03e3)
ax12.set_ylim(-0.3,1.2)
ax12.set_ylabel('Normalised Flux')
#ax12.set_xlabel(u'Time [\u03bcs]')
ax12.set_yticks([0,1])
point = ax12.scatter(x_b3_1us[0], I_b3_1us[0], facecolors='none', edgecolors='none')
plotlabel=ax12.legend((point,point), ('B3',u'1\u03bcs'), loc='upper left',handlelength=0,bbox_to_anchor=(0.06, 0.95), handletextpad=-0.5,frameon=False,markerscale=0)
plt.gca().add_artist(plotlabel)
plotlabel=ax12.legend((point,point), ('b',), loc='upper left',handlelength=0,bbox_to_anchor=(0.93, 0.95), handletextpad=-0.5,frameon=False,markerscale=0,fontsize=10)
plt.gca().add_artist(plotlabel)

plt.savefig("Nimmo_fig5.eps",  format='eps', dpi=300)
plt.show()

exit()
