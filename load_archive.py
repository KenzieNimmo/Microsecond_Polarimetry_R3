import psrchive
import numpy as np

def load_archive(archive_name,rm=0,dm=0,tscrunch=None,remove_baseline=True,extent=False,model=False):
    archive = psrchive.Archive_load(archive_name)
    archive.tscrunch()
    
    if remove_baseline == True:
        #remove the unpolarised background -- note this step can cause Stokes I < Linear
        archive.remove_baseline()

    if tscrunch!=None:
        archive.bscrunch(tscrunch)

    if dm!=0:
        archive.set_dispersion_measure(dm)
        archive.set_dedispersed(False)
        archive.dedisperse()

    if rm!=0:
        archive.set_rotation_measure(rm)
        archive.set_faraday_corrected(False)
        archive.defaraday()

    ds = archive.get_data().squeeze()
    w = archive.get_weights().squeeze()
    print(ds.shape)
    if model==False:
        if len(ds.shape)==3:
            for j in range(np.shape(ds)[0]):
                ds[j,:,:] = (w[:]*ds[j,:,:].T).T
        else: ds = (w*ds.T).T

        ds = np.flip(ds,axis=1)
    if model==True:
        ds = ds

    if extent==True:
        extent = [0, archive.get_first_Integration().get_duration()*1000,\
        archive.get_centre_frequency()+archive.get_bandwidth()/2.,\
        archive.get_centre_frequency()-archive.get_bandwidth()/2.]
        return ds, extent
    else:
        return ds
