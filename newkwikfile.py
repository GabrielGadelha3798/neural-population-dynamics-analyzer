from klusta.kwik import KwikModel
import pandas as pd
from scaffold import Scaffold
import numpy as np

class KwikFile(Scaffold):
    """!  @brief Model for Kwik file, strongly based on KwikModel from phy project

    The main purpose of this class is provide an abstraction for kwik files provided by phy project. The current version contains a basic set of fundamental methods used in kwik file      
    @author: Nivaldo A P de Vasconcelos & Gabriel G. Gadelha
    @date: 2021.Jul.27
    """
    

    def __init__(self,path=None,name=None):
        
        
        self.kwik_model=None
        self.name = name
        self.kpath=path
        self.dat_file_path = None
        if self.kpath is not None:
            self.kwik_model=KwikModel(path)
            if (name is None):
                self.name = self.kwik_model.name
            # print ("Adding model %s" % path)
        else:
            print ("You need to define the kwikfile path (kpath).\n You can use the function set_file_path(kpath)")

        if (self.name is None) and (self.kwik_model is not None):
            self.name = self.kwik_model.name
            print(f"Defining default name to kwikfile: {self.name}")
        self.channel_positions = self.kwik_model.probe.positions

    def get_name(self):
        """! @brief Returns the found in name field in kwik file.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        return (self.name)

    def set_name(self,name):
        """! @brief Defines kwik file name.

        Author: Gabriel G. Gadelha
        Date: 2021.jul.27
        """
        self.name = name

    def set_file_path (self,kpath):
        """! @brief Defines the corresponding kwik file

        Author: Gabriel G. Gadelha
        Date: 2021.jul.27
        """
        self.kwik_model=KwikModel(kpath)
        self.name = self.kwik_model.name
        self.kpath=kpath   

    def duration(self,dat_file_path = None):
        """! @brief Returns duration of recording in seconds.

        Author: Gabriel G. Gadelha
        Date: 2021.jul.27
        """        
        if (dat_file_path is None):
            if (self.dat_file_path is None):
                spk = self.kwik_model.spike_times
                return (spk[-1])
            #Não sei exatamente a formatação dos dat_files para retornar a duração
            #else:
            #    return get_duration_from_dat_file(self.dat_file_path)

    
    def sampling_rate (self):
        """! @brief Returns the sampling rate used during the recordings 

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        return (self.kwik_model.sample_rate)
    
    def get_spike_times(self,a=None,b=None,clu_list=None,return_mode='series'):
        """! @brief Returns the spike's times on the recordings.

        Parameters: 
        clu_list: list containing group of clusters, where the spikes will be searched.
        When this input is 'None' all clusters are taken.
        a,b: interval [a,b] in which the spike times will be searched. 
        When this input is 'None' it wil use the inteval [0,duration].
        return_mode: defines if the return mode will be either a panda series ('series'), a dict('dict') or a simple list('list').

        Author: Gabriel G Gadelha
        Date: 14.jul.2021
        """
        if a is None:
            a = 0
        if b is None:
            b = self.duration()
        
        spk = self.kwik_model.spike_times
        clu = self.kwik_model.spike_clusters
        series = pd.Series(data = clu, index= spk)
        series = series.loc[a:b]
        
        if clu_list is not None:
            filter = series.isin(clu_list)
            series = series[filter]

        if return_mode is 'series':
            return series
        elif return_mode is 'dict':
            return dict(series)
        elif return_mode is 'list':
            return list(series.index)
        else:
            print('return mode not supported, please choose between "series", "dict" or "list"')

    def get_spike_clusters(self,a=None,b=None,clu_list=None, return_mode = 'list'):
        """! @brief Returns the spike's clusters on the recordings.

        Parameters: 
        clu_list: list containing group of clusters, where the clusters will be searched.
        When this input is 'None' all clusters are taken.
        a,b: interval [a,b] in which the clusters will be searched. 
        When this input is 'None' it wil use the inteval [0,duration].

        Returns list of spike clusters.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        if a is None:
            a = 0
        if b is None:
            b = self.duration()
        
        spk = self.kwik_model.spike_times
        clu = self.kwik_model.spike_clusters
        series = pd.Series(data = clu, index= spk)
        series = series.loc[a:b]
        
        if clu_list is not None:
            filter = series.isin(clu_list)
            series = series[filter]

        if return_mode is 'series':
            return series
        elif return_mode is 'dict':
            return dict(series)
        elif return_mode is 'list':
            return list(series.values)
        else:
            print('return mode not supported, please choose between "series", "dict" or "list"')

    
    def get_clusters(self):
        """! @brief Returns all clusters in the kwik file
        
        Author: Gabriel G Gadelha
        Date: 31/08/21
        """
        return list(set(self.kwik_model.spike_clusters))
    
    def get_non_noisy_clusters(self):
        clu_dict = self.groups()
        non_noisy_group_names = self.list_of_non_noisy_groups()
        non_noisy_clu = []
        # Iterate over all the items in dictionary and filter items which has even keys
        for (key, value) in clu_dict.items():
        # Check if key is even then add pair to new dictionary
            if value in non_noisy_group_names:
                non_noisy_clu.append(key)
        return non_noisy_clu

    def num_of_channels (self):
        """! @brief Returns the number of channels in the kwik file.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """        

        try:
            if (not(type(self.kwik_model) is KwikModel)):
                raise ValueError       
        except ValueError:
                print ("Exception: internal Kwik Model is not valid !!")
                return
        

        w=self.kwik_model.all_waveforms
        return w.shape[2]

    def groups(self,return_mode = 'dict'):
        """! @brief Returns a dict with cluster label and its respective group

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        if return_mode is 'series':
            return pd.Series(self.kwik_model.cluster_groups)
        if return_mode is 'dict':
            return (self.kwik_model.cluster_groups)

    def list_of_non_noisy_groups (self):

        """! @brief Returns the list of groups found in kwik file which are not called noise

        The result has a list's form.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """

        lgroups = list(self.groups().values())
        lgroups = list(set(lgroups)-set(['noise',]))
        return (lgroups)

    def list_of_groups (self):
        """! @brief Returns the list of groups found in kwik file

        The result has a list's form.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """

        lgroups = list(self.groups().values())
        lgroups = list(set(lgroups))

        return (lgroups)


    def get_spikes_count(self,a=None,b=None,clu_list=None):
        """! @brief Returns number of spikes on the recordings.

        Parameters: 
        clu_list: list containing group of clusters, where the spikes will be searched.
        When this input is 'None' all clusters are taken.
        a,b: interval [a,b] in which the spike times will be searched. 
        When this input is 'None' it wil use the inteval [0,duration].

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        spk = self.get_spike_times(a = a, b = b, clu_list=clu_list)
   
        return len(spk)

    def get_cluster_waveforms (self,cluster_id):


        try:
            if (not(type(self.kwik_model) is KwikModel)):
                raise ValueError       
        except ValueError:
                print ("Exception: internal Kwik Model is not valid !!")
                return
        
        clusters = self.kwik_model.spike_clusters
        try:
            if ((not(cluster_id in clusters))):
                raise ValueError       
        except ValueError:
                print ("Exception: cluster_id (%d) not found !! " % cluster_id)
                return

        w=self.kwik_model.all_waveforms[clusters==cluster_id]
        return w