from numpy import not_equal
from klusta.kwik import KwikModel
import numpy as np
import pandas as pd

from newkwikfile import KwikFile

class DataSource():
    
    def __init__(self,path = None, name = None):
        self.class_selector ={'kwik':KwikFile}
        self.models = {}
        self.model_names = list(self.models.keys())
        self.default_model = None
        if path is None:
            print('No model has been passed. \n You can add the models using the add_model() function.')
        else:
            self.add_model(path = path, name = name)
            
    
    def __select_model(self,model_name):
        """! @brief Private function used to define which model will be used of every function on the spkpop
        Author: Gabriel G Gadelha
        Date: 31/08/2021
        """
        if model_name is None:
            model = self.default_model
        else:
            if model_name in self.model_names:
                model = self.models[model_name]
            else:
                raise ValueError("Model Name not avaliable")
        return model 

    def add_model(self,path,name = None):
        if type(path) is not str:
            raise ValueError('path must be a string')
        
        file_type = path.split('.')[-1]
        
        if file_type not in self.class_selector.keys():
            raise ValueError('Input format not found')
        
        FileModel = self.class_selector[file_type](path=path, name = name)

        name = FileModel.get_name()
        self.models[name] = FileModel
        if self.default_model is None:
            self.default_model = self.models[name]
        self.model_names = list(self.models.keys())

    def delete_model(self,model_name):
        del self.models[model_name]
        self.model_names = list(self.models.keys())

    def delete_unsorted_models(self):
        for model in self.model_names:
            groups = self.models[model].list_of_groups()
            if 'unsorted' in groups:
                self.delete_model(model_name=model)

    def inst_fr(self,model_name = None,a = None, b = None, clu_list = None,bin_size=50e-3,mean = True):
        """! @brief Returns the instant fire rate of a group of clusters.

        Parameters: 
        clu_list: list containing group of clusters, where FR will be calculated.
        When this input is 'None' all clusters are taken.
        a,b: interval [a,b] in which the spike times will be searched. 
        When this input is 'None' it wil use the inteval [0,duration].
        bin_size: interval between each value of fr.    
        mean: defines if the values will be the mean or the sum of fr for the group of clusters.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        model = self.__select_model(model_name)

        if a is None:
            a = 0
        if b is None:
            b = model.duration()
                
        if clu_list is None:
            clu_list = model.get_non_noisy_clusters()

        spikes = model.get_spike_times(a=a,b=b,clu_list=clu_list,return_mode='list')
        t=np.arange(a,b,bin_size)
        count, edges = np.histogram(spikes,bins=t)
        fr=pd.Series(count, index=edges[0:-1])
        if mean is True:
            fr = fr/len(clu_list)
        return fr

    def get_CV(self, model_name = None, a=None, b=None, clu_list=None, bin_size=50e-3, width=10, T=None):
        """! @brief Returns the CV of a group of clusters.

        Parameters: 
        clu_list: list containing group of clusters, where CV will be calculated.
        When this input is 'None' all clusters are taken.
        a,b: interval [a,b] in which the spike times will be searched. 
        When this input is 'None' it wil use the inteval [0,duration].
        bin_size: interval between each value of fr.        

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        model = self.__select_model(model_name)

        if a is None:
            a = 0
        if b is None:
            b = model.duration() 
        
        fr = self.inst_fr(model_name = model_name, a = a, b = b, clu_list= clu_list,bin_size = bin_size)
        
        if (T is None):
            T=np.arange(a,b,bin_size)
        cv=pd.Series(index=T)
        for t in T:
            samples=fr.loc[t:t+width]
            mu=samples.mean()
            sigma=samples.std()
            if mu != 0:
                cv.loc[t]=sigma/mu
        cv = cv.dropna()
        return (cv)

    def get_isi(self,model_name = None, a = None, b = None, clu_list=None):
        """! @brief returns array containing isi values for a cluster list.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        model = self.__select_model(model_name)
        isi_list = []
        if a is None:
            a = 0
        if b is None:
            b = model.duration()  

        spk = model.get_spike_times(a=a,b=b,return_mode='series')
        
        if clu_list is None:
            clu_list = model.get_non_noisy_clusters()
        print(clu_list)

        for clu in clu_list:
            clu_spk = spk[spk==clu]
            isi = np.diff(list(clu_spk.index))
            isi_list+=list(isi)

        return isi_list

    def single_unit_spikes (self, unit_id, model_name = None, a=None, b=None):
        """! @brief returns slice of spikes for a given unit_id.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        model = self.__select_model(model_name)        
        spk = model.get_spike_times(a=a,b=b,clu_list=list([unit_id]),return_mode='list')
        return spk

    def get_MUA(self,model_name = None): 
        """! @brief returns slice of spikes for the MUA group.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        model = self.__select_model(model_name)     
        groups = model.groups(return_mode = 'series')
        spk = model.get_spike_times(return_mode = 'series')
        filtered_groups = groups.isin(['mua'])
        groups = groups[filtered_groups]

        filtered_spk = spk.isin(list(groups.index))
        spk = spk[filtered_spk]
        
        return spk, list(set(groups.index))

    def get_SUA(self, model_name = None): 
        """! @brief returns slice of spikes for the SUA group.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        model = self.__select_model(model_name)     
        groups = model.groups(return_mode = 'series')
        spk = model.get_spike_times(return_mode = 'series')
        filtered_groups = groups.isin(['good'])
        groups = groups[filtered_groups]

        filtered_spk = spk.isin(list(groups.index))
        spk = spk[filtered_spk]
        
        return spk, list(set(groups.index))

    def get_SUA_clusters(self,model_name = None):
        model = self.__select_model(model_name)     
        groups = model.groups(return_mode = 'series')
        filtered_groups = groups.isin(['good'])
        groups = groups[filtered_groups]
        
        return list(set(groups.index))

    def get_MUA_clusters(self,model_name = None):
        model = self.__select_model(model_name)     
        groups = model.groups(return_mode = 'series')
        filtered_groups = groups.isin(['mua'])
        groups = groups[filtered_groups]
        
        return list(set(groups.index))        

    def isMUA(self,unit_id,model_name = None):
        """! @brief returns True/false if unit is/isn't in the MUA group.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        model = self.__select_model(model_name) 
        clu_list = model.groups()
        if clu_list[unit_id] == 'mua':
            return True
        else:
            return False
    
    def isSUA(self,unit_id,model_name = None):
        """! @brief returns True/false if unit is/isn't in the SUA group.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        model = self.__select_model(model_name) 
        clu_list = model.groups()
        if clu_list[unit_id] == 'good':
            return True
        else:
            return False

    def unit_ids(self,model_name = None):
        """! @brief Returns list of non noise clusters.

        Author: Gabriel G Gadelha
        Date: 27.jul.2021
        """
        model = self.__select_model(model_name) 
        clu_series = model.groups(return_mode = 'series')
        clu_series = clu_series[clu_series.values != 'noise']
        return clu_series

    def firing_rate_of_MUA(self,a = None, b = None, model_name = None):
        """! @brief Returns the mean fr of the MUA group.      

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        model = self.__select_model(model_name)

        if a is None:
            a = 0
        if b is None:
            b = model.duration()
            
        spk,_ = self.get_MUA(model_name = model_name)
        spk = spk.loc[a:b]
        return (len(spk)/(b - a))

    def firing_rate_of_SUA(self,a= None, b = None, model_name = None):
        """! @brief Returns the mean fr of the SUA group.       

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """        
        model = self.__select_model(model_name)

        if a is None:
            a = 0
        if b is None:
            b = model.duration()
            
        spk,_ = self.get_SUA(model_name = model_name)
        spk = spk.loc[a:b]
        return (len(spk)/(b - a))

    def group_fr(self,clu_list = None, a = None, b = None, model_name = None):
        group_fr = {}
        model = self.__select_model(model_name)
        if clu_list is None:
            clu_list = model.get_non_noisy_clusters()
        if a is None:
            a = 0
        if b is None:
            b = model.duration()
            
        for clu in clu_list:
            fr = self.cluster_fr(cluster = clu,model_name=model_name,a=a,b=b)
            group_fr[clu] = fr

        return pd.Series(group_fr)

    def cluster_fr(self,cluster,model_name = None,a = None, b = None):
        """! @brief Returns the mean fr of the SUA group.       

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """        
        model = self.__select_model(model_name)
        spk = model.get_spike_times(a=a,b=b,clu_list=[cluster],return_mode='list')
        return len(spk)/(spk[-1])
    
    def number_of_MUA(self,model_name = None):
        """! @brief Returns the number of MUA clusters in this shank.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        model = self.__select_model(model_name) 
        _, mua_list = self.get_MUA(model_name=model_name)
        return len(mua_list)

    def number_of_SUA(self,model_name = None):
        """! @brief Returns the number of SUA clusters in this shank.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        model = self.__select_model(model_name)
        _, sua_list = self.get_SUA(model_name=model_name)
        return len(sua_list)
    
    def get_cluster_contamination(self,cluster_id, model_name = None, th = None):
        """! @brief returns contamination value for a cluster.

          Author: Gabriel G Gadelha
          Date: 15.07.2021
          """
        isi = self.get_isi(model_name = model_name,clu_list=[cluster_id])
        contamination = (len(isi[isi<=1e-3]))/(len(isi))
        return contamination*100
    
    def get_group_contamination(self,cluster_group,model_name = None):
        """! @brief returns average contamination and std value for a cluster group.

          Author: Gabriel G Gadelha
          Date: 15.07.2021
          """
        group_contamination=[]
        for cluster in cluster_group:
            cluster_contamination = self.get_cluster_contamination(cluster,model_name=model_name)
            group_contamination.append(cluster_contamination)
        return np.mean(group_contamination)    


