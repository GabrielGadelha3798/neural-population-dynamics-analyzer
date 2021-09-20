from klusta.kwik import KwikModel
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


class KwikFile:
    """!  @brief Model for Kwik file, strongly based on KwikModel from phy project

    The main purpose of this class is provide an abstraction for kwik files provided by phy project. The current version contains a basic set of fundamental methods used in kwik file      
    @author: Nivaldo A P de Vasconcelos
    @date: 2018.Feb.02
    """
    
    #get_path
    def __init__(self,kpath=None,name=None):
        
        
        self.kwik_model=None
        self.name = name
        self.kpath=None
        if (kpath is not None):
            self.kwik_model=KwikModel(kpath)
            self.kpath=kpath
            if (name is None):
                self.name = self.kwik_model.name
            print ("Created class on = %s !" % kpath)
        else:
            print ("It still with no path:(")
    


    def get_name(self):
        """! @brief Returns the found in name field in kwik file.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        return (self.name)

    def set_kwik_file (self,kpath):
        """! @brief Defines the corresponding kwik file

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        self.kwik_model=KwikModel(kpath)
        self.name = self.kwik_model.name
        self.kpath=kpath


    def sampling_rate (self):
        """! @brief Returns the sampling rate used during the recordings 

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        return (self.kwik_model.sample_rate)
    
    def shank (self):
        """! @brief Returns the shank/population's id used to group the recordings.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        return (self.kwik_model.name)
    
    def get_spike_times(self):
        """! @brief Returns the spike's times on the recordings.

        Author: Gabriel G Gadelha
        Date: 14.07.2021
        """
        return (self.kwik_model.spike_times)
    
    def get_spike_samples (self):
        """! @brief Returns the spike's samples on the recordings.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """

        return (self.kwik_model.spike_samples)
    
    def get_spike_clusters (self):
        """! @brief Returns the corresponding spike's clusters on the recordings.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        return (self.kwik_model.spike_clusters)
    
    def get_duration(self):
        """! @brief Returns the duration of the recording in seconds.

        Author: Gabriel G Gadelha
        Date: 14.07.2021
        """
        spk = self.kwik_model.spike_times
        return (spk[-1])
    
    def get_spikes_count(self):
        """! @brief Returns the total spike count.

        Author: Gabriel G Gadelha
        Date: 14.07.2021
        """
        spk = self.kwik_model.spike_times
        return len(spk)

    def describe(self):
        """! @brief Describes the kwik file

        It calls the describe method in KwikModel

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        self.kwik_model.describe()

    def close (self):
        """! @brief Closes the corresponding kwik model

        It calls the close method in KwikModel

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        self.kwik_model.close()

    def groups(self):
        """! @brief Returns a dict with cluster label and its respective group

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        if not(isinstance(self.kwik_model,KwikModel)):
            raise ValueError("There is no KwikModel assigned for this object.")
        return (self.kwik_model.cluster_groups)

    def list_of_groups (self):
        """! @brief Returns the list of groups found in kwik file

        The result has a list's form.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """

        lgroups = list(self.groups().values())
        lgroups = list(set(lgroups))

        return (lgroups)

    def list_of_non_noisy_groups (self):

        """! @brief Returns the list of groups found in kwik file which are not called noise

        The result has a list's form.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """

        lgroups = list(self.groups().values())
        lgroups = list(set(lgroups)-set(['noise',]))
        return (lgroups)


    def all_clusters (self):
        """! @brief Returns the list of all clusters in kwik file

        The result has a list's form.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """

        llabels = list(self.groups().keys())
        llabels = list(set(llabels))

        return (llabels)


    def clusters (self,group_name=None):
        """! @brief Returns the list of clusters on kwik file

        It can be used to get the list of clusters for a given group by pproviding
        this information the group_name.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02
        """
        if (group_name is None):
            return (self.all_clusters())
 
        if not(group_name in self.list_of_groups()):
            raise ValueError("\nThis group was not found in kwik file: %s\n" % group_name)
        group=self.groups()

        clusters=[]
        for c in self.all_clusters():
            if (group[c]==group_name):
                clusters.append(c)
        clusters.sort()
        return (clusters)

    def non_noise_clusters(self):
        """! @brief Returns the total spike count.

        Author: Gabriel G Gadelha
        Date: 14.07.2021
        """
        non_noise_clusters = []
        non_noise_groups = self.list_of_non_noisy_groups()
        for group in non_noise_groups:
            clusters = self.clusters(group)
            for cluster in clusters:
                non_noise_clusters.append(cluster)
        non_noise_clusters.sort()
        return non_noise_clusters

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



    def plot_cluster_waveforms (self,cluster_id,nspikes, save=False,save_path=None, wave_color="gray"):
    
    
        w = self.get_cluster_waveforms (cluster_id)
        model=self.kwik_model
        y_scale = .7
        x_scale = 4
        num_channels = w.shape[2]
        waveform_size = w.shape[1]
        np.random.seed(int(time.time()))
        
        fig=plt.figure(num=None, figsize=(6, 18), dpi=50, facecolor='w', edgecolor='k')
        plt.clf()
        spike_id = np.arange(w.shape[0])
        np.random.shuffle(spike_id)
        spike_id = spike_id[0:nspikes]
        posx=np.flipud (model.channel_positions [:,0])
        posy=np.flipud (model.channel_positions [:,1])
        for ch in range (0,num_channels):
            x_offset = model.channel_positions [ch,0]
            y_offset =posy[ch]*y_scale
            #y_offset = model.channel_positions [ch,1]*y_scale
            mu_spikes = np.mean(w[:,:,ch],0)
            for i in spike_id:
                spike = w[i,:,ch]
                x=x_scale*x_offset+range(0,waveform_size)
                plt.plot (x,0.05*spike+y_offset,color=wave_color,alpha=0.5)
            plt.plot (x,0.05*mu_spikes+y_offset,"--",color="black",linewidth=3,alpha=0.3)
        plt.tight_layout()
        plt.axis("off")
        plt.show()

        if (save):
            if (save_path):
                filename = "%s/%s_waveform_%02d.pdf" % (save_path,self.kwik_model.name,cluster_id)
            else:
                filename = "waveform_%02d.pdf" % cluster_id
            fig.savefig (filename)
            
    def all_spikes_on_groups (self,group_names):
        """! @bri\ef Returns the all spike samples within a list of groups

        Usually the clusters are organized in groups. Ex: noise, mua, sua,
        unsorted This method returns, in a single list of spike samples, all
        spikes found in a lists of groups (group_names). 

        Parameters:
        group_names: list of group names, where the spikes will be searched. 

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02 
        """

        spikes=[]
        all_spikes=self.get_spike_samples()
        all_labels=self.get_spike_clusters()
        
        if not(isinstance(group_names,list)):
            raise ValueError("\nThe argument must be a list.") 
        
        for group_name in group_names:
            if not(group_name in self.list_of_groups()):
                raise ValueError("\nThis group was not found in kwik file: %s\n" % group_name)
            for c in self.clusters(group_name=group_name):
                spikes=spikes+list(all_spikes[all_labels==c])
        spikes.sort()
        return (spikes)

    def shank_event_rate (self,noise = False):
        """! @brief returns firing rate in a chunk.

        noise variable defines if noise group is or isn't included

        Author: Gabriel G Gadelha
        Date: 14.07.2021
        """
        if not noise:
            cluster_groups = self.list_of_non_noisy_groups()
        else:
            cluster_groups = self.list_of_groups()
        spk = self.all_spikes_on_groups(cluster_groups)
        fr = len(spk)/(self.get_duration())
        return fr

    def group_event_rate(self,group_name,return_type = 'clusters'):
        """! @brief returns event rate in a group.

        Parameters:
        group_name: string with name of group to be analised.
        return type: 'clusters' returns event rate for each cluster in a list.
                     'average' returns average  and std from event rate from entire group.

        Author: Gabriel G Gadelha
        Date: 16.07.2021
        """
        group_er = []
        clusters = self.clusters(group_name)
        for cluster in clusters:
            fr = self.cluster_firing_rate(cluster)
            group_er.append(fr)
        if return_type == 'clusters':
            return group_er       
        elif return_type == 'average':
            return np.mean(group_er),np.std(group_er)
        else:
            print(f'return_type deve ser "clusters" ou "average". {return_type} não é um formato aceitado')
 
    def group_firing_rate (self,group_names=None,a=None,b=None): 
        """! @brief Returns firing rate in a given set of groups found in kwik file.

        Usually, the clusters are organized in groups. Ex: noise, mua, sua,
        unsorted. This method returns, in a doubled dictionary, the firing rate
        for each cluster, organized by groups.

        Parameters: 
        group_names: list of group names, where the spikes will be
        searched. When this input is 'None' all groups are taken. The resulting
        dictionary has the first keys as groups, and the second keys as the
        respective cluster id's, whereas the value, is the corresponding firing
        rate within [a,b].


        Please refer to the method cluster_firing_rate in order to get more 
        details about the firing calculation.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02 
        """
        if not(isinstance(group_names,list)) and not(group_names is None):
            raise ValueError("\nThe argument must be a list or a None.") 
        spk=dict()
        if group_names is None:
            group_names=self.list_of_non_noisy_groups()
        for group_name in group_names:
            if not(group_name in self.list_of_groups()):
                raise ValueError("\nThis group was not found in kwik file: %s\n" % group_name)
            spk[group_name]=dict()
            for c in self.clusters(group_name=group_name):
                spk[group_name][c]=self.cluster_firing_rate(c,a=a,b=b)
        return (spk)


    def cluster_firing_rate (self,cluster_id,a=None,b=None):
        """! @brief Returns firing rate in a given cluster_id found in the kwik file

        In the kwik file, a cluster stores the spike times sorted for a given neuronal
        unit. The firing rate here is calculated by dividing the number of spike times
        by the number of seconds of the time period definedd by [a,b]. 
        If a is 'None' a is assingned to zero; if b is 'None', it is assigned to the time
        of the last spike within the cluster.

        Parameters:
        cluster_id: id which identifies the cluster.
        a,b: limits of the time period where the firing rate must be calculated.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02 
        """
        #sr=self.sampling_rate()
        spikes=np.array(self.spikes_on_cluster (cluster_id))
        if a is None:
            a=0
        if b is None:
            b=spikes[-1]
        if (a==b):
            raise ValueError ("\nThe limits of the time interval are equal\n")
        piece=spikes[(spikes>=a)]
        piece=piece[piece<=b]
        return (len(piece)/(b-a))

    def group_firing_rate_to_dataframe (self,group_names=None,a=None,b=None):
        """! @brief Exports the group's firing rate into a pandas dataframe


        Usually, the clusters are organized in groups. Ex: noise, mua, sua,
        unsorted. This method returns, in a pandas dataframe, which contains the
        following information for each unit: 'shank', 'group', 'label', and 'fr';

        
        Parameters:
        group_names: list of group names, where the spikes will be
        searched. When this input is 'None' all groups are taken. The resulting
        dictionary has the first keys as groups, and the second keys as the
        respective cluster id's, whereas the value, is the corresponding firing
        rate within [a,b].

        Please refer to the method cluster_firing_rate in order to get more 
        details about the firing calculation.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02 
        """
        d=self.group_firing_rate (group_names=group_names,a=a,b=b)

        shank_id = self.name
        group_names = d.keys()
        data=[]
        for group_name in group_names:
            for label in d[group_name].keys():        
                fr=d[group_name][label]
                data.append ({"shank_id":shank_id, "group":group_name,"label":label,"fr":fr})
        return (pd.DataFrame(data))

    def spikes_on_cluster (self,cluster_id,return_type = 'times'):
        """! @brief Returns the all spike samples within a single cluster

        Parameters:
        cluster_id: id used to indentify the cluster.
        return_type:  times > return spike times
                      samples > return spike samples
        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Feb.02 
        """

        if not(cluster_id in self.all_clusters()):
            raise ValueError("\nThis cluster was not found in kwik file: %s\n" % cluster_id)
        if return_type == 'times':
            all_spikes = self.get_spike_times()
        elif return_type == 'samples':
            all_spikes=self.get_spike_samples()
        else:
            raise ValueError("\nThis return type is not supported: %s\n" % return_type)
        all_labels=self.get_spike_clusters()
        spikes=list(all_spikes[all_labels==cluster_id])
        spikes.sort()
        return (spikes)

    def get_cluster_isi(self,cluster_id):
        """! @brief returns array containing isi values for a cluster.

        Author: Gabriel G Gadelha
        Date: 15.07.2021
        """
        spk = self.spikes_on_cluster(cluster_id)
        isi = np.diff(spk)
        return isi

    def get_cluster_contamination(self,cluster_id):
        """! @brief returns contamination value for a cluster.

          Author: Gabriel G Gadelha
          Date: 15.07.2021
          """
        isi = self.get_cluster_isi(cluster_id)
        contamination = (len(isi[isi<=1e-3]))/(len(isi))
        return contamination
    
    def get_group_contamination(self,cluster_group):
        """! @brief returns average contamination and std value for a cluster group.

          Author: Gabriel G Gadelha
          Date: 15.07.2021
          """
        group_contamination=[]
        for cluster in cluster_group:
            cluster_contamination = self.get_cluster_contamination(cluster)
            group_contamination.append(cluster_contamination)
        return np.mean(group_contamination)
    
    def get_spikes_on_groups(self, group_list):
        clusters = []
        spikes = []
        for group in group_list:
            clusters.extend(self.clusters(group))
        for cluster in clusters:
            spikes.extend(self.spikes_on_cluster(cluster))
        return spikes
        
        
    def get_group_FR_plot(self,spikes,bin_size=50e-3):
        duration = self.get_duration()
        t=np.arange(0,duration,bin_size)
        count, edges = np.histogram(spikes,bins=t)
        fr=pd.Series(count, index=edges[0:-1])
        return fr
        
        
    def get_cv(self,spikes,bin_size=50e-3,window_size=10,T = None):
        duration = self.get_duration()
        fr = self.get_group_FR_plot(spikes,bin_size = bin_size)
        if (T is None):
            T=np.arange(0,duration,window_size)
        cv=pd.Series(index=T)
        for t in T:
            samples=fr.loc[t:t+window_size]
            mu=samples.mean()
            sigma=samples.std()
            if mu != 0:
                cv.loc[t]=sigma/mu
        cv = cv.dropna()
        return (cv)
     
    
    def get_details(self,verbose = True):
        details = {}
        plot_infos = {}
        details['FR'] = self.shank_event_rate()
        details['total_mua'] = len(self.clusters('mua'))
        details['total_sua'] = len(self.clusters('good'))
        details['contamination'] = self.get_group_contamination(self.clusters('good'))
        details['sua_FR_mean'], details['sua_FR_std'] = self.group_event_rate('good','average')
        details['mua_FR_mean'], details['mua_FR_std'] = self.group_event_rate('mua','average')
        spikes = self.get_spikes_on_groups(self.list_of_non_noisy_groups())
        plot_infos['CV'] = self.get_cv(spikes)
        spikes = []
        clusters = self.clusters('good')
        for cluster in clusters:
            spikes.extend(self.spikes_on_cluster(cluster))
        plot_infos['SUA_FR'] = self.get_group_FR_plot(spikes,bin_size = 50e-2)
        spikes = []
        clusters = self.clusters('mua')
        for cluster in clusters:
            spikes.extend(self.spikes_on_cluster(cluster))
        plot_infos['MUA_FR'] = self.get_group_FR_plot(spikes,bin_size = 50e-2)
        plot_infos['SUA_Hist'] = self.group_event_rate(group_name = 'good')
        plot_infos['MUA_Hist'] = self.group_event_rate(group_name = 'mua')
        
        if verbose == True:
            print(f'animal: {self.get_name()}')
            print(f'Shank: {self.kwik_model.name}')
            print(f'Last spike: {self.get_duration():.2f}')
            print('Population firing rate: {:.2f}'.format(details['FR']))
            print('Number of SUA clusters: {}'.format(details['total_sua']))
            print('Number of MUA clusters: {}'.format(details['total_mua']))
            print('Average contamination: {:.5f}%'.format(details['contamination']*100))
            print('SUA average fire rate: {:.2f}'.format(details['sua_FR_mean']))
            print('SUA fire rate std: {:.2f}'.format(details['sua_FR_std']))
            print('MUA average fire rate: {:.2f}'.format(details['mua_FR_mean']))
            print('MUA fire rate std: {:.2f}'.format(details['mua_FR_std']))
            
        return details,plot_infos
    
          