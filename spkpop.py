from klusta.kwik import KwikModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from datasource import DataSource

class SpkPop():
    
    def __init__(self, data_source):
        
        if type(data_source) is not DataSource:
            raise ValueError("Data source type not supported")
        self.data_source = data_source
        print(f'The following models where find in this data source:{self.data_source.models.keys()}')
        self.model_names = list(self.data_source.models.keys())
        #Default model is used when no specific model name is passed on functions
        self.default_model = self.data_source.models[list(self.model_names)[0]]

    
    def plot_cv (self,model_name = None, cv = None, ax=None, a = None, b = None, bin_size = 50e-3):
        """! @brief Plots the CV.

        Parameters: 
        cv: The CV values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        if cv is None:
            cv = self.data_source.get_CV(model_name=model_name,a=a,b=b,bin_size = 50e-3)
        cv = cv.plot()
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()
        cv = self.improve_axis(ax = cv,title=name,xlabel='Time (s)',ylabel= 'CV' )

    def plot_ifr (self,model_name = None, ifr = None, ax=None, a = None, b = None):
        """! @brief Plots the intant fire rate.

        Parameters: 
        ifr: The IFR values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        if ifr is None:
            ifr = self.data_source.inst_fr(model_name=model_name,a = a, b = b)
        ifr = ifr.plot()
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()
        ifr = self.improve_axis(ax = ifr,title=name,xlabel='Time (s)',ylabel= 'IFR' )
    
    def plot_isi (self, model_name = None, isi = None, bin_size = 50e-3, ax=None, a = None, b = None, max_isi_value = None):
        """! @brief Plots the isi.

        Parameters: 
        isi: The isi values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Nivaldo A P de Vasconcelos
        Date: 11.aug.2021
        """        
        if isi is None:
            isi = self.data_source.get_isi(model_name=model_name, a = a, b = b)
        isi = np.array(isi)
        if not max_isi_value is None:
            isi = isi[isi<=max_isi_value]
        max_isi = np.max(isi)
        min_isi = np.min(isi)
        print(f'minimum: {min_isi}')
        print(f'maximum: {max_isi}')
        t=np.arange(min_isi,max_isi,bin_size)
        count, edges = np.histogram(isi,bins=t)
        isi=pd.Series(count, index=edges[0:-1])

        isi = isi.plot()
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()
            
        isi_plot = self.improve_axis(ax = isi,title=name,xlabel='isi (s)', ylabel= 'count' )
    
    def box_plot_fr(self,model_name = None,clu_list = None, a = None, b = None):
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()
        if clu_list is None:
            clu_list = self.data_source.get_SUA_clusters(model_name=model_name)
        group_fr = self.data_source.group_fr(clu_list = clu_list, a = a, b = b, model_name = model_name)
        print(group_fr)
        df = pd.DataFrame(group_fr, columns=[name])
        ax = df.plot.box() 
        box_plot_fr = self.improve_axis(ax = ax,title='Fire rate per cluster', ylabel= 'Fire Rate',xrotation=0 )

    def improve_axis (self,ax, title=None, xlabel=None, ylabel=None, fontsize=14 ,fontname="B612",xrotation=45):
        """! @brief Auxiliar function to the plot functions, used to define title, labels of axis, fontsize etc.

        Parameters: 
        cv: The CV values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Nivaldo 
        Date: 11.aug.2021
        """
        if not(title is None):
            ax.set_title(title, fontsize=18)
        if (xlabel is None):
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
        else:
            ax.set_xlabel(xlabel, fontsize=18)

        if (ylabel is None):
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        else:
            ax.set_ylabel(ylabel, fontsize=18)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname(fontname)
            label.set_fontsize(fontsize)

        for label in (ax.get_xticklabels()):
            label.set_rotation(xrotation)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        return (ax)

    def plot_cluster_waveforms(self,model_name,cluster_list = None,nspikes = 30, save_path=None, wave_color="gray"):
        
        model= self.data_source.models[model_name]

       
        if cluster_list is None:
            cluster_list = self.data_source.get_SUA_clusters(model_name = model_name)
            if len(cluster_list)>10:
                group_fr = self.data_source.group_fr(clu_list = cluster_list, model_name = model_name)
                group_fr.sort_values(inplace = True)
                low_fr = group_fr.iloc[0:5]
                high_fr = group_fr.iloc[-6:-1]
                group_fr = low_fr.append(high_fr)
                cluster_list = list(group_fr.index)
                print(cluster_list)
        fig, ax = plt.subplots(figsize=(30, 9),num=None,  dpi=50, facecolor='w', edgecolor='k')
        ax.set_axis_off()
        ax.plot()
        np.random.seed(int(time.time()))
        position = 0
        for clu in cluster_list:
            w = model.get_cluster_waveforms(clu)
            axins = ax.inset_axes([0.1*position, 0, 0.1, 1])
            position += 1
            y_scale = .7
            x_scale = 4
            num_channels = w.shape[2]
            waveform_size = w.shape[1]
            spike_id = np.arange(w.shape[0])
            np.random.shuffle(spike_id)
            spike_id = spike_id[0:nspikes]
            posx=np.flipud (model.channel_positions[:,0])
            posy=np.flipud (model.channel_positions[:,1])
            for ch in range (0,num_channels):
                x_offset = model.channel_positions[ch,0]
                y_offset =posy[ch]*y_scale
                #y_offset = model.channel_positions[ch,1]*y_scale
                mu_spikes = np.mean(w[:,:,ch],0)
                for i in spike_id:
                    spike = w[i,:,ch]
                    x=x_scale*x_offset+range(0,waveform_size)
                    axins.plot(x,0.05*spike+y_offset,color=wave_color,alpha=0.5)
                    axins.set_axis_off()
                    axins.set_title(clu)
                axins.plot(x,0.05*mu_spikes+y_offset,"--",color="black",linewidth=3,alpha=0.3)