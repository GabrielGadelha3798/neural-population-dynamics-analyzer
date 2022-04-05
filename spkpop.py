import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from phylib.stats.ccg import correlograms

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

    
    def plot_cv (self,model_name = None, cv = None,cluster_list = None, ax=None, a = None, b = None, bin_size = 50e-3,window_size = 10,vertical_line = None,result_path = None,xrotation = 0,title=None,xlabel='Time (s)',ylabel= 'CV'):
        """! @brief Plots the CV.

        Parameters: 
        cv: The CV values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        if cv is None:
                cv = self.data_source.get_CV(model_name=model_name,bin_size = bin_size,window_size = window_size, clu_list = cluster_list)
        cv = cv.loc[a:b]
        
        ymax = max(cv)
        ymin = min(cv)
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()

        if ax is None:
            fig = plt.figure()
            fig.suptitle(name)
            ax = fig.add_subplot()

        ax.plot(cv)

        ax = self.improve_axis(ax = ax,title=title,xlabel=xlabel,ylabel= ylabel,vertical_line = vertical_line,ymax = ymax,ymin=ymin,xrotation = xrotation)
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()
        return (cv,ax,ymax,ymin)
    
    def plot_cv_hist (self,model_name = None, cv = None,cluster_list = None, ax=None, a = None, b = None, bin_size = 50e-3,window_size = 10,xrotation = 0,result_path = None,title = None,xlabel='Time (s)',ylabel= 'CV',density = True,color = "black"):
        """! @brief Plots the CV.

        Parameters: 
        cv: The CV values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        if cv is None:
                cv = self.data_source.get_CV(model_name=model_name,bin_size = bin_size,window_size = window_size, clu_list = cluster_list)
        cv = cv.loc[a:b]
        

        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()

        fig = plt.figure(figsize=[20,5])
        fig.suptitle(name)

        if ax is None:
            fig = plt.figure()
            fig.suptitle(name)
            ax = fig.add_subplot()

        if density == False:
            ax.hist(cv, bins=50,histtype='step')
        else:
            # max_cv = np.max(cv)
            # min_cv = np.min(cv)
            # t=np.arange(min_cv,max_cv,(max_cv-min_cv)/50)
            # count, edges = np.histogram(cv,bins=t)
            # cv=pd.Series(count, index=edges[0:-1])
            ax = sns.kdeplot(ax=ax,
            data= cv,
            shade = True,
            log_scale=False, legend=False,color=color
            )
            # sns.distplot(ax =ax, a = cv, hist=True, kde=True, 
            # bins=int(50), color = 'darkblue', 
            # hist_kws={'edgecolor':'black'},
            # kde_kws={'linewidth': 4})
            # min_x,max_x = ax.get_xlim()
            # ax.set_xlim(0,max_x)
            # ax.hist(cv)

        ax = self.improve_axis(ax = ax,title=title,xlabel=xlabel,ylabel= ylabel,xrotation = xrotation)
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()
        return (cv,ax)

    def plot_ifr (self,model_name = None, ifr = None,cluster_list = None, ax=None, a = None, b = None,bin_size = 50e-3,vertical_line = None,result_path = None, xrotation = 0,title = None,xlabel='Time (s)',ylabel= 'IFR'):
        """! @brief Plots the intant fire rate.

        Parameters: 
        ifr: The IFR values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        if ifr is None:
            if cluster_list is None:
                cluster_list = self.data_source.get_SUA_clusters(model_name=model_name)
            ifr = self.data_source.inst_fr(model_name=model_name,a = a, b = b,clu_list = cluster_list, bin_size = bin_size)
        
        ymax = max(ifr)
        ymin = min(ifr)
        
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()
        
        if ax is None:
            fig = plt.figure()
            fig.suptitle(name)
            ax = fig.add_subplot()
        
        ax.plot(ifr)
        ax = self.improve_axis(ax = ax,xlabel=xlabel,ylabel= ylabel,vertical_line = vertical_line,ymax = ymax,ymin=ymin ,xrotation = xrotation)
       
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()
        
        return (ifr,ax,ymax,ymin)

    def plot_ifr_hist (self,model_name = None, ifr = None,cluster_list = None, ax=None, a = None, b = None,bin_size = 50e-3,result_path = None,xrotation = 0, title = None,xlabel='Time (s)',ylabel= 'IFR',density = True,color = "black"):
        """! @brief Plots the intant fire rate.

        Parameters: 
        ifr: The IFR values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Gabriel G Gadelha
        Date: 11.aug.2021
        """
        if ifr is None:
            if cluster_list is None:
                cluster_list = self.data_source.get_SUA_clusters(model_name=model_name)
            ifr = self.data_source.inst_fr(model_name=model_name,a = a, b = b,clu_list = cluster_list, bin_size = bin_size)
        

        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()

        if ax is None:
            fig = plt.figure()
            fig.suptitle(name)
            ax = fig.add_subplot()
        
        if density == False:
            ax.hist(ifr, bins=50,histtype='step')
        else:
            # max_ifr = np.max(ifr)
            # min_ifr = np.min(ifr)
            # t=np.arange(min_ifr,max_ifr,(max_ifr-min_ifr)/50)
            # count, edges = np.histogram(ifr,bins=t)
            # ifr=pd.Series(count, index=edges[0:-1])
            sns.kdeplot(ax=ax,
            data= ifr,
            shade = True,
            log_scale=False, legend=False,color=color
            )
            # min_x,max_x = ax.get_xlim()
            # ax.set_xlim(0,max_x)

        ax = self.improve_axis(ax = ax,title = title, xlabel=xlabel,ylabel = ylabel,xrotation = xrotation)
       
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()
        return (ifr,ax)

    def plot_isi (self, model_name = None, isi = None,cluster_list = None, bin_size = 50, ax=None, a = None, b = None, max_isi_value = None,result_path = None,xrotation = 0,title= None,xlabel='isi (s)', ylabel= 'count',density = True,color="black"):
        """! @brief Plots the isi.

        Parameters: 
        isi: The isi values.
        When this input is 'None' all clusters are taken, and all duration of the exam is considered.

        Author: Nivaldo A P de Vasconcelos
        Date: 11.aug.2021
        """        
        if isi is None:
            isi = self.data_source.get_isi(model_name=model_name,clu_list=cluster_list, a = a, b = b)
        isi = np.array(isi)
        #getting isi im miliseconds
        isi = isi*1000
        if not max_isi_value is None:
            isi = isi[isi<=max_isi_value]
        else:
            isi = np.sort(isi)
            isi = isi[0:int(len(isi)*0.96)]
            print(isi[int(len(isi)*0.96)])
        
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()

        if ax is None:
            fig = plt.figure()
            fig.suptitle(name)
            ax = fig.add_subplot()
        
        
        if density == True:
            ax = sns.kdeplot(ax=ax,
            data= isi,
            shade = True,
            log_scale=False, legend=False,color=color
            )
            ax.set_xlim(0,800)            
        else:
            max_isi = np.max(isi)
            min_isi = np.min(isi)
            t=np.arange(min_isi,max_isi,bin_size)
            count, edges = np.histogram(isi,bins=t)
            isi=pd.Series(count, index=edges[0:-1])
            ax.plot(isi)

        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()
            
        ax = self.improve_axis(ax = ax,title=title,xlabel=xlabel, ylabel= ylabel,xrotation = xrotation )
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()
        return (isi,ax)

    def box_plot_fr(self,model_name = None,clu_list = None, a = None, b = None,result_path = None):
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()
        if clu_list is None:
            clu_list = self.data_source.get_SUA_clusters(model_name=model_name)
        group_fr = self.data_source.group_fr(clu_list = clu_list, a = a, b = b, model_name = model_name)
        # print(group_fr)
        df = pd.DataFrame(group_fr, columns=[name])
        ax = df.plot.box() 
        box_plot_fr = self.improve_axis(ax = ax,title='Fire rate per cluster', ylabel= 'Fire Rate',xrotation=0 )
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()


    def improve_axis (self,ax, title=None, xlabel=None, ylabel=None, fontsize=14 ,fontname="B612",xrotation=45, vertical_line = None,ymax = None,ymin = None):
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
        if vertical_line is not None:
            ax.vlines(vertical_line,ymin=ymin,ymax = ymax,linestyle= '--',colors = 'black',label='drug injection time',zorder= 10)
        return (ax)


    def plot_cluster_waveforms(self,model_name,cluster_list = None,nspikes = 30, result_path=None, wave_color="gray"):
        
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
                # print(cluster_list)
            else:
                group_fr = self.data_source.group_fr(clu_list = cluster_list, model_name = model_name)
                group_fr.sort_values(inplace = True)
                cluster_list = list(group_fr.index)
                # print(cluster_list)                
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
                y_offset = posy[ch]*y_scale
                #y_offset = model.channel_positions[ch,1]*y_scale
                mu_spikes = np.mean(w[:,:,ch],0)
                for i in spike_id:
                    spike = w[i,:,ch]
                    x=x_scale*x_offset+range(0,waveform_size)
                    axins.plot(x,0.05*spike+y_offset,color=wave_color,alpha=0.5)
                    axins.set_axis_off()
                    axins.set_title(clu)
                axins.plot(x,0.05*mu_spikes+y_offset,"--",color="black",linewidth=3,alpha=0.3)
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()

    def get_best_cluster_waveform(self,sheet):
        sheet = np.mean(sheet, axis=0)
        nchannels=np.shape(sheet)[1]
        A=pd.Series(index=range(nchannels), dtype=float)
        for ch in range(nchannels):
            A.loc[ch]=np.max(sheet[:,ch])-np.min(sheet[:,ch])
        best_channel=A.sort_values(ascending=False).index[0]
        return (best_channel)

    def plot_best_cluster_waveforms(self,model_name,cluster,nspikes = 30, result_path=None):
        
        model= self.data_source.models[model_name]

        
        w = model.get_cluster_waveforms(cluster)
        spikes_count = w.shape[0]
        ch = self.get_best_cluster_waveform(w)
        plt.figure(figsize=[4,4])
        ax=plt.subplot(1,1,1) 
        mu=np.mean(w, axis=0)
        if (spikes_count<nspikes):
            nspikes = spikes_count
        I=np.random.permutation(spikes_count)[0:nspikes]
        for i in I:
            sheet=w[i]
            ax.plot (sheet[:,ch],'k')
        ax.plot (mu[:,ch],'r--')
        ax.set_title(f'{cluster}:{ch}')
        ax.set_xlabel('time (ms)')
        ax.set_ylabel('amplitude (mV)')
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()

    def plot_correlogram(self,model_name, cluster,ax = None, a = None, b = None,sample_rate = 30000, bin_size = 1e-3,window_size = 1, result_path=None, wave_color="gray"):
        
        model = self.data_source.models[model_name]

        if a is None:
            a = 0
        if b is None:
            b = model.duration()
        
        if model_name is None:
            name  = self.data_source.default_model
        else:
            name = self.data_source.models[model_name].get_name()

        if ax is None:
            fig = plt.figure()
            fig.suptitle(name)
            ax = fig.add_subplot()  

        clu_list = []
        spikes = model.get_spike_times(a=a,b=b,clu_list=[cluster],return_mode='list') 
        for _ in range(len(spikes)):
            clu_list.append(cluster)
        acf=correlograms(spikes, clu_list, sample_rate = sample_rate,  bin_size = bin_size, window_size = window_size)
        acf= acf.ravel()
        x=np.arange(len(acf))-(len(acf)/2)
        ax.stem (x,acf, markerfmt=" ", basefmt=" ")
        ax.set_xlabel('time (ms)')
        ax.set_title(f'{cluster}')
        if result_path is not None:
            plt.savefig(result_path, pad_inches=0.2)
            plt.close()

    def all_database_plots(self,result_path):
        
        for shank in self.data_source.model_names:
            model_path = os.path.join(result_path,shank.split(' ')[0].replace('/','.'))
            shank_path = model_path + '/' + shank.split(' ')[2]
            population_path = shank_path + '/' + 'population'
            sua_path = shank_path + '/' + 'SUA'
            sua_clusters = self.data_source.get_SUA_clusters(shank)
            mua_clusters = self.data_source.get_MUA_clusters(shank)
            population_clusters = sua_clusters.extend(mua_clusters)
            self.__create_directories(model_path,shank_path,sua_clusters)
            
            print('ploting population of : ' + shank)
            try:

                self.plot_ifr(model_name=shank,bin_size=10,vertical_line=3600,result_path=population_path + '/full_ifr.pdf',cluster_list=population_clusters)
                self.plot_ifr(model_name=shank,a=3000,b=4200,bin_size=10,vertical_line=3600,result_path=population_path + '/zoom_ifr.pdf',cluster_list=population_clusters)
                self.plot_cv(model_name=shank,bin_size=50e-3,window_size = 10,vertical_line=3600,result_path=population_path + '/zoom_CV.pdf',cluster_list=population_clusters)
                self.plot_cv(model_name=shank,bin_size=50e-3,window_size = 10,vertical_line=3600,a= 3000, b = 4200, result_path=population_path + '/CV.pdf',cluster_list=population_clusters)
                self.plot_isi(model_name=shank,cluster_list = population_clusters,b=3600,bin_size = 50,max_isi_value=1000)
                self.plot_isi(model_name=shank,cluster_list = population_clusters,a=3600,bin_size  = 50,max_isi_value=1000)
                plt.savefig(population_path + '/isi.pdf')
                plt.close()
            except Exception as e:
                print('error in population plots: ' + str(e))
            
            print('ploting SUA of : ' + shank)
            try:
                self.plot_ifr(model_name=shank,bin_size=10,vertical_line=3600,result_path=sua_path + '/full_ifr.pdf',cluster_list=sua_clusters)
                self.plot_ifr(model_name=shank,a=3000,b=4200,bin_size=10,vertical_line=3600,result_path=sua_path + '/zoom_ifr.pdf',cluster_list=sua_clusters)
                self.plot_cv(model_name=shank,bin_size=50e-3,window_size = 10,vertical_line=3600,result_path=sua_path + '/zoom_CV.pdf',cluster_list=sua_clusters)
                self.plot_cv(model_name=shank,bin_size=50e-3,window_size = 10,vertical_line=3600,a= 3000, b = 4200, result_path=sua_path + '/CV.pdf',cluster_list=sua_clusters)
                self.plot_cluster_waveforms(model_name=shank,nspikes=40,result_path=sua_path + '/waveforms.pdf')      
                self.plot_isi(model_name=shank,cluster_list = sua_clusters,b=3600,bin_size = 50,max_isi_value= 1000)
                self.plot_isi(model_name=shank,cluster_list = sua_clusters,a=3600,bin_size  = 50,max_isi_value= 1000)
                plt.savefig(sua_path + '/isi.pdf')
                plt.close()
            except Exception as e:
                print('error in SUA plots: ' + str(e))

            #cluster specific plots
            for cluster in sua_clusters:
                print('ploting cluster nº ' + str(cluster) + ' with {} spikes'.format(self.data_source.get_cluster_spikes_count(cluster,shank)))
                try:
                    cluster_path = shank_path + '/' + f'cluster {cluster}'
                    self.plot_best_cluster_waveforms(model_name=shank, cluster = cluster,nspikes = 50, result_path=cluster_path + '/bestwaveform.pdf')
                    self.plot_ifr(model_name=shank,bin_size=10,vertical_line=3600,result_path=cluster_path + '/full_ifr.pdf',cluster_list=[cluster])
                    self.plot_ifr(model_name=shank,a=3000,b=4200,bin_size=10,vertical_line=3600,result_path=cluster_path + '/zoom_ifr.pdf',cluster_list=[cluster])
                except Exception as e:
                    print('error in cluster ' + str(cluster) + ' : ' + str(e))
                    continue


    def all_database_united_plots(self,result_path,plot_population = True, plot_SUA = True,plot_clusters = True,plot_waveforms = True):

        for shank in self.data_source.model_names:
            shank_name = shank.split(' ')[0].replace('/','.')
            model_path = os.path.join(result_path,shank_name)
            shank_path = model_path + '/' + shank.split(' ')[2]
            population_path = shank_path + '/' + 'population'
            sua_path = shank_path + '/' + 'SUA'
            sua_clusters = self.data_source.get_SUA_clusters(shank)
            mua_clusters = self.data_source.get_MUA_clusters(shank)
            population_clusters = sua_clusters.extend(mua_clusters)
            self.__create_directories(model_path,shank_path,sua_clusters,plot_population = plot_population,plot_sua=plot_SUA,plot_clusters=plot_clusters)
            
            drug_injection_time = 3600
            if plot_population == True:
                population_fig = plt.figure(figsize=[12,12])
                # population_fig.suptitle("Population of : " + shank,fontsize=30)

                gs = GridSpec(3, 3,wspace= 0.4,hspace = 0.5, figure = population_fig)
                gs.update(left=0.07,right=0.96,top=0.93,bottom=0.08)
                population_fig.text(0.04,0.98,'a)',horizontalalignment='left',
     verticalalignment='top',fontsize=25)
                population_fig.text(0.04,0.66,'b)',horizontalalignment='left',
verticalalignment='top',fontsize=25)
                population_fig.text(0.04,0.34,'c)',horizontalalignment='left',
verticalalignment='top',fontsize=25)
                population_fig.text(0.36,0.34,'d)',horizontalalignment='left',
verticalalignment='top',fontsize=25)
                population_fig.text(0.68,0.34,'e)',horizontalalignment='left',
verticalalignment='top',fontsize=25)
                # gssub = gs[0:3].subgridspec(1, 2,width_ratios = [3,1],wspace = 0.0)
                # gssub2 = gs[3:6].subgridspec(1, 2,width_ratios = [3,1],wspace = 0.0)
                # ifr_ax = population_fig.add_subplot(gssub[0])
                # ifr_hist_ax = population_fig.add_subplot(gssub[1])
                # cv_ax = population_fig.add_subplot(gssub2[0])
                # cv_hist_ax = population_fig.add_subplot(gssub2[1])
                ifr_ax = population_fig.add_subplot(gs[0:3])
                cv_ax = population_fig.add_subplot(gs[3:6])
                ifr_hist_ax = population_fig.add_subplot(gs[6])
                cv_hist_ax = population_fig.add_subplot(gs[7])   
                # ifr_zoom_ax = population_fig.add_subplot(gs[6])
                # cv_zoom_ax = population_fig.add_subplot(gs[7])
                isi_ax = population_fig.add_subplot(gs[8])
                
                print('ploting population of : ' + shank)
                try:

                
                    ymax = []
                    ymin = []
                    # self.plot_ifr(ax = ifr_ax ,model_name=shank,bin_size=4,vertical_line=drug_injection_time,cluster_list=population_clusters,xrotation = 0,xlabel = 'Time (s)',ylabel = 'IFR')
                    ifr, ifr_ax,ifr_max,ifr_min = self.plot_ifr(ax = ifr_ax ,model_name=shank,bin_size=4,b=drug_injection_time+4,cluster_list=population_clusters,xrotation = 0,xlabel = 'Time (s)',ylabel = 'IFR')
                    ymax.append(ifr_max)
                    ymin.append(ifr_min)
                    pre_cocaine_mean_ifr = np.mean(ifr)
                    pre_cocaine_std_ifr = np.std(ifr)
                    ifr, ifr_ax,ifr_max,ifr_min = self.plot_ifr(ax = ifr_ax ,model_name=shank,bin_size=4,a=drug_injection_time,cluster_list=population_clusters,xrotation = 0,xlabel = 'Time (s)',ylabel = 'IFR')
                    ymax.append(ifr_max)
                    ymin.append(ifr_min)
                    post_cocaine_mean_ifr = np.mean(ifr)
                    post_cocaine_std_ifr = np.std(ifr)
                    lines = ifr_ax.get_lines()
                    lines[0].set_color("darkblue")
                    lines[1].set_color("orangered")
                    ifr_ax.vlines(drug_injection_time,max(ymax),min(ymin),color='black',linestyle='--',zorder = 10)
                    left,right = ifr_ax.get_xlim()
                    ifr_ax.set_xlim(0,right)
                    self.plot_ifr_hist( ax = ifr_hist_ax , model_name=shank,bin_size=4, b = drug_injection_time,cluster_list=population_clusters,xlabel = 'IFR', ylabel = 'Density',color = "darkblue" )                    
                    self.plot_ifr_hist( ax = ifr_hist_ax , model_name=shank,bin_size=4, a = drug_injection_time,cluster_list=population_clusters,xlabel = 'IFR', ylabel = 'Density',color = "orangered")
                    left,right = ifr_hist_ax.get_xlim()
                    ifr_hist_ax.set_xlim(0,right)
                    
                    ymax=[]
                    ymin=[]
                    cv, cv_ax, cv_max,cv_min = self.plot_cv(ax = cv_ax, model_name=shank,bin_size=2,window_size = 10,b=drug_injection_time+10,cluster_list=population_clusters,xrotation = 0, xlabel = 'Time (s)',ylabel = 'CV')
                    ymax.append(cv_max)
                    ymin.append(cv_min)
                    pre_cocaine_mean_cv = np.mean(cv)
                    pre_cocaine_std_cv = np.std(cv)
                    cv, cv_ax, cv_max,cv_min = self.plot_cv(ax = cv_ax, model_name=shank,bin_size=2,window_size = 10, a=drug_injection_time,cluster_list=population_clusters,xrotation = 0, xlabel = 'Time (s)',ylabel = 'CV')
                    ymax.append(cv_max)
                    ymin.append(cv_min)
                    post_cocaine_mean_cv = np.mean(cv)
                    post_cocaine_std_cv = np.std(cv)
                    lines = cv_ax.get_lines()
                    lines[0].set_color("darkblue")
                    lines[1].set_color("orangered")
                    cv_ax.vlines(drug_injection_time,max(ymax),min(ymin),color='black',linestyle='--',zorder = 10)
                    left,right = cv_ax.get_xlim()
                    cv_ax.set_xlim(0,right)
                    self.plot_cv_hist(ax = cv_hist_ax, model_name=shank,bin_size=4,b = drug_injection_time,window_size = 20,cluster_list=population_clusters,xlabel = 'CV', ylabel = 'Density',color="darkblue")      
                    self.plot_cv_hist(ax = cv_hist_ax, model_name=shank,bin_size=4,a = drug_injection_time,window_size = 20,cluster_list=population_clusters,xlabel = 'CV', ylabel = 'Density',color = "orangered")
                    left,right = cv_hist_ax.get_xlim()
                    cv_hist_ax.set_xlim(0,right)
                    
                    self.plot_isi(ax = isi_ax, model_name=shank,cluster_list = population_clusters,b=drug_injection_time,bin_size = 5,xrotation = 0,xlabel = 'Interspike Interval (ms)',ylabel = 'Density',max_isi_value=1000,color = "darkblue")
                    self.plot_isi(ax = isi_ax, model_name=shank,cluster_list = population_clusters,a=drug_injection_time,bin_size  = 5, xrotation = 0,xlabel = 'Interspike Interval (ms)',ylabel = 'Density',max_isi_value=1000,color = "orangered")
                    left,right = isi_ax.get_xlim()
                    isi_ax.set_xlim(0,right)
                    population_fig.savefig(population_path + '/panel.pdf')
                    plt.close(fig=population_fig)
                    
                    shank_name = shank_name.replace('/','_')
                    population_caption_path = population_path + f'/{shank_name}populationcaption.tex'
                    if os.path.exists(population_caption_path):
                        os.remove(population_caption_path)
                    with open(population_caption_path,'w') as f:
                        f.write("\textbf{Visão geral da atividade unitária de população neuronal do animal 2021.01.23 no Núcleo Tegmento Laterodorsal no shank 01.}")
                        f.write(f" Todos os plots estão divididos em 2 curvas, a curva em azul representa o período pré-injeção da cocaína, já a curva em laranja representa o histograma do período pós-injeção da cocaína.")
                        f.write(f" (a)  Taxa de disparo instantânea (IFR). A taxa de disparo média ao longo de todo experimento foi de {((pre_cocaine_mean_ifr + post_cocaine_mean_ifr)/2):.2f} $\\pm$ {((pre_cocaine_std_ifr + post_cocaine_std_ifr)/2):.2f}, para o período pré-injeção foi de {pre_cocaine_mean_ifr:.2f} $\pm$ {pre_cocaine_std_ifr:.2f} e pós-injeção foi de {post_cocaine_mean_ifr:.2f} $\pm$ {post_cocaine_std_ifr:.2f}.")
                        f.write(f" (c)  Coeficiente de variação (CV). O CV médio ao longo de todo experimento foi de {((pre_cocaine_mean_cv + post_cocaine_mean_cv)/2):.2f} $\pm$ {((pre_cocaine_std_cv + post_cocaine_std_cv)/2):.2f}, para o período pré-injeção foi de {pre_cocaine_mean_cv:.2f} $\pm$ {pre_cocaine_std_cv} e pós-injeção foi de {post_cocaine_mean_cv} $pm$ {post_cocaine_std_cv}.")
                        f.write(f" (c) Histograma do IFR apresentado em (a). (d) Histograma do CV apresentado em (c). (e) Histograma da distribuição do intervalo interspike.")
                         
                except Exception as e:
                    print('error in population plots: ' + str(e))
            
            if plot_SUA == True:
                sua_fig = plt.figure(figsize=[12,12])
                # sua_fig.suptitle("SUA of : " + shank,fontsize = 30)
                sua_fig.text(0.04,0.98,'a)',horizontalalignment='left',
     verticalalignment='top',fontsize=25)
                sua_fig.text(0.04,0.66,'b)',horizontalalignment='left',
verticalalignment='top',fontsize=25)
                sua_fig.text(0.04,0.34,'c)',horizontalalignment='left',
verticalalignment='top',fontsize=25)
                sua_fig.text(0.36,0.34,'d)',horizontalalignment='left',
verticalalignment='top',fontsize=25)
                sua_fig.text(0.68,0.34,'e)',horizontalalignment='left',
verticalalignment='top',fontsize=25)
                gs = GridSpec(3, 3,wspace= 0.4,hspace = 0.5, figure = sua_fig)
                gs.update(left=0.07,right=0.96,top=0.93,bottom=0.08)
                # gssub = gs[0:3].subgridspec(1, 2,width_ratios = [3,1],wspace = 0.0)
                # gssub2 = gs[3:6].subgridspec(1, 2,width_ratios = [3,1],wspace = 0.0)
                # ifr_ax = sua_fig.add_subplot(gssub[0])
                # ifr_hist_ax = sua_fig.add_subplot(gssub[1])
                # cv_ax = sua_fig.add_subplot(gssub2[0])
                # cv_hist_ax = sua_fig.add_subplot(gssub2[1])
                ifr_ax = sua_fig.add_subplot(gs[0:3])
                cv_ax = sua_fig.add_subplot(gs[3:6])
                ifr_hist_ax = sua_fig.add_subplot(gs[6])
                cv_hist_ax = sua_fig.add_subplot(gs[7])   
                # ifr_zoom_ax = sua_fig.add_subplot(gs[6])
                # cv_zoom_ax = sua_fig.add_subplot(gs[7])
                isi_ax = sua_fig.add_subplot(gs[8])
                print('ploting SUA of : ' + shank)

                try:
                    ymax = []
                    ymin = []
                    ifr, ifr_ax,ifr_max,ifr_min = self.plot_ifr(ax = ifr_ax ,model_name=shank,bin_size=4,b=drug_injection_time+4,cluster_list=sua_clusters,xrotation = 0,xlabel = 'Time (s)',ylabel = 'IFR')
                    ymax.append(ifr_max)
                    ymin.append(ifr_min)
                    ifr, ifr_ax,ifr_max,ifr_min = self.plot_ifr(ax = ifr_ax ,model_name=shank,bin_size=4,a=drug_injection_time,cluster_list=sua_clusters,xrotation = 0,xlabel = 'Time (s)',ylabel = 'IFR')
                    ymax.append(ifr_max)
                    ymin.append(ifr_min)
                    lines = ifr_ax.get_lines()
                    lines[0].set_color("darkblue")
                    lines[1].set_color("orangered")
                    ifr_ax.vlines(drug_injection_time,max(ymax),min(ymin),color='black',linestyle='--',zorder = 10)
                    left,right = ifr_ax.get_xlim()
                    ifr_ax.set_xlim(0,right)
                    self.plot_ifr_hist( ax = ifr_hist_ax , model_name=shank,bin_size=4, b = drug_injection_time,cluster_list=sua_clusters,xlabel = 'IFR', ylabel = 'Density',color = "darkblue" )                    
                    self.plot_ifr_hist( ax = ifr_hist_ax , model_name=shank,bin_size=4, a = drug_injection_time,cluster_list=sua_clusters,xlabel = 'IFR', ylabel = 'Density',color = "orangered")
                    left,right = ifr_hist_ax.get_xlim()
                    ifr_hist_ax.set_xlim(0,right)
                    
                    ymax=[]
                    ymin=[]
                    cv, cv_ax, cv_max,cv_min = self.plot_cv(ax = cv_ax, model_name=shank,bin_size=2,window_size = 10,b=drug_injection_time+10,cluster_list=sua_clusters,xrotation = 0, xlabel = 'Time (s)',ylabel = 'CV')
                    ymax.append(cv_max)
                    ymin.append(cv_min)
                    cv, cv_ax, cv_max,cv_min = self.plot_cv(ax = cv_ax, model_name=shank,bin_size=2,window_size = 10, a=drug_injection_time,cluster_list=sua_clusters,xrotation = 0, xlabel = 'Time (s)',ylabel = 'CV')
                    ymax.append(cv_max)
                    ymin.append(cv_min)
                    lines = cv_ax.get_lines()
                    lines[0].set_color("darkblue")
                    lines[1].set_color("orangered")
                    cv_ax.vlines(drug_injection_time,max(ymax),min(ymin),color='black',linestyle='--',zorder = 10)
                    left,right = cv_ax.get_xlim()
                    cv_ax.set_xlim(0,right)
                    self.plot_cv_hist(ax = cv_hist_ax, model_name=shank,bin_size=4,b = drug_injection_time,window_size = 20,cluster_list=sua_clusters,xlabel = 'CV', ylabel = 'Density',color="darkblue")      
                    self.plot_cv_hist(ax = cv_hist_ax, model_name=shank,bin_size=4,a = drug_injection_time,window_size = 20,cluster_list=sua_clusters,xlabel = 'CV', ylabel = 'Density',color = "orangered")
                    left,right = cv_hist_ax.get_xlim()
                    cv_hist_ax.set_xlim(0,right)
                    
                    self.plot_isi(ax = isi_ax, model_name=shank,cluster_list = sua_clusters,b=drug_injection_time,bin_size = 5,xrotation = 0,xlabel = 'Interspike Interval (ms)',ylabel = 'Density',max_isi_value=1000,color = "darkblue")
                    self.plot_isi(ax = isi_ax, model_name=shank,cluster_list = sua_clusters,a=drug_injection_time,bin_size  = 5, xrotation = 0,xlabel = 'Interspike Interval (ms)',ylabel = 'Density',max_isi_value=1000,color = "orangered")
                    left,right = isi_ax.get_xlim()
                    isi_ax.set_xlim(0,right)
                    sua_fig.savefig(sua_path + '/panel.pdf')
                    plt.close(fig=sua_fig)
                    
                    minimum_fr_clusters = []
                    for clu in sua_clusters:
                        if self.data_source.get_cluster_spikes_count(clu,shank) > 2000:
                            minimum_fr_clusters.append(clu)

                    #plot de correlação
                    vertical = int(np.ceil(np.sqrt(len(minimum_fr_clusters))))
                    horizontal = int(np.ceil(len(minimum_fr_clusters)/vertical))
                    correlogram_fig = plt.figure (figsize=(21,21))
                    count=1
                    for clu in minimum_fr_clusters:
                        correlogram_ax=plt.subplot(vertical,horizontal,count)
                        self.plot_correlogram(model_name=shank,ax=correlogram_ax, cluster = clu)
                        count=count+1
                    correlogram_fig.tight_layout()
                    correlogram_fig.savefig(sua_path + '/correlogram.pdf')
                    plt.close(fig=correlogram_fig)
                    
                    zoom_correlogram_fig = plt.figure (figsize=(21,21))
                    count=1
                    for clu in minimum_fr_clusters:
                        zoom_correlogram_ax=plt.subplot(vertical,horizontal,count)
                        self.plot_correlogram(model_name=shank,ax=zoom_correlogram_ax, cluster = clu,window_size = 0.2)
                        count=count+1
                    zoom_correlogram_fig.tight_layout()
                    zoom_correlogram_fig.savefig(sua_path + '/zoom_correlogram.pdf')
                    plt.close(fig=correlogram_fig)
                    
                    
                    if plot_waveforms:
                        self.plot_cluster_waveforms(model_name=shank,nspikes=40, result_path=sua_path + '/waveforms.pdf')   
                except Exception as e:
                    print('error in SUA plots: ' + str(e))

            if plot_clusters == True:
                #cluster specific plots
                for cluster in sua_clusters:
                    print('ploting cluster nº ' + str(cluster) + ' with {} spikes'.format(self.data_source.get_cluster_spikes_count(cluster,shank)))
                    try:
                        cluster_path = shank_path + '/Clusters/' + f'cluster {cluster}'
                        cluster_fig = plt.figure(figsize=[12,4])
                        # cluster_fig.suptitle("IFR of Cluster " + str(cluster),fontsize=30)

                        gs = GridSpec(1, 2,wspace= 0.4,width_ratios = [2,1], figure = cluster_fig)
                        ifr_ax = cluster_fig.add_subplot(gs[0])
                        hist_ax = cluster_fig.add_subplot(gs[1])
                        self.plot_ifr(ax = ifr_ax, model_name=shank,bin_size=10,vertical_line=3600,cluster_list=[cluster],title = None,ylabel='IFR', xlabel = 'Time (s)')
                        self.plot_ifr_hist(ax = hist_ax, model_name=shank,b = drug_injection_time, bin_size=10,cluster_list=[cluster],title = None, xlabel = 'count',ylabel = None)
                        self.plot_ifr_hist(ax = hist_ax, model_name=shank,a = drug_injection_time, bin_size=10,cluster_list=[cluster],title = None, xlabel = 'count',ylabel = None)
                        hist_ax.set_yticklabels([])
                        hist_ax.set_yticks([])
                        cluster_fig.savefig(cluster_path + '/ifr.pdf')
                        plt.close(fig=cluster_fig)
                        # self.plot_ifr(model_name=shank,a=3000,b=4200,bin_size=10,vertical_line=3600,result_path=cluster_path + '/zoom_ifr.pdf',cluster_list=[cluster])
                        # self.plot_best_cluster_waveforms(model_name=shank, cluster = cluster,nspikes = 50, result_path=cluster_path + '/bestwaveform.pdf')
                    except Exception as e:
                        print('error in cluster ' + str(cluster) + ' : ' + str(e))
                        continue

    def __create_directories(self,model_path,shank_path,sua_clusters,plot_population,plot_clusters,plot_sua):
        try:
            os.mkdir(model_path)
        except Exception as e:
            pass
        try:
            os.mkdir(shank_path)
        except Exception as e:
            pass
        if plot_population:
            try:
                os.mkdir(shank_path + '/population')
            except Exception as e:
                pass
        if plot_sua:
            try:
                os.mkdir(shank_path + '/SUA')
            except Exception as e:
                pass
        if plot_clusters:
            try:
                os.mkdir(shank_path + '/Clusters')
            except Exception as e:
                pass

            for clu in sua_clusters:
                try:
                    os.mkdir(shank_path + '/Clusters/' + f'cluster {clu}')
                except Exception as e:
                    pass
