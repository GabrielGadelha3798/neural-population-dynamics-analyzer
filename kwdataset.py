

from klusta.kwik import KwikModel
import bact.kwikfile as kf
import warnings
import numpy as np
import pandas as pd
import imp

imp.reload (kf)

class KWDataset:
	"""! Model for a dataset based on kwik files, strongly based on KwikModel from phy project.

	Usually we had organized the neuronal data per shanks, using one kwik file per shank. Once 
	the KwikFile class provides an abstraction to handle contents in kwik file, the KWDataset
	class provides an abstraction to handle a collection of KwikFile objects.

	 Author: Nivaldo A P de Vasconcelos
	 Date: 2018.Feb.02
	"""

	def __init__(self,name,kpath=None):
		"""! @brief  Creates a KWDataset instance

        Usually, we had organized the neuronal data per shanks, using one kwik
        file per shank. Once the KwikFile class provides an abstraction to
        handle contents in kwik file, the KWDataset can be created directly
        from paths to kwik files by using as input a valid dictionary with
        paths'information.

        
        @param name	 name to identify the object.
		@param kpath A dictionary,  where each key is a shank id, and the corresponding value is its the path to the corresponding kwik file where the spike data recorded in that shank is stored. When it is None means nothing happens.

		@author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.07"""
		
		self.kpath = kpath  
		self.name=name
		self.shank=dict()

		if (self.kpath is not None):
			if not(isinstance(self.kpath,dict)):
				raise ValueError("The klist argument must be a *dict* of full paths to kwik files.")
			for shank_id in self.kpath.keys():
				self.add_kwik_file (kpath[shank_id],shank_id)
 
 	
	def sampling_rate (self):
		"""! @brief  Returns the sampling rate used during the recordings, in each shank

		It returns the information by using a dictionary, where each key is a shank id, and the corresponding value is the sampling rate using during the recording of that shank.

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Feb.02
        """
		sr=dict()
		for shank_id in  self.get_shank_ids():
			sr[shank_id]=self.shank[shank_id].sampling_rate()
		l=list(set(list(sr.values())))
		if (len(l)>1):
			return (sr)
		else:
			return (l[0])



	def close(self):
		"""! @brief  Closes the collection 

		It implies to call close method for each KwikFile object.
		This method is usefull to release the resource allocated during the  startup of KwikFile instances

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Feb.02
        """
		
		for shank_id in  self.get_shank_ids():
			self.shank[shank_id].close()

	def list_of_groups(self):
		"""! @brief  Returns a list of groups in each shank

		Returns, in a dictionary, where each key is shank id, and the value is the corresponding list the of groups in that shank.

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Feb.02
        """

		data=dict()
		for shank_id in  self.get_shank_ids():
			data[shank_id]=self.shank[shank_id].list_of_groups()
		return (data)

	def all_clusters (self):
		"""! @brief Returns the list of all clusters in each shank

        Returns, in a dictionary, where each key is shank id, and the value is the corresponding list clusters in that shank.

        Author: Nivaldo A P de Vasconcelos
        Date: 2018.Mar.07
        """
		data=dict()
		for shank_id in  self.get_shank_ids():
			data[shank_id]=self.shank[shank_id].all_clusters()
		return (data)


	def groups(self):
		"""! @brief  Returns the information of groups in each shank

        Returns, in a dictionary, where each key is shank id, and the value is the corresponding dictionary with the group's information in that shank. Please refer to KwikFile.groups()

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.07
        """
		data=dict()
		for shank_id in  self.get_shank_ids():
			data[shank_id]=self.shank[shank_id].groups()
		return (data)


	def clusters (self,group_name=None):
		"""! @brief   Returns the information of cluster in each shank

        Returns, in a dictionary, where each key is shank id, and the value is the corresponding dictionary with the cluster's information in that shank. Please refer to cluster() in KwikFile.

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.07
        """
		data=dict()
		for shank_id in  self.get_shank_ids():
			data[shank_id]=self.shank[shank_id].clusters(group_name=group_name)
		return (data)

	def all_spikes_on_groups (self,group_names, shank_id=None):
		"""! @brief   Returns the all spike samples

        Usually the clusters are organized in groups. Ex: noise, mua, sua,
        unsorted. This method returns all spike samples , in a single dictionary, list of spike samples, all
        spikes found in a lists of groups (group_names). 

        Parameters:
        group_names: list of group names, where the spikes will be searched. 

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.07
        """


		if (shank_id is None):
			spikes=[]
			for shank_id in  self.get_shank_ids():
				spikes=spikes+self.shank[shank_id].all_spikes_on_groups(group_names)
				spikes.sort()
		else:
			if not(shank_id in self.shank.keys()):
				warnings.warn("\nThere is no shank with the given id: %s.\n" %  shank_id)
			spikes=self.shank[shank_id].all_spikes_on_groups(group_names)
		return (spikes)

	def spikes_on_cluster (self,shank_id,cluster_id):
		"""! @brief   Returns all spike samples within given cluster

        Usually, the clusters are organized in groups. Ex: noise, mua, sua, unsorted. This method allows getting the spike samples in a specific cluster in a specific shank. 

        Parameters:
        @param shank_id the shank id where the spike samples must be got 
        @param cluster_id the cluster id within the shank identified by the shank id, where the spike samples must be taken.

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.07
        """
		if not(shank_id in self.shank.keys()):
			raise ValueError("\nThere is no shank with the given id: %s.\n" %  shank_id)
		return (self.shank[shank_id].spikes_on_cluster(cluster_id))


	def has_shank(self,shank_id):
		"""! @brief    Informs whether a shank's id is in the dataset

        @param shank_id the shank id to be checked;

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """
		return (shank_id in self.shank.keys())

	def add_kwik_file (self,kpath,shank_id):
		"""! @brief   Adds to the dataset the information from a new shank.

        @param kpath path to kwik file where is the information to be added
        @param shank_id the id to be used for this new information

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """

		import warnings

		if self.has_shank(shank_id):
			warnings.warn("\nOnce this shank id was found on the current dataset, it can not be added.\n")

		self.shank[shank_id]=kf.KwikFile(kpath=kpath)
		name_on_kwik_file = self.shank[shank_id].get_name()
		
		if not(shank_id==name_on_kwik_file):
			warnings.warn("\nThe given shank_id (%s) is different of the name found in kwik file (%s): \n" %  (shank_id,name_on_kwik_file))

	def del_shank (self,shank_id):
		"""! @brief   Remove from the dataset the information of a given shank.

        
        @param shank_id the id of the shank to be removed from the dataset

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """

		if not(shank_id in self.shank.keys()):
			warnings.warn("\nThere is no shank with the given id: %s.\n" %  shank_id)
		del self.shank[shank_id]

	def get_name(self):
		"""! @brief   Returns the dataset's name.

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """
		return (self.name)

	def set_name (self,name):
		"""! @brief Sets the dataset's name.

		@param name a string which will be used as the dataset's name 

        @author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """
		self.name=name

	def get_shank_ids (self):
		"""! @brief Returns the list of shank ids 


		Returns the shank ids in a list.

		@author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """
		l=list(self.shank.keys())
		l.sort()
		return (l)

	def update_shank (self,shank_id,kpath):
		"""! @brief Updates the information for a given shank

		Once a while it demands updates the information for a given shank from the
		corresponding kwik file. This is the goal of this method.

		@param shank_id the shank's id in the dataset;
		@param kpath the full path to the kwik file.

		Returns the shank ids in a list.

		@author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """

		if not(shank_id in self.shank.keys()):
   			raise ValueError("\nThere is no shank with the given id: %s.\n" %  shank_id)
		self.del_shank (shank_id)
		self.add_kwik_file (kpath,shank_id)


	def group_firing_rate(self,shank_list=None,group_names=None,a=None,b=None):
		"""! @brief Returns the global firing taking a subset of groups of a subset of shanks, during a given time period.

		@param shank_list list of shank ids in which the firing rate must be calculated. All shanks are taken when this parameter is `None` (default)
		@param group_names lits of group names in which the firing rate must be calculated. All groups are taken when this parameter is `None` (default);
		@param a the beginning of the period time. The default value is `None`. The meaning of this value will interpreted by method `group_firing_rate` in `KwikFile` class;
		@param b the end of the period time. The default is `None`. The meaning of this value will interpreted by method `group_firing_rate` in `KwikFile` class.

		@author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """

		if not(isinstance(shank_list,list)) and not(shank_list is None):
			raise ValueError("\nThe first argument must be a list or a None.") 
		if shank_list is None:
			shank_list=list(self.shank.keys())
		spk=dict()
		for shank_id in shank_list:
			spk[shank_id]=self.shank[shank_id].group_firing_rate(group_names=group_names,a=a,b=b)
		return (spk)


	def group_firing_rate_to_dataframe (self,shank_list=None,group_names=None,a=None,b=None):
		"""! @brief Returns panda datafram with the global firing taking a subset of groups of a subset of shanks, during a given time period.

		@param shank_list list of shank ids in which the firing rate must be calculated. All shanks are taken when this parameter is `None` (default)
		@param group_names lits of group names in which the firing rate must be calculated. All groups are taken when this parameter is `None` (default);
		@param a the beginning of the period time. The default value is `None`. The meaning of this value will interpreted by method `group_firing_rate_to_dataframe` in `KwikFile` class;
		@param b the end of the period time. The default is `None`. The meaning of this value will interpreted by method `group_firing_rate_to_dataframe` in `KwikFile` class.

		@author: Nivaldo A P de Vasconcelos
        @date: 2018.Mar.08
        """
		d=self.group_firing_rate (shank_list=shank_list,group_names=group_names,a=a,b=b)

		df=pd.DataFrame()
		for shank_id in d.keys():
			df=df.append (self.shank[shank_id].group_firing_rate_to_dataframe(group_names=group_names,a=a,b=b))
		df["dataset"]=self.name
		return (df)



