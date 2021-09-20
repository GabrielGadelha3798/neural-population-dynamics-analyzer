from klusta.kwik import KwikModel
import pandas as pd
from abc import ABC, abstractmethod

class Scaffold(ABC):
    """!  @brief basic structure for all types of files. All file types need all functions bellow to allow the application to work.

    @author: Nivaldo A P de Vasconcelos & Gabriel G. Gadelha
    @date: 2021.Jul.27
    """
    
    @abstractmethod
    def get_name(self):
        """! @brief Returns the name.
        """
        pass

    @abstractmethod
    def set_name(self):
        """! @brief Defines file name
        """
        pass
    
    @abstractmethod
    def set_file_path (self):
        """! @brief Defines the corresponding file path
        """
        pass

    @abstractmethod
    def duration(self):
        """! @brief Returns duration of recording in seconds.
        """        
        pass

    @abstractmethod
    def sampling_rate (self):
        """! @brief Returns the sampling rate used during the recordings 
        """
        pass
    
    @abstractmethod
    def get_spike_times(self):
        """! @brief function used to return spike times.
            Default return method must be a list
        """
        pass
    
    @abstractmethod
    def get_spike_clusters(self):
        """! @brief function used to return spike clusters.
            Default return method must be a list
        """
        pass
    @abstractmethod
    def get_clusters(self):
        """! @brief function used to return all unique spike clusters.
            Default return method must be a list
        """
        pass

    @abstractmethod    
    def num_of_channels (self):
        """! @brief Returns the number of channels in file.
        """
        pass