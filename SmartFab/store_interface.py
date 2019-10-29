##########################################################################################################
# Author: Jaya Ram Kollipara                                                                             #
# Task: Smarfab_Q1                                                                                       #
# Date: 10/10/2019                                                                                       #
# Details: Interface Class, acts like a template. Any changes in the DB technology, corresponding        #
#          interface solutions can be added here, the user will uses this interface class to access the  #
#          meta data store.                                                                              #
##########################################################################################################

import abc


class IMetaStore:
    """
    Abstract Class
    Remember to implement the abstract methods in the Sub-Class
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def query_nunique_lt_10(self):
        pass

    @abc.abstractmethod
    def query_by_category_categorical_perc_na(self):
        pass
