##########################################################################################################
# Author: Jaya Ram Kollipara                                                                             #
# Task: Smarfab_Q1                                                                                       #
# Date: 10/10/2019                                                                                       #
# Details: Store is a sub-class that inherits the Super Class IMetastore. IMetastore is a interface for  #
#          the user to manage the Meta Data Store                                                        #
# API :  query_nunique_lt_10                                                                             #
#     :  query_by_category_categorical_perc_na                                                           #
##########################################################################################################
from store_interface import IMetaStore
from database_operations.db_utils import create_connector


class Store(IMetaStore):
    """
    Inherits the IMetastore Interface to access the Meta Store.
    API : query_nunique_lt_10
    API : query_by_category_categorical_perc_na
    Features:
        1. Get all variables with nunique < 10
        2. Get all the variables with category == 'categorical' and perc_na < 0.1
    """
    def __init__(self):
        self.connector = None

    def get_connector(self):
        if not self.connector:
            self.connector = create_connector()
        return self.connector

    def query_nunique_lt_10(self):
        return self.get_connector().get_nunique_lt_10()

    def query_by_category_categorical_perc_na(self):
        return self.get_connector().get_by_category_categorical_perc_na()
