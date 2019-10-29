##########################################################################################################
# Author: Jaya Ram Kollipara                                                                             #
# Task: Smarfab_Q1                                                                                       #
# Date: 10/10/2019                                                                                       #
# Details: Sample test to specify usage of the API                                                       #
##########################################################################################################


from user_store_api import Store


print(Store().query_nunique_lt_10())

print(Store().query_by_category_categorical_perc_na())
