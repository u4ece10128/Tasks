##########################################################################################################
# Author: Jaya Ram Kollipara                                                                             #
# Task: Smarfab_Q1                                                                                       #
# Date: 09/10/2019                                                                                       #
# Details: Load data Into DB schema by sql_orm                                                           #
##########################################################################################################


from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Enum, ForeignKey, TEXT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from sql_orm import Required, RequiredForNumeric, ForEnumerable, ForStrList, TypeEnum, Base, TypeCategory, Ternary, \
    ForDateTime


def create_schema():
    try:
        engine = create_engine('mysql+pymysql://root:root1234@localhost:3306/')
        engine.execute("CREATE DATABASE my_db")  # create db
    except SQLAlchemyError:
        pass

    engine = create_engine('mysql+pymysql://root:root1234@localhost:3306/my_db', echo=True)
    Base.metadata.create_all(engine)


def load_db(engine):
    # input dictionary
    # as the mode of input is unknown, for now i have included the provison to pass a  dictionary

    # uncomment below line to pass through command line
    # entry = sys.argv[1]

    # Sample Input
    entry = {'size': 1000,
             'type': 'integer',
             'nunique': 161,
             'category': 'non-categorical',
             'ordinal': True,
             'num_na': 46,
             'perc_na': 0.046,
             'min': 0.0,
             'max': 326.0,
             'mean': 250.74423,
             'std': 39.28161,
             'hist_bins': [0, 1, 2, 4, 5],
             'hist_counts': [0, 1, 2, 4, 5]
             }

    Session = sessionmaker(bind=engine)
    sess = Session()

    # if type  == int or float
    if entry['type'] == TypeEnum.type_int.value or entry['type'] == TypeEnum.type_float.value:
        # load data into the child node
        parent = Required(size=entry['size'], type=TypeEnum(entry['type']).value, nunique=entry['nunique'],
                          category=TypeCategory(entry['category']).value,
                          ordinal=Ternary(entry['ordinal']).value, num_na=entry['num_na'],
                          perc_na=entry['perc_na'])

        child1 = RequiredForNumeric(min=entry['min'], max=entry['max'], mean=entry['mean'], std_dev=entry['std'],
                                    hist_bins=str(entry['hist_bins']), hist_counts=str(entry['hist_counts']),
                                    normal=True, owner=parent)
        sess.add(parent)
        sess.add(child1)
        if entry['category'] == TypeCategory.type_categorical.value or \
                entry['category'] == TypeCategory.type_binary.value or \
                entry['category'] == TypeCategory.type_constant.value:
            child2 = ForEnumerable(unique_vals=entry['unique_vals'], counts=entry['counts'], owner=parent)
            sess.add(child2)

    # input type str or list
    if entry['type'] == TypeEnum.type_str.value or entry['type'] == TypeEnum.type_list.value:
        parent = Required(size=entry['size'], type=TypeEnum(entry['type']).value, nunique=entry['nunique'],
                          category=TypeCategory(entry['category']).value,
                          ordinal=Ternary(entry['ordinal']).value, num_na=entry['num_na'],
                          perc_na=entry['perc_na'])

        child = ForStrList(len_min=entry['len_min'], len_max=entry['len_max'], owner=parent)
        sess.add(parent)
        sess.add(child)
        if entry['category'] == TypeCategory.type_categorical.value or \
                entry['category'] == TypeCategory.type_binary.value or \
                entry['category'] == TypeCategory.type_constant.value:
            child = ForEnumerable(unique_vals=str(entry['unique_vals']), counts=str(entry['counts']), owner=parent)
            sess.add(child)

    # for date time:
    if entry['type'] == TypeEnum.type_date.value:
        parent = Required(size=entry['size'], type=TypeEnum(entry['type']).value, nunique=entry['nunique'],
                          category=TypeCategory(entry['category']).value,
                          ordinal=Ternary(entry['ordinal']).value, num_na=entry['num_na'],
                          perc_na=entry['perc_na'])
        child1 = ForDateTime(min_dt=entry['min'], max_dt=entry['max'], timedelta_mean=entry['timedelta_mean']
                             , timedelta_std=entry['timedelta_std'], owner=parent)
        sess.add(parent)
        sess.add(child1)
        if entry['category'] == TypeCategory.type_categorical.value or \
                entry['category'] == TypeCategory.type_binary.value or \
                entry['category'] == TypeCategory.type_constant.value:
            child2 = ForEnumerable(unique_vals=entry['unique_vals'], counts=entry['counts'], owner=parent)
            sess.add(child2)

    sess.commit()
    sess.close()
