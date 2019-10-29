##########################################################################################################
# Author: Jaya Ram Kollipara                                                                             #
# Task: Smarfab_Q1                                                                                       #
# Date: 09/10/2019                                                                                       #
# Details: db_utils- Supported Features:                                                                 #
#                                  1. Create Connection to Data Base                                     #
#                                  1. Get all variables with nunique < 10                                #
#                                  2. Get all variables where category == categorical and perc_na < 0.1  #
##########################################################################################################


from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Integer, Float, String
from sqlalchemy.ext.declarative import declarative_base


class Base(object):
    id = Column(Integer, primary_key=True)


Base = declarative_base(cls=Base)


class Required(Base):
    """
    Object : Required
    Table: Required
    """
    __tablename__ = "required"
    __table_args__ = {'extend_existing': True}
    size = Column("size", Integer, primary_key=False)
    type = Column("type", String(20), primary_key=False)
    nunique = Column("nunique", Integer, primary_key=False)
    category = Column("category", String(20), primary_key=False)
    ordinal = Column("ordinal", String(20), primary_key=False)
    num_na = Column("num_na", Integer, primary_key=False)
    perc_na = Column("perc_na", Float, primary_key=False)

    def __repr__(self):
        """

        :return: List of Queries in the following format ---> List
        """
        return "<id='%s', size='%s', type='%s', nunique='%s', category='%s', " \
               ",ordinal='%s', num_na='%s', perc_na='%s'>" % (self.id,
                                                              self.size,
                                                              self.type,
                                                              self.nunique,
                                                              self.category,
                                                              self.ordinal,
                                                              self.num_na,
                                                              self.perc_na)


class db_utils:
    """
    Data Base Utilities Class. Methods include Connection to the Data Base specified in a URL, Two Data Base Query types
    """
    def __init__(self, url):
        self.url = url
        self.driver = None

    def connect(self):
        """
        Connects to Data Base
        :return: DB Cursor Object  ---> Session
        """
        if not self.driver:
            engine = create_engine(self.url)
            session = sessionmaker(bind=engine)
            self.driver = session()
        return self.driver

    # API
    def get_nunique_lt_10(self):
        """
        Get all the variables with nunique < 10
        :return: List of Required Objects present   ---> List
        """
        # sqlalchemy query syntax
        # query() loads Required Instances
        # filter() Required Instance is filtered with a condition
        return self.connect().query(Required).filter(Required.nunique < 10).all()

    def get_by_category_categorical_perc_na(self):
        """
        Get all the Variables with category == 'categorical' and perc_na < 0.1
        :return:  List of Required Objects present   ---> List
        """
        # sqlalchemy query syntax
        # query() loads Required Instances
        # filter() Required Instance is filtered with a condition
        return self.connect().query(Required).filter(Required.category == 'categorical',
                                                     Required.perc_na < 0.1).all()


def create_connector():
    """
    :return: Connection to the destined DB
    """
    return db_utils('mysql+pymysql://root:root1234@localhost:3306/my_db')
