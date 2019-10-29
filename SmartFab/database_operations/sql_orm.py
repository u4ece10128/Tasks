###############################################################
# Author: Jaya Ram Kollipara                                  #
# Task: Smarfab_Q1                                            #
# Date: 08/10/2019                                            #
# SQLALchemy Version: 1.2.1                                   #
# Details: Implement DB Schema:                               #
#                 Parent Table:                               #
#                              1. Required                    #
#                 Child table:                                #
#                              1. Required for Numeric        #
#                              2. Enumerable                  #
#                              3. For Str or list             #
#                              4. For Date Time               #
# Input Type:  Dictionary                                     #
# Output: MySQL DB Setup with Schema under db = my_db         #
###############################################################


import enum
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Enum, ForeignKey, ARRAY, TEXT
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
#
# Base = declarative_base()


class Base(object):
    id = Column(Integer, primary_key=True)

Base = declarative_base(cls=Base)


class Ternary(enum.Enum):
    FALSE = False
    TRUE = True
    UNKNOWN = None


class TypeCategory(enum.Enum):
    type_non_categorical = "non-categorical"
    type_categorical = "categorical"
    type_binary = "binary"
    type_constant = "constant"


class TypeEnum(enum.Enum):
    type_float = "float"
    type_int = "integer"
    type_str = "string"
    type_date = "datetime"
    type_list = "list"


class Required(Base):
    __tablename__ = "required"
    __table_args__ = {'extend_existing': True}
    size = Column("size", Integer, primary_key=False)
    type = Column("type", String(20), primary_key=False)
    nunique = Column("nunique", Integer, primary_key=False)
    category = Column("category", String(20), primary_key=False)
    ordinal = Column("ordinal", String(20), primary_key=False)
    num_na = Column("num_na", Integer, primary_key=False)
    perc_na = Column("perc_na", Float, primary_key=False)

    children1 = relationship("RequiredForNumeric", backref="owner")
    children2 = relationship("ForEnumerable", backref="owner")
    children3 = relationship("ForStrList", backref="owner")
    children4 = relationship("ForDateTime", backref="owner")

    def set_attributes(self, size, type, nunique, category, ordinal, num_na, perc_na):
        self.size = size
        self.type = type
        self.nunique = nunique
        self.category = category
        self.ordinal = ordinal
        self.num_na = num_na
        self.perc_na = perc_na


class RequiredForNumeric(Base):
    __tablename__ = "required_for_numberic"
    __table_args__ = {'extend_existing': True}

    min = Column("min", Float)
    max = Column("max", Float)
    mean = Column("mean", Float)
    std_dev = Column("std_dev", Float)
    hist_bins = Column("hist_bins", TEXT)
    hist_counts = Column("hist_counts", TEXT)
    normal = Column("normal", Boolean)

    # Relationship
    parent_id = Column(Integer, ForeignKey('required.id'), nullable=False)

    def set_attributes(self, min, max, mean, std_dev, hist_bins, hist_counts, normal):
        self.min = min
        self.max = max
        self.mean = mean
        self.std_dev = std_dev
        self.hist_bins = hist_bins
        self.hist_counts = hist_counts
        self.normal = normal


class ForEnumerable(Base):
    __tablename__ = "for_enumerable"
    __table_args__ = {'extend_existing': True}
    unique_vals = Column("unique_vals", TEXT, nullable=True)
    counts = Column("counts", TEXT, nullable=True)
    mode = Column("mode", Integer, nullable=True)
    mode_perc = Column("mode_perc", Float, nullable=True)

    # Relationship
    parent_id = Column(Integer, ForeignKey('required.id'), nullable=False)

    def set_attributes(self, unique_vals, counts, mode, mode_perc):
        self.unique_vals = unique_vals
        self.counts = counts
        self.mode = mode
        self.mode_perc = mode_perc


class ForStrList(Base):
    __tablename__ = "for_str_list"
    __table_args__ = {'extend_existing': True}
    # id_parent = Column("id_parent", Integer, primary_key=True)
    len_min = Column("len_min", Integer)
    len_max = Column("len_max", Integer)

    def set_attributes(self, len_min, len_max):
        self.len_min = len_min
        self.len_max = len_max

    # Relationship
    parent_id = Column(Integer, ForeignKey('required.id'), nullable=False)


class ForDateTime(Base):
    __tablename__ = "for_date_time"
    __table_args__ = {'extend_existing': True}
    min_dt = Column("min_dt", DateTime)
    max_dt = Column("max_dt", DateTime)
    timedelta_mean = Column("timedelta_mean", DateTime)
    timedelta_std = Column("timedelta_std", DateTime)

    # Relationship
    parent_id = Column(Integer, ForeignKey('required.id'), nullable=False)

    def set_attributes(self, min_dt, max_dt):
        self.min_dt = min_dt
        self.max_dt = max_dt

