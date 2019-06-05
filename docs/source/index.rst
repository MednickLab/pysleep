.. mednickdb_pysleep documentation master file, created by
   sphinx-quickstart on Sun Jun  2 00:12:14 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to mednickdb_pysleep's documentation!
=============================================

pysleep is a module for calculating various measures, statistics and useful information about sleep records.
It can be used stand alone to act on raw files, and it is also called by the microservices of the MednickDB.

See the Manual_ for more information of the MednickDB, and the examples folder for some common usecases.

There are 5 included modules that a user will interface with, please see their API for more details:

   * :py:mod:`mednickdb_pysleep.frequency_features` - extract band power
   * :py:mod:`mednickdb_pysleep.scorefiles` - Parse sleep scoring files, split wake->waso, wbso, wase etc
   * :py:mod:`mednickdb_pysleep.sleep_dynamics` - extract sleep dynamics measures
   * :py:mod:`mednickdb_pysleep.sleep_features` - Algos for extracting spindles, slow osc, rem and assigning stages etc
   * :py:mod:`mednickdb_pysleep.sleep_architecture` - extract standard sleep measures like mins in stage, etc

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Manual: https://docs.google.com/document/d/18tD_ddjSYGFzIE07Uzoi0E3woXQ61TnQjqc4uXM4eXg/edit?usp=sharing
