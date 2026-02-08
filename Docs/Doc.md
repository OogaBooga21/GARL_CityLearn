# CityLearn Project Documentation

## CityLearn Dataset Location

The CityLearn datasets are typically located within the installed `citylearn` Python package.
A common path for these datasets is:

`/home/oli/.local/share/virtualenvs/EC_RL_py311/lib/python3.11/site-packages/citylearn/data`

When using `citylearn.data.DataSet(dataset_name).get_env()`, CityLearn automatically finds and loads the necessary data from this location (or a similar path depending on your Python environment). You do not need to manually copy these files into your project directory.

from citylearn.data import DataSet
print(DataSet().get_dataset_names())
-- to get a list of all availble datasets.


