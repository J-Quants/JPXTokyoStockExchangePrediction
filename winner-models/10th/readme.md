Hello!

Below you can find a outline of how to reproduce my solution for the JPX Tokyo Stock Exchange Prediction competition.
If you run into any trouble with the setup/code or have any questions please contact me at giulio.rav@gmail.com

# ARCHIVE CONTENTS
documentation/ Model Documentation - JPX.pdf   : High-level documentation of the solution
notebooks/simulations                          : Folder containing the notebooks for the analysis simulations performed
notebooks/submission                           : Folder containing the notebook to be run on Kaggle to submit the solution

# HARDWARE: (The following specs were used to create the original solution)
Standar Kaggle Notebook Enviroment (with CPU only)

# SOFTWARE (python packages are detailed separately in `requirements.txt`):
Python 3.7.12

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
This notebooks needs to run inside Kaggle Competition environment, no additional data setup is required

# NOTEBOOKS
### notebooks/simulations/simulations.ipynb
In this notebook, the 100000 portfolio simulations of each day of the history are generated and saved in the folder "jpx-solution/notebooks/simulations/days_montecarlo_simulations".

### notebooks/simulations/simulation_aggregation.ipynb
In this notebook, the simulations of each day are aggregated and the best spread to achieve to maximise the historical sharpe ratio

### notebooks/submission/submission-notebook.ipynb
This is the notebook with all the model logic inside for use within the Kaggle environment of the competition