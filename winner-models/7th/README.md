# JPX Kaggle Challenge 8th Place Solution Code
Note: JPX data has to be placed in a directory called /input/jpx-tokyo-stock-exchange-prediction/. It is the same structure as on the Kaggle competition site. 
## Hardware
Besides the Kaggle resources, we used a Intel E5-2620v4 CPU and a Nvidia GTX 1080 TI as GPU to test our approaches.

## OS
All experiments were executed under Debian buster Version 10

## Setup Conda Environment:

```sh
conda env create -f environment.yml
conda activate jpx
```

All results on the public testset from the supplementary files can be reproduced with the following bash commands executed from the root directory.

```sh
mkdir ./results # create dir to save results
python evaluate_models.py -m lgbmhierarch -p ./results -dp input/jpx-tokyo-stock-exchange-prediction/
```

## Training 
The model can be trained using the `evaluation.ipynb`notebook or by creating a new file and executing the following code:

```py
from models import LGBMHierarchModel

train, test = data_pipeline(dir_path="./input/jpx-tokyo-stock-exchange-prediction")
seed = 69
model = LGBMHierarchModel(seed=seed)
model.train(train=train, use_params=True)
```

## Testing
Initialize the model as described above and use the `predict` method of the model to measure performance on test data.

```py
performance = model.predict(test, {"rmse":(mean_squared_error, {"squared":False})})
```