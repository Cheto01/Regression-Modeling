Regression modeling analysis
==============================

This project aims to present a study that elucidates the correlation between various parameters within the provided dataset. 
Specifically, the parameters we are examining include: sex, lang, country, age, first name, last name, hours studied, dojo class, test preparation, pass/fail status, and notes.

Our goal is to analyze the correlation between dependent and independent features through Exploratory Data Analysis(EDA) techniques, and to employ various models to predict student notes. 
For more detailed information, please refer to the annexed notebook: [EDA Notebook](https://github.com/Cheto01/Regression-Modeling/blob/master/notebooks/EDA.ipynb).


Project Organization. This repository was created using a cookie cutter template.
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources [EMPTY].
    │   ├── interim        <- Intermediate data that has been transformed [EMPTY].
    │   ├── processed      <- The final data [EMPTY].
    │   └── raw            <- The original, provided data.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. The exploratory data Analysis notebook is can be found here.
    │
    ├── references         <- Data dictionaries, provided explanatory materials.
    │
    ├── reports            <- Generated analysis .
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to load or process data
    │   │   └── data_processing.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations(TBD)
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

# Installation

Install the environment dependancies by running 

`pip install -r requirements.txt`


# main.py
---------------------------------------------------------------------
To run the code and test the training and the test accuracy, please, use thefollowing command:

`python main.py --train True --predict True --model RF --crossval True --scale_features True`

### Flags

* --`train` : Will train a new model if the flag is `True`, will test a trained one elsewhere. Before setting this flag to False, make sure the model specified the `model` flag exists. 
For instance, `--model DT` can be used as long as the following exists: `models/DT.pkl`.
* --`predict`: If the flag is `True`, it will perform a prediction on a test dataset and print out a test f1 accuracy score. it will also save the predicted label in `reports/model_output/`.
* --`model`: Will this flag, a user can choose which model to use from the following list ('`LR`', '`RF`', '`DT`', '`GNB`', '`KNN`', '`SVM`', '`AB`', '`BG`','`MLP` '`all`'). If `all` is used, the training and prediction will be perfomed by all the models.
* --`crossval`: This flag allows to use or not a cross validation in the training. Since the dataset is limited, this can help to increase the performance and the generalization.
* --`scale_features`: This flag allows to use or not a feature normalization. If set to `True` a feature normalization using `StandardScaler` is performed on `age` and `hours_studied`. This has a tiny better impact on the accuracy for this dataset.






