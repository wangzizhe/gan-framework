# import optuna and MLflow
# Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning.
# MLflow is an open source platform for the machine learning lifecycle.
import optuna
from optuna.integration.mlflow import MLflowCallback

# import all functions from train.py
from train import *

def objective(trial, kwargs):

    # hyperparameters to be optimized
    # more hyperparameters can be added here in this format
    kwargs["cfg"] = trial.suggest_categorical("cfg", ["auto", "stylegan2", "paper1024"])
    kwargs["mirror"] = trial.suggest_categorical("mirror", ["0", "1"])
    kwargs["gamma"] = trial.suggest_int("gamma", 5, 15)
    kwargs["target"] = trial.suggest_float("target", 0.5, 0.7)
    kwargs["augpipe"] = trial.suggest_categorical("augpipe", ["blit","geom","color","filter","noise","cutout","bg","bgc","bgcf", "bgcfn", "bgcfnc"])

# set the main function
if __name__ == "__main__":
    kwargs = {} # read the hyperparameters which are typed in

    main()  # run main function of train.py

    # class optuna.integration.MLflowCallback(tracking_uri=None, metric_name='value', nest_trials=False, tag_study_user_attrs=False)
    # for more details of "tracking_uri" please refer to https://optuna.readthedocs.io/en/stable/reference/generated/optuna.integration.MLflowCallback.html
    # and https://www.mlflow.org/docs/latest/python_api/mlflow.html#mlflow.set_tracking_uri
    mlflc = MLflowCallback(metric_name="fid50k_full")   # set MLflow

    study = optuna.create_study(direction="minimize")   # direction is to minimize the metric

    study.optimize(lambda trial: objective(trial, kwargs), n_trials=20, callbacks=[mlflc])  # set hyperparameter optimize
