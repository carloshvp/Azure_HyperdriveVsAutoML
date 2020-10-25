from azureml.widgets import RunDetails
from azureml.train.sklearn import SKLearn
from azureml.train.hyperdrive.run import PrimaryMetricGoal
from azureml.train.hyperdrive.policy import BanditPolicy
from azureml.train.hyperdrive.sampling import RandomParameterSampling
from azureml.train.hyperdrive.runconfig import HyperDriveConfig
from azureml.train.hyperdrive.parameter_expressions import uniform
import os

# Specify parameter sampler
ps = RandomParameterSampling(
    {
        '--C': choice(0.001, 0.005, 0.01, 0.05, 0.1, 1.0), # Regularization
        '--max_iter': choice(50, 100, 200, 300) # Max number of iterations (aka epochs)
    }
)

# Specify a Policy
policy = BanditPolicy(slack_factor = 0.1, evaluation_interval=1, delay_evaluation=5)

if "training" not in os.listdir():
    os.mkdir("./training")

# Create a SKLearn estimator for use with train.py
est = SKLearn(source_directory="./training",
                          inputs=[], # TODO: Something to add here
                          pip_packages=['azureml-sdk'], # ...so we need azureml-dataprep (it's in the SDK!)
                          entry_script='train.py',
                          compute_target = training_cluster,)

# Create a HyperDriveConfig using the estimator, hyperparameter sampler, and policy.
hyperdrive_config = ### YOUR CODE HERE ###
