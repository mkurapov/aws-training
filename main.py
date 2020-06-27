# S3 prefix
from sagemaker.sklearn.estimator import SKLearn
from sklearn import datasets
import os
import numpy as np
from sagemaker import get_execution_role
import sagemaker
prefix = 'Scikit-iris'

sagemaker_session = sagemaker.Session()


# Get a SageMaker-compatible role used by this Notebook Instance.
role = get_execution_role()

# Load Iris dataset, then join labels and features
iris = datasets.load_iris()
joined_iris = np.insert(iris.data, 0, iris.target, axis=1)

# Create directory and write csv
os.makedirs('./data', exist_ok=True)
np.savetxt('./data/iris.csv', joined_iris, delimiter=',',
           fmt='%1.1f, %1.3f, %1.3f, %1.3f, %1.3f')

WORK_DIRECTORY = 'data'

# train_input = sagemaker_session.upload_data(
#     WORK_DIRECTORY, key_prefix="{}/{}".format(prefix, WORK_DIRECTORY))

# script_path = 'scikit_learn_iris.py'

# sklearn = SKLearn(
#     entry_point=script_path,
#     train_instance_type="ml.c4.xlarge",
#     role=role,
#     sagemaker_session=sagemaker_session,
#     hyperparameters={'max_leaf_nodes': 30})

# sklearn.fit({'train': train_input})
