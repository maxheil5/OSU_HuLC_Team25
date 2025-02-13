import sagemaker
from sagemaker.pytorch import PyTorch
import os

# Initialize a SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()  # Modify if necessary

# Define S3 bucket and prefix (adjust according to your S3 structure)
bucket = 'satseg'  # Your S3 bucket name
prefix = 'train'   # Folder containing training data (with subfolders images_color/ and masks/)
s3_train_data = f's3://{bucket}/{prefix}'

# Define the PyTorch estimator
estimator = PyTorch(
    entry_point='train.py',
    role=role,
    framework_version='1.13.1',  # Specify your PyTorch version
    py_version='py38',
    instance_count=1,
    instance_type='ml.g6e.xlarge',  # Choose an appropriate instance type
    hyperparameters={
        'epochs': 20,
        'batch_size': 4,
        'learning_rate': 1e-4
    },
    output_path=f's3://{bucket}/satellite-segmentation/output',
    sagemaker_session=sagemaker_session
)

# Launch the training job
estimator.fit({'train': s3_train_data})
