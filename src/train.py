import argparse
import json
import os

import boto3
import mlflow
import mlflow.gluon
import numpy as np
import pandas as pd
from mxnet import gpu, cpu
from mxnet.gluon import HybridBlock, Trainer
from mxnet.gluon.contrib.estimator import estimator
from mxnet.gluon.data import ArrayDataset, DataLoader
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import HybridSequential, Dense, Dropout
from mxnet.initializer import Xavier
from mxnet.metric import Accuracy
from pymysql import converters


def train(args: argparse.Namespace) -> HybridBlock:
    session = boto3.session.Session()

    client = session.client(service_name="secretsmanager",
                            region_name="us-east-1")
    mlflow_secret = client.get_secret_value(SecretId=args.mlflow_secret)
    mlflowdb_conf = json.loads(mlflow_secret["SecretString"])

    converters.encoders[np.float64] = converters.escape_float
    converters.conversions = converters.encoders.copy()
    converters.conversions.update(converters.decoders)

    mlflow.set_tracking_uri(f"mysql+pymysql://{mlflowdb_conf['username']}:{mlflowdb_conf['password']}@{mlflowdb_conf['host']}/mlflow")

    if mlflow.get_experiment_by_name(args.mlflow_experiment) is None:
        mlflow.create_experiment(args.mlflow_experiment,
                                 args.mlflow_artifacts_location)
    mlflow.set_experiment(args.mlflow_experiment)

    col_names = ["target"] + [f"kinematic_{i}" for i in range(1, 22)]

    train_df = pd.read_csv(f"{args.train_channel}/train.csv.gz",
                           header=None, names=col_names)

    val_df = pd.read_csv(f"{args.validation_channel}/val.csv.gz",
                         header=None, names=col_names)

    train_X = train_df.drop("target", axis=1)
    train_y = train_df["target"]
    train_dataset = ArrayDataset(train_X.to_numpy(dtype="float32"),
                                 train_y.to_numpy(dtype="float32"))
    train = DataLoader(train_dataset, batch_size=args.batch_size)

    val_X = val_df.drop("target", axis=1)
    val_y = val_df["target"]
    val_dataset = ArrayDataset(val_X.to_numpy(dtype="float32"),
                               val_y.to_numpy(dtype="float32"))
    validation = DataLoader(val_dataset, batch_size=args.batch_size)

    ctx = [gpu(i) for i in range(args.gpus)] if args.gpus > 0 else cpu()

    mlflow.gluon.autolog()

    with mlflow.start_run():
        net = HybridSequential()
        with net.name_scope():
            net.add(Dense(256))
            net.add(Dropout(.2))
            net.add(Dense(64))
            net.add(Dropout(.1))
            net.add(Dense(16))
            net.add(Dense(2))

        net.initialize(Xavier(magnitude=2.24), ctx=ctx)
        net.hybridize()

        trainer = Trainer(net.collect_params(),
                          "sgd",
                          {"learning_rate": args.learning_rate})
        est = estimator.Estimator(net=net,
                                  loss=SoftmaxCrossEntropyLoss(),
                                  trainer=trainer,
                                  train_metrics=Accuracy(),
                                  context=ctx)
        est.fit(train, epochs=args.epochs, val_data=validation)

    return net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training configuration
    parser.add_argument("--train-channel", type=str,
                        default=os.getenv("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation-channel", type=str,
                        default=os.getenv("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--gpus", type=int,
                        default=os.getenv("SM_NUM_GPUS"))

    # MLflow configuration
    parser.add_argument("--mlflow-secret", type=str)
    parser.add_argument("--mlflow-artifacts-location", type=str)
    parser.add_argument("--mlflow-experiment", type=str,
                        default="higgs-bosons-process-classification")

    # Actual hyperparameters
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=4)

    arguments = parser.parse_args()
    train(arguments)
