import os
import torch

from src.constant import Paths, ModelConstants
from src.data_process import DataPreProcess
from src.model import RegressionModel, Train
from src.utils import TrainingUtils
from src.independence_test import HilbertSchmidtInformationCriterion
from src.evaluation_metrics import EvaluationMetrics

if __name__ == "__main__":
    predictions = []
    TrainingUtils.start()
    for idx, filename in enumerate(sorted(os.listdir(Paths.RAW_DATA_RELATIVE_PATH))):

        TrainingUtils.counter(int(str(filename.split("pair")[1]).split(".txt")[0]))
        # Get train data, same train data will obtain thanks to fixed seed
        dataset = DataPreProcess.read_dataset(filename)
        x_train, y_train, x_test, y_test = DataPreProcess.get_test_train_split(dataset)

        # Build Models
        model_x = RegressionModel.build_model()
        model_x.apply(RegressionModel.init_weights)
        optimizer_x = torch.optim.Adam(model_x.parameters(), ModelConstants.LEARNING_RATE)

        model_y = RegressionModel.build_model()
        model_y.apply(RegressionModel.init_weights)
        optimizer_y = torch.optim.Adam(model_y.parameters(), ModelConstants.LEARNING_RATE)

        # Train models and save weights
        Train.train_loop(RegressionModel.get_num_of_epoch(len(x_train)), x_train, y_train, optimizer_x, model_x)
        torch.save(model_x.state_dict(), Paths.MODEL_WEIGHTS+"model_x_"+str(idx + 1))

        Train.train_loop(RegressionModel.get_num_of_epoch(len(y_train)), y_train, x_train, optimizer_y, model_y)
        torch.save(model_y.state_dict(), Paths.MODEL_WEIGHTS+"model_y_"+str(idx + 1))

        # Get residuals for each causal direction
        residual_x = RegressionModel.get_residuals(model_x, x_test, y_test)
        residual_y = RegressionModel.get_residuals(model_y, y_test, x_test)

        # Get independence score
        hsic_x = HilbertSchmidtInformationCriterion.hsic_score(residual_x, x_test)
        hsic_y = HilbertSchmidtInformationCriterion.hsic_score(residual_y, y_test)

        # Prediction
        predictions.append(HilbertSchmidtInformationCriterion.predict_causal_relation(hsic_x, hsic_y))

    TrainingUtils.complete()

    # Calculate evaluation metrics
    accuracy = EvaluationMetrics.accuracy(predictions)
    auc = EvaluationMetrics.auc(predictions)

    print("Accuracy: {}%".format(accuracy * 100))
    print("AUC: {}".format(auc))

