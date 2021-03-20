import os
import torch

from src.evaluation_metrics import EvaluationMetrics
from src.model import RegressionModel
from src.independence_test import HilbertSchmidtInformationCriterion
from src.data_process import DataPreProcess
from src.constant import Paths

if __name__ == "__main__":

    predictions = []

    for idx, filename in enumerate(sorted(os.listdir(Paths.RAW_DATA_RELATIVE_PATH))):

        model_num = int(str(filename.split("pair")[1]).split(".txt")[0])
        # Get test data, correct test will be obtain thanks to fixed seed
        dataset = DataPreProcess.read_dataset(filename)
        _, _, x_test, y_test = DataPreProcess.get_test_train_split(dataset)

        # Load Model Weights
        model_x = RegressionModel.build_model()
        model_x.load_state_dict(torch.load(Paths.MODEL_WEIGHTS+"model_x_"+str(model_num)))
        model_x.eval()

        model_y = RegressionModel.build_model()
        model_y.load_state_dict(torch.load(Paths.MODEL_WEIGHTS + "model_y_" + str(model_num)))
        model_y.eval()

        # Get residuals for each causal direction
        residual_x = RegressionModel.get_residuals(model_x, x_test, y_test)
        residual_y = RegressionModel.get_residuals(model_y, y_test, x_test)

        # Get independence score
        hsic_x = HilbertSchmidtInformationCriterion.hsic_score(residual_x, x_test)
        hsic_y = HilbertSchmidtInformationCriterion.hsic_score(residual_y, y_test)

        # Prediction
        predictions.append(HilbertSchmidtInformationCriterion.predict_causal_relation(hsic_x, hsic_y))

    # Calculate evaluation metrics
    accuracy = EvaluationMetrics.accuracy(predictions)
    auc = EvaluationMetrics.auc(predictions)

    print("Accuracy: {}%".format(accuracy*100))
    print("AUC: {}".format(auc))
