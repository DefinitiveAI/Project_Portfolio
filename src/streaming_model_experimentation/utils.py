from type_annotations import Dataset, Metric, Estimator, 

from beartype import beartype
from river import datasets, linear_model, metrics
from expectation.seqtest.sequential_e_testing import SequentialTest, TestType
from expectation.modules.hypothesistesting import EValueConfig

@beartype
def assess_predictor(
    dataset: Dataset,
    model: Estimator,
    metric: Metric,
    test: SequentialTest,
    ) -> None:

    for i, (x, y) in enumerate(dataset):
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred)
    
        # Convert prediction to binary (0/1)
        y_pred_bin = int(y_pred == 1)
        y_true_bin = int(y == 1)
    
        # Update e-process with outcome: was prediction correct? (1 if correct, 0 if not)
        correct = int(y_pred_bin == y_true_bin)
        result = test.update([correct])
    
        # Print running accuracy and e-process value
        if (i + 1) % 100 == 0:
            print(f"Sample {i+1}, Accuracy: {metric.get():.3f}, E-value: {test.e:.4f}")
    
        # Sequential test: reject null if e-process exceeds 1/alpha
        if result.reject_null:
            print(f"\nNull hypothesis rejected after {i+1} samples!")
            print(f"Final Accuracy: {metric.get():.3f}")
            break
    else:
        print("\nStream ended without rejecting the null hypothesis.")
        print(f"Final Accuracy: {metric.get():.3f}")


"""
In general, for a binary predictor, a good rule of thumb for its predictive expressivity
is if it's able to correctly predict better than random selection.
The below showcases this with a representative phishing dataset and a logistic regressor
"""

# 1. Initialize streaming dataset and model
dataset = datasets.Phishing()  # Binary classification stream
model = linear_model.LogisticRegression()
metric = metrics.Accuracy()

# 2. Initialize e-process for sequential testing
# We'll test if the model's accuracy is better than random guessing (null: acc <= 0.5)
config = EValueConfig(significance_level=0.05, allow_infinite=False)
test = SequentialTest(
    test_type=TestType.MEAN,
    null_value=0.5,
    alternative="greater",
    config=config
)

# 3. Stream data, make predictions, and update e-process
assess_predictor(dataset, model, metric, test)
