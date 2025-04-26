from river import datasets, linear_model, metrics
from expectations import eprocesses

def assess_predictor(dataset, model, metric, alpha, e_proc):
    for i, (x, y) in enumerate(dataset):
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        metric.update(y, y_pred)
    
        # Convert prediction to binary (0/1)
        y_pred_bin = int(y_pred == 1)
        y_true_bin = int(y == 1)
    
        # Update e-process with outcome: was prediction correct? (1 if correct, 0 if not)
        correct = int(y_pred_bin == y_true_bin)
        e_proc.update(correct)
    
        # Print running accuracy and e-process value
        if (i + 1) % 100 == 0:
            print(f"Sample {i+1}, Accuracy: {metric.get():.3f}, E-value: {e_proc.e:.4f}")
    
        # Sequential test: reject null if e-process exceeds 1/alpha
        if e_proc.e >= 1/alpha:
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
# eprocesses.BernoulliEProcess expects a stream of 0/1 outcomes
alpha = 0.05  # significance level
e_proc = eprocesses.BernoulliEProcess(null=0.5, alpha=alpha)

# 3. Stream data, make predictions, and update e-process
assess_predictor(dataset, model, metric, alpha, e_proc)
