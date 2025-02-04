import numpy as np

def error_metrics(pred, ref):
    """
    Calculates metrics such as ME, MAE, SD
    """
    me = np.mean(pred - np.squeeze(ref))
    print(f"Mean Error: {me}")
    mae = np.mean(np.abs(pred - np.squeeze(ref)))
    print(f"Mean Absolute Error: {mae}")
    std = np.std(pred - np.squeeze(ref))
    print(f"Standard Deviation: {std} \n")