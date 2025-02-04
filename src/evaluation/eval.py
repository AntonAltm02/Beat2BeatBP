import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

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

def corr_plot(pred, ref, train_type, pat_type, bp_type):
    """
    Plot the correlation plot
    """
    if ref.ndim == 2:
        ref = ref.squeeze(-1)
    plt.figure(figsize=(12, 8))
    plt.scatter(ref, pred, linewidths=0.5, s=5)

    # Add labels and title
    if train_type == "pat":
        plt.xlabel(f'Reference PAT({pat_type})')
        plt.ylabel(f'Predicted PAT({pat_type})')
        plt.title(f'Correlation Plot - PAT({pat_type})')
    else:
        plt.xlabel(f'Reference {bp_type} (mmHg)', fontsize=24)
        plt.ylabel(f'Predicted {bp_type} (mmHg)', fontsize=24)
        plt.title(f'Correlation Plot for {bp_type}', fontsize=26)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    # linear relationship
    z = np.polyfit(ref, pred, 1)
    p = np.poly1d(z)
    plt.plot(ref, p(ref), color='red')
    corr_coefficient, _ = scipy.stats.pearsonr(ref.flatten(), pred)
    plt.text(plt.xlim()[0] + 0.05 * (plt.xlim()[1] - plt.xlim()[0]),
             plt.ylim()[1] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]),
             f'R: {corr_coefficient:.2f}', fontsize=24, color="black", ha='left', va='top')

    # path = os.path.join("Plots", f"Vital_{self.bp_type}_BestCase_ePATit+AF_corr_plot.pdf")
    # plt.savefig(path, format='pdf')
    plt.show()