import subprocess
import sys
import sklearn
import umap.umap_ as umap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# import xgboost as xgb  # 未使用，已注释
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
# from xgboost import DMatrix, train as xgb_train  # 未使用，已注释
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy.optimize import approx_fprime



def install_and_import(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        __import__(package)


# List of equivalent Python packages
python_packages = [
    "plotnine",  # ggplot2 equivalent
    "pandas",  # part of tidyverse equivalent
    "matplotlib", "seaborn",  # part of cowplot, ggpubr, ggsci equivalents
    # "scikit-learn", # part of glmnet, e1071, caret, class equivalents
    # "xgboost",  # 未使用，已注释
    "numpy", "scipy"

]

# Loop through the list and apply the function
for pkg in python_packages:
    install_and_import(pkg)



# 未使用的分类器函数（仅被eval_classifier调用，但eval_classifier未被使用）
# def LOGIS(train_data, train_labels, test_data, test_labels):
#     ... (函数体已注释)
#     pass

# def SVM(train_data, train_labels, test_data, test_labels):
#     ... (函数体已注释)
#     pass

# def KNN(train_data, train_labels, test_data, test_labels):
#     ... (函数体已注释)
#     pass

# def RF(train_data, train_labels, test_data, test_labels):
#     ... (函数体已注释)
#     pass

# def XGB(train_data, train_labels, test_data, test_labels):
#     ... (函数体已注释)
#     pass


# 未使用的函数：eval_classifier（在notebook中导入但未调用）
# def eval_classifier(whole_generated, whole_groups, n_candidate, n_draw=5, log=True, methods=None):
#     ... (函数体已注释，约100行代码)
#     pass


def heatmap_eval(dat_real,dat_generated=None):
    r"""
    This function creates a heatmap visualization comparing the generated data and the real data.
    dat_generated is applicable only if 2 sets of data is available.

    Parameters
    -----------
    dat_real: pd.DataFrame
            the original copy of the data
    dat_generated : pd.DataFrame, optional
            the generated data
    
    """
    if dat_generated is None:
        # Only plot dat_real if dat_generated is None
        plt.figure(figsize=(6, 6))
        sns.heatmap(dat_real, cbar=True)
        plt.title('Real Data')
        plt.xlabel('Features')
        plt.ylabel('Samples')
    else:
        # Plot both dat_generated and dat_real side by side
        fig, axs = plt.subplots(ncols=2, figsize=(12, 6),
                                gridspec_kw=dict(width_ratios=[0.5, 0.55]))

        sns.heatmap(dat_generated, ax=axs[0], cbar=False)
        axs[0].set_title('Generated Data')
        axs[0].set_xlabel('Features')
        axs[0].set_ylabel('Samples')

        sns.heatmap(dat_real, ax=axs[1], cbar=True)
        axs[1].set_title('Real Data')
        axs[1].set_xlabel('Features')
        axs[1].set_ylabel('Samples')



def UMAP_eval(dat_generated, dat_real, groups_generated, groups_real, random_state = 42, legend_pos="top"):
    r"""
    This function creates a UMAP visualization comparing the generated data and the real data.
    If only 1 set of data is available, dat_generated and groups_generated should have None as inputs.

    Parameters
    -----------
    dat_generated : pd.DataFrame
            the generated data, input None if unavailable
    dat_real: pd.DataFrame
            the original copy of the data
    groups_generated : pd.Series
            the groups generated, input None if unavailable
    groups_real : pd.Series
            the real groups
    legend_pos : string
            legend location
    
    """

    if dat_generated is None and groups_generated is None:
        # Only plot the real data
        reducer = UMAP(random_state=random_state)
        embedding = reducer.fit_transform(dat_real.values)

        umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
        umap_df['Group'] = groups_real.astype(str)  # Ensure groups are hashable for seaborn

        # Plotting
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', style='Group', palette='bright')
        plt.legend(title='Group', loc=legend_pos)
        plt.title('UMAP Projection of Real Data')
        plt.show()
        return
    
    # Filter out features with zero variance in generated data
    non_zero_var_cols = dat_generated.var(axis=0) != 0

    # Use loc to filter columns by the non_zero_var_cols boolean mask
    dat_real = dat_real.loc[:, non_zero_var_cols]
    dat_generated = dat_generated.loc[:, non_zero_var_cols]

    # Combine datasets
    combined_data = np.vstack((dat_real.values, dat_generated.values))  
    combined_groups = np.concatenate((groups_real, groups_generated))
    combined_labels = np.array(['Real'] * dat_real.shape[0] + ['Generated'] * dat_generated.shape[0])

    # Ensure that group labels are hashable and can be used in seaborn plots
    combined_groups = [str(group) for group in combined_groups]  # Convert groups to string if not already

    # UMAP dimensionality reduction
    reducer = UMAP(random_state=random_state)
    embedding = reducer.fit_transform(combined_data)

    umap_df = pd.DataFrame(embedding, columns=['UMAP1', 'UMAP2'])
    umap_df['Data Type'] = combined_labels
    umap_df['Group'] = combined_groups

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='Data Type', style='Group', palette='bright')
    plt.legend(title='Data Type/Group', loc="best")
    plt.title('UMAP Projection of Real and Generated Data')
    plt.show()



# 未使用的函数：power_law和fit_curve（仅被vis_classifier调用，但vis_classifier未被使用）
# def power_law(x, a, b, c):
#     return (1 - a) - (b * (x ** c))
# 
# def fit_curve(acc_table, metric_name, n_target=None, plot=True, ax=None, annotation=("Metric", "")):
#     ... (函数体已注释，约35行代码)
#     pass
    

def get_data_metrics(real_file_name, generated_file_name):
    """
    Load and preprocess real and generated datasets for downstream evaluation.

    Parameters
    ----------
    real_file_name : str
        Path to the CSV file containing real data.
    generated_file_name : str
        Path to the CSV file containing generated data.

    Returns
    -------
    real_data : pd.DataFrame
        Log2-transformed real data feature matrix.
    groups_real : pd.Series
        Binary-encoded group labels for real data (0/1).
    generated_data : pd.DataFrame
        Feature matrix for generated data (no transformation).
    groups_generated : pd.Series
        Group labels from generated data (as-is, assumed last column).
    unique_types : np.ndarray
        Array of unique binary group values (after mapping, i.e., [0, 1]).
    """
    # Load real dataset and drop non-feature column
    real = pd.read_csv(real_file_name, header=0)
    real.drop(columns='samples', inplace=True)

    # Load generated dataset and assign same column names
    generated = pd.read_csv(generated_file_name, header=None, names=real.columns)
    
    unique_types = real['groups'].unique()
    # Consistently encode the first and second group as 0 and 1
    if not np.issubdtype(real['groups'].dtype, np.number):
        type_map = {unique_types[0]: 0, unique_types[1]: 1}
        real['groups'] = real['groups'].map(type_map)
    
    # Extract group labels
    groups_real = real.groups
    groups_generated = generated.groups
    
    # Extract feature matrices
    real_data = real.iloc[:, :-1]
    real_data = np.log2(real_data + 1)  # Log-transform real data
    generated_data = generated.iloc[:, :-1]
    unique_types = real['groups'].unique() 

    # Return processed matrices and labels
    return real_data, groups_real, generated_data, groups_generated, unique_types



def visualize(real_data, groups_real, unique_types, generated_data=None, groups_generated=None, ratio=1, seed=42):
    """
    Visualize real and optionally generated data using heatmap and UMAP projections.

    Supports both binary and multi-class settings. For each class, samples from both datasets
    are drawn based on real data class proportions.

    Parameters
    ----------
    real_data : pd.DataFrame
        Feature matrix of real dataset (without 'groups' column).
    groups_real : pd.Series
        Group labels for the real dataset.
    unique_types : array-like
        Unique class labels to iterate over.
    generated_data : pd.DataFrame, optional
        Feature matrix of generated dataset (same columns as real_data).
    groups_generated : pd.Series, optional
        Group labels for the generated dataset.
    ratio : float, default=1
        Sampling ratio within each class (based on real data).
    seed : int, default=42
        Random seed for reproducibility.
    """
    np.random.seed(seed)

    real_indices = []
    generated_indices = []

    for group in unique_types:
        # Sample from real
        real_idx = np.where(groups_real == group)[0]
        n_sample = round(len(real_idx) * ratio)
        sampled_real = np.random.choice(real_idx, size=n_sample, replace=False)
        real_indices.extend(sampled_real.tolist())

        # Sample from generated if provided
        if generated_data is not None and groups_generated is not None:
            gen_idx = np.where(groups_generated == group)[0]
            if len(gen_idx) < n_sample:
                raise ValueError(f"Not enough samples in generated data for group '{group}'")
            sampled_gen = np.random.choice(gen_idx, size=n_sample, replace=False)
            generated_indices.extend(sampled_gen.tolist())

    # Heatmap
    if generated_data is None:
        heatmap_eval(dat_real=real_data.iloc[real_indices, :])
    else:
        heatmap_eval(
            dat_real=real_data.iloc[real_indices, :],
            dat_generated=generated_data.iloc[generated_indices, :]
        )

        # UMAP
        UMAP_eval(
            dat_real=real_data.iloc[real_indices, :],
            dat_generated=generated_data.iloc[generated_indices, :],
            groups_real=groups_real.iloc[real_indices],
            groups_generated=groups_generated.iloc[generated_indices],
            legend_pos="bottom"
        )




# 未使用的函数：vis_classifier（在notebook中导入但未调用）
# def vis_classifier(metric_real, n_target, metric_generated=None, metric_name='f1_score', save = False):
#     ... (函数体已注释，约55行代码)
#     pass
