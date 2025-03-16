import numpy as np
import seaborn as sns  
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting
from sklearn.metrics import confusion_matrix  

from path import Path, Paths


def plot_heap_map(matrix : np.ndarray, saved_dir_path : Path, fig_name : str = "correlation_matrix.svg") -> None:
    # plot heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap="RdBu_r", interpolation="nearest")  
    plt.colorbar()  
    plt.title("Heatmap")  
    plt.xlabel("X-axis")  
    plt.ylabel("Y-axis")  
    plt.savefig(saved_dir_path / fig_name, format="svg")  
    plt.close() 

def draw_atlas(atlas : nib.nifti1.Nifti1Image, saved_path : Path = Paths.Fig_Dir / "atlas.png") -> None:
    glass_brain = plotting.plot_glass_brain(atlas, colorbar=False, plot_abs=False)
    glass_brain.savefig(saved_path)  
    glass_brain.close() 

def plot_confusion_matrix(target : np.ndarray, prediction : np.ndarray, saved_path : str, figsize : tuple = (10, 8)) -> None:
    cm = confusion_matrix(target, prediction)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(saved_path)
    plt.close()