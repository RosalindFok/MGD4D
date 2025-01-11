import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nilearn import plotting

from path import Path, Paths


def plot_heap_map(matrix : np.array, saved_dir_path : Path, fig_name : str = "correlation_matrix.svg") -> None:
    # plot heatmap
    plt.figure(figsize=(10, 10))
    plt.imshow(matrix, cmap="RdBu_r", interpolation="nearest")  
    plt.colorbar()  
    plt.title("Heatmap")  
    plt.xlabel("X-axis")  
    plt.ylabel("Y-axis")  
    plt.savefig(saved_dir_path / fig_name, format="svg")  
    plt.close() 

def draw_atlas(atlas : nib.nifti1.Nifti1Image, saved_path : Path = Paths.Fig_Dir / "atlas.png"):
    glass_brain = plotting.plot_glass_brain(atlas, colorbar=False, plot_abs=False)
    glass_brain.savefig(saved_path)  
    glass_brain.close() 