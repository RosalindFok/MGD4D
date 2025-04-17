# MGD4MD
Multi-modal Graph Diffusion for identifying neuroimaging biomarkers of mental disorder

## 1. Preprocess
Platform: Windows11, Python 3.11.4

``` shell
# dependencies
pip3 install numpy
pip install antspyx
pip install nibabel
pip install tqdm
pip install nilearn
pip install seaborn
pip install openpyxl
# script
python preprocess.py
```


## 2. Train
Platform: Beijing Super Cloud Computing Center - N32EA14P: `NVIDIA A100-PCIE-40GB * 8`

``` shell
# conda enviorment
module load cuda/12.1 miniforge  # N32EA14P
conda create --name MGD4MD python=3.11
source activate MGD4MD

# dependencies for MGD4MD
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Successfully installed MarkupSafe-2.1.5 filelock-3.13.1 fsspec-2024.6.1 jinja2-3.1.4 mpmath-1.3.0 networkx-3.3 numpy-2.1.2 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.1.105 nvidia-nvtx-cu12-12.1.105 pillow-11.0.0 sympy-1.13.1 torch-2.5.1+cu121 torchaudio-2.5.1+cu121 torchvision-0.20.1+cu121 triton-3.1.0 typing-extensions-4.12.2
pip install nilearn -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed certifi-2025.1.31 charset-normalizer-3.4.1 idna-3.10 importlib-resources-6.5.2 joblib-1.4.2 lxml-5.3.0 nibabel-5.3.2 nilearn-0.11.1 packaging-24.2 pandas-2.2.3 python-dateutil-2.9.0.post0 pytz-2025.1 requests-2.32.3 scikit-learn-1.6.1 scipy-1.15.1 six-1.17.0 threadpoolctl-3.5.0 tzdata-2025.1 urllib3-2.3.0
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed tqdm-4.67.1
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed contourpy-1.3.1 cycler-0.12.1 fonttools-4.55.8 kiwisolver-1.4.8 matplotlib-3.10.0 pyparsing-3.2.1
pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed seaborn-0.13.2


# script
python run.py  # run MGD4MD, sumit all folds in one stroke
python view.py # view average results of all folds
cat average_result.json # view results of all folds and their average values
python clear.py # clear the log file

conda deactivate
conda env remove -n MGD4MD
```

# Raw experimental results
## baselines
### ORC-GNN, 2024
> Li, Y., Dong, Y., Peng, S., Gao, L., & Xin, Y. (2025). ORC-GNN: A novel open set recognition based on graph neural network for multi-class classification of psychiatric disorders. Information Fusion, 117, 102887. https://doi.org/https://doi.org/10.1016/j.inffus.2024.102887 

No official code

### DGCN, 2024
> Zhou, N., Yuan, Z., Zhou, H., Lyu, D., Wang, F., Wang, M., Lu, Z., Huang, Q., Chen, Y., Huang, H., Cao, T., Wu, C., Yang, W., & Hong, W. (2025). Using dynamic graph convolutional network to identify individuals with major depression disorder. Journal of Affective Disorders, 371, 188–195. https://doi.org/https://doi.org/10.1016/j.jad.2024.11.035

No official code

### GNNMA, 2025
> Si, W., Wang, G., Liu, L., Zhang, L., & Qiao, L. (2025). Graph neural network with modular attention for identifying brain disorders. Biomedical Signal Processing and Control, 102, 107252. https://doi.org/https://doi.org/10.1016/j.bspc.2024.107252

[Official Code](https://github.com/siwei9898/GNNMA)

### AFMDD, 2025
> Zhang, Y., Liu, X., Tang, P., & Zhang, Z. (2025). AFMDD: Analyzing Functional Connectivity Feature of Major Depressive Disorder by Graph Neural Network-Based Model. Journal of Computational Biology, 32(2), 156–163. https://doi.org/10.1089/cmb.2024.0505

No official code


### Chen, 2022
> F. Chen, L. Wang, Z. Ding, Alteration of whole-brain amplitude of low-frequency fluctuation and degree centrality in patients with mild to moderate depression: A resting-state functional magnetic resonance imaging study, Frontiers in Psychiatry. 13 (2022).

[ROCA] https://github.com/Luoyu-Wang/ROCA

### Wavelet coherence blob analysis + Tree, 2023
> Cîrstian R, Pilmeyer J, Bernas A, et al. Objective biomarkers of depression: A study of Granger causality and wavelet coherence in resting-state fMRI. J Neuroimaging. 2023; 33: 404–414. https://doi.org/10.1111/jon.13085

No official code

### DCM, 2024
> S. J.C. Schielen, D. Stepanov, R. Cîrstian, D. Ruijters, A. P. Aldenkamp and S. Zinger, "Dynamic Causal Modelling Applied to Functional MRI of Depression: An Objective Diagnosis," 2024 IEEE International Symposium on Medical Measurements and Applications (MeMeA), Eindhoven, Netherlands, 2024, pp. 1-6, doi: 10.1109/MeMeA60663.2024.10596739. keywords: {Accuracy;Current measurement;Biological system modeling;Stochastic processes;Independent component analysis;Lead;Depression;neurodynamics;fMRI;causality;depression;diagnosis},

No official code

### STANet, 2024
> Zhang, W.; Zeng, W.; Chen, H.; Liu, J.; Yan, H.; Zhang, K.; Tao, R.; Siok, W.T.; Wang, N. STANet: A Novel Spatio-Temporal Aggregation Network for Depression Classification with Small and Unbalanced FMRI Data. Tomography 2024, 10, 1895-1914. https://doi.org/10.3390/tomography10120138

No official code

## MGD4MD
### Results of different hyperparameter combinations
Epoch, batch size, learning rate, optimizer, weight decay, random seed 

### Ablation Study
auxiliary information, diffusion model