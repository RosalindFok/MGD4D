# MGD4MD
Multi-modal Graph Diffusion for identifying neuroimaging biomarkers of mental disorder

## 1. Preprocess
Platform: Windows11, Python 3.11.4

``` shell
# dependencies
pip3 install numpy
pip install antspyx
pip install nibabel
pip install pandas
pip install tqdm
pip install matplotlib
pip install nilearn

# script
python preprocess.py
```


## 2. Train
Platform: Beijing Super Cloud Computing Center - N32EA14P: `NVIDIA A100-PCIE-40GB * 8`

``` shell
# conda enviorment
module load cuda/12.1 # N32EA14P
conda create --name MGD4MD python=3.11
conda activate MGD4MD

# dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Successfully installed MarkupSafe-2.1.5 filelock-3.13.1 fsspec-2024.6.1 jinja2-3.1.4 mpmath-1.3.0 networkx-3.3 numpy-2.1.2 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.1.105 nvidia-nvtx-cu12-12.1.105 pillow-11.0.0 sympy-1.13.1 torch-2.5.1+cu121 torchaudio-2.5.1+cu121 torchvision-0.20.1+cu121 triton-3.1.0 typing-extensions-4.12.2
pip install nilearn -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed certifi-2025.1.31 charset-normalizer-3.4.1 idna-3.10 importlib-resources-6.5.2 joblib-1.4.2 lxml-5.3.0 nibabel-5.3.2 nilearn-0.11.1 packaging-24.2 pandas-2.2.3 python-dateutil-2.9.0.post0 pytz-2025.1 requests-2.32.3 scikit-learn-1.6.1 scipy-1.15.1 six-1.17.0 threadpoolctl-3.5.0 tzdata-2025.1 urllib3-2.3.0
pip install tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed tqdm-4.67.1
pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple/ # Successfully installed contourpy-1.3.1 cycler-0.12.1 fonttools-4.55.8 kiwisolver-1.4.8 matplotlib-3.10.0 pyparsing-3.2.1

# script
chmod 777 *.sh
sbatch --gpus=1 -p gpu run.sh # submit the job
parajobs # check id of the job
scancel job_id # cancel the job via its id
sh clear.sh # clear the log file

conda deactivate
conda env remove -n MGD4MD
```
