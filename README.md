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
python fMRI2Graph.py
```


## Prepare the Enviorment:
``` shell
module load anaconda/2021.11 cuda/12.1 # N32EA14P
module load anaconda/2022.10 cuda/12.1 # N40R4
conda create --name BraVO python=3.11
source activate BraVO
```

``` shell
chmod 777 *.sh
sbatch --gpus=num_gpus -p gpu run.sh # submit the job
parajobs # check id of the job
scancel job_id # cancel the job via its id

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
pip install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple/ 

```
