# MGD4MD
Multi-modal Graph Diffusion for identifying neuroimaging biomarkers of mental disorder

## Windows11上配准
pip install antspyx

通用预处理步骤
1. 去除头骨
2. 时间层校正(Slice-Timing Correction):旨在解决MRI扫描中由于切片采集顺序不同而导致的时间差问题
3. 配准与归一化(Registration and Normalization):将每个被试的大脑图像都映射到同一个标准空间
4. 对齐与运动校正(Alignment and Motion Correction):纠正被试在扫描过程中可能产生的头部运动
5. 平滑(Smoothing):对图像进行平均处理来减少噪音，提升信号的质量
6. 掩膜(or 掩码)与缩放(Masking and Scaling):用于生成头部掩膜，并对数据进行缩放处理，以确保数据的一致性
7. 检查预处理结果:


## Prepare the Enviorment:
``` shell
module load anaconda/2021.11 cuda/12.1 # N32EA14P
module load anaconda/2022.10 cuda/12.1 # N40R4
conda create --name BraVO python=3.11
source activate BraVO

``` shell
chmod 777 *.sh
sbatch --gpus=num_gpus -p gpu run.sh # submit the job
parajobs # check id of the job
scancel job_id # cancel the job via its id
```

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 
pip install nibabel -i https://pypi.tuna.tsinghua.edu.cn/simple/ 
pip install SimpleITK -i https://pypi.tuna.tsinghua.edu.cn/simple/ 

