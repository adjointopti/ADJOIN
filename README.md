
# [Texture Optimization](https://adjointopti.github.io/adjoin.github.io/)

## Installation
To implement our code, a pytorch3d environment should be installed:

```
conda env create -f environment.yml
conda activate pytorch3d
conda install pytorch3d=0.2.5 -c pytorch3d -y
```

## File Introduction

'Main.py': optimiation code


## Running
To optimize noised data:
```
CUDA_VISIBLE_DEVICES=0 sh run.sh
```




