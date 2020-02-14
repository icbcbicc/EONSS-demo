The procedure to install and run EONSS is provided below.

**********************************
INSTALLATION
**********************************
EONSS is based on PyTorch, please follow the installation guide on their website (https://pytorch.org/get-started/locally/)
Alternatively, you may follow the installation guide (Mar. 2019) that we provide below for installing the dependencies:

1. Install Anaconda3 (A distribution of Python3, please see https://www.anaconda.com/distribution/)
2. Create a new virtual environment using the envirionment setup file we provided:
	2.1. Open Anaconda3 Prompt (Windows) as administrator or terminal (Ubuntu)
    2.2. Run `conda env create -f chooseyourenv.yml`
    2.3. chooseyourenv.yml should be one of the following depending on your system setup:
    	a) env_CPU_Ubuntu.yml (CPU only version for Ubuntu, envname=EONSS_CPU)
    	b) env_GPU_Ubuntu.yml (GPU and CPU version for Ubuntu, envname=EONSS_GPU)
    	c) env_CPU_Windows.yml (CPU only version for Windows, envname=EONSS_CPU)
    	d) env_GPU_Windows.yml (GPU and CPU version for Windows, envname=EONSS_GPU)


We have tested the code on Ubuntu 18.04 and Windows 10 (1809) with both CPU and GPU mode.

**********************************
EONSS USAGE
**********************************
1. Open Anaconda3 Prompt (Windows) or terminal (Ubuntu)
2. Activate the virtual environment: `conda activate envname` (envname depends on the chooseyourenv.yml you use)
3. Run EONSS, run `python demo.py --img yourimgname --use_cuda`
    3.1. If you are using GPU, add `--use_cuda` to your command above.
	3.2. If you want to save the EONSS score to the disk, add `--save_result` to your command above.
4. Example using the test image provided with release: `python demo.py --img Dist.jpg --save_result`
Note: EONSS requires RGB Color images as input.

**********************************
Citation
**********************************
We are making the EONSS model available to the research community free of charge. 
If you use this database in your research, we kindly ask that you reference our papers listed below:

Z. Wang, S. Athar, Z. Wang, “Blind Quality Assessment of Multiply Distorted Images Using Deep Neural Networks”, 
16th International Conference on Image Analysis and Recognition, Waterloo, Ontario, Canada, August 27-29, 2019.

@InProceedings{wang2019blind,
author="Wang, Zhongling and Athar, Shahrukh and Wang, Zhou",
title="Blind Quality Assessment of Multiply Distorted Images Using Deep Neural Networks",
booktitle="International Conference on Image Analysis and Recognition",
year="2019",
pages="89--101"
}