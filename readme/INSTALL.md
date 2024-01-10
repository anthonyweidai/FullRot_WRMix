# Installation

The code was tested with [Anaconda](https://www.anaconda.com/download) Python 3.9.12, CUDA 10.1, and [PyTorch]((http://pytorch.org/)) 1.10.1.
After installing Anaconda:

0. [Optional but highly recommended] create a new conda environment. 

    ~~~
    conda create -n fullrot python=3.9.12
    ~~~
    And activate the environment.
    
    ~~~
    conda activate fullrot
    ~~~

1. Install PyTorch:

    ~~~
    conda install pytorch=1.11.0 torchvision=0.12.0 cudatoolkit=11.3.1 -c pytorch
    ~~~

3. Clone this repo:

    ~~~
    git clone https://github.com/anthonyweidai/FullRot_WRMix.git
    ~~~

4. Install the requirements

    ~~~
    cd $FullRot_WRMix_ROOT
    pip install -r requirements.txt
    ~~~
    
