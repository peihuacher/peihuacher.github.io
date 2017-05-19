## Running treelstm in DGX1

Recent works by [Tai et. al. 2015](https://arxiv.org/abs/1503.00075) has shown that treelstm performs better than lstm. They applied the algorithm on Sentence Relation and Sentiment Analysis. (Detailed information on treelstm can be found in https://github.com/stanfordnlp/treelstm)

In my project, I would like to train a classifier using treelstm and compare its performance with traditional classifiers (Naive Bayes etc). The main purpose of this write up is to share and document the setting up of the environment to run treelstm on the CPU or GPU in NVIDIA-DGX1. 

### On the NVIDIA-DGX1 platform using [stanfordnlp/treelstm](https://github.com/stanfordnlp/treelstm)
In NVIDIA-DGX1, we are working in a Docker with NVIDIA torch environment installed and configured by NVIDIA. 

```
nvidia-docker run -it —name torch-cph-treelstm -v ~/data:/opt/data -w /opt/data compute.nvidia.com/nvidia/torch bash

## git clone treelstm
git clone https://github.com/stanfordnlp/treelstm
```

You do not need to execute the command `luarocks install nngraph` as this is pre-installed in the docker. You can check the installed packages in luarocks using `luarocks list`.

After executing `./fetch_and_preprocess.sg`, we invoke `th relatedness/main.lua`. This command works but it consumed the CPU to over 3000%. 

According to the website http://kbullaughey.github.io/lstm-play/2015/09/21/torch-and-gpu.html, to invoke use of GPU, you will need to invoke the command :cuda() in torch.  

After entering these in the codes, the error occured
```
/usr/bin/luajit: /usr/share/lua/5.1/nn/Linear.lua:57: invalid arguments: DoubleTensor number DoubleTensor CudaTensor 
expected arguments: *DoubleTensor~1D* [DoubleTensor~1D] [double] DoubleTensor~2D DoubleTensor~1D | *DoubleTensor~1D* double [DoubleTensor~1D] double DoubleTensor~2D DoubleTensor~1D
stack traceback:
        [C]: in function 'addmv'
```

While seeking for a solution to fix the above error, I try to use treelstm.pytorch instead.


### On the NVIDIA-DGX1 platform using [dasguptar/treelstm.pytorch](https://github.com/dasguptar/treelstm.pytorch)
The first thing is still to start up you torch nvidia-docker. This time clone the treelstm.pytorch
```
nvidia-docker run -it —name torch-cph-treelstm -v ~/data:/opt/data -w /opt/data compute.nvidia.com/nvidia/torch bash

## git clone treelstm
git clone https://github.com/dasguptar/treelstm.pytorch
```

Install the required softwares
```
## pytorch http://pytorch.org 
pip2 install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
pip2 install torchvision

## tqdm
pip2 install tqdm

## java 
export JAVA_HOME=/opt/data/jdk1.8.0_131
echo $JAVA_HOME
export PATH=$JAVA_HOME:$PATH
echo $PATH

## python
python -V
```

Thereafter, execute `./fetch_and_preprocess.sh` or `sh fetch_and_preprocess.sh`.

Finally run `python main.py --lr 0.01 --wd 0.0001 --optim adagrad --batchsize 25`.

### Implementing event classification using treelstm.pytorch

To apply the treelstm in my event classification task, I will need to change the some source files. They are:
- main.py
- dataset.py
- trainer.py
- model.py
- metrics.py


