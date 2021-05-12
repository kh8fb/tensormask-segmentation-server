# tensormask-segmentation-server
A cli-based server for obtaining [integrated directional gradients](https://github.com/kh8fb/intermediate-gradients) and [integrated gradients](https://arxiv.org/abs/1703.01365) from images using `curl` requests.

### Installation

#### Initial Setup

This package requires the installation of both this repository as well as [Facebook's detectron](https://github.com/facebookresearch/detectron2) in an Anaconda environment.

First, create an Anaconda environment:

       conda create -n tensormask-segmentation-server python=3.8

Next, activate the environment, and `conda install` torch, torchvision, and cudatoolkit,

      conda activate tensormask-segmentation-server
      conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c nvidia

#### Installing Detectron and TensorMask

Make sure that the path to cuda is available and set the `$CUDA_HOME` environmental variable. On some HPC servers this involves `module load gcccuda` and running `which nvcc` to obtain and set the `$CUDA_HOME` variable.;

      export CUDA_HOME=/path/to/cuda-11.x.x

Then, clone the detectron2.git and `pip install` it.  Also `pip install` the Detectron's TensorMask module located at `detectron2/projects/TensorMask/`

      git clone https://github.com/facebookresearch/detectron2.git
      python -m pip install -e /path/to/detectron2
      pip install -e /path/to/detectron2/projects/TensorMask

Alternatively, follow their instruction steps [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) and [here](https://github.com/facebookresearch/detectron2/tree/master/projects/TensorMask)

#### Installing the tensormask-segmentation-server

Finally, `cd` into this project's directory and install the requirements with

      pip install -e .
 
Now your environment is set up and you're ready to go.

### Usage
Activate the server directly from the command line with

	 intgrads-images -bp /path/to/bit_model.pth --cpu

OR

	intgrads-images -lb /path/to/lanet_model.pth --cuda

This command starts the server and load the model so that it's ready to go when called upon.
The pretrained and finetuned BiT and LaNet models can be downloaded from this [Google drive folder](https://drive.google.com/drive/u/0/folders/1KtuVv2GPtbcuy9fifuCXySuqQhcPc-nO)

You can provide additional arguments such as the hostname, port, and a cuda flag.

After the software has been started, run `curl` with the "model" filepath to get and download the attributions.

      curl http://localhost:8888/model/ --data @input_json_file.json --output saved_file.gzip -H "Content-Type:application/json; chartset=utf-8"

#### Preparing Inputs for the Server

The input_json_file.json can be produced from an image with the script `prepare_input.py`. This will store the image as a JSON file of RGB values and the image can thus be passed to the server. `prepare_input.py` also takes in the classification of the image which gradients are taken in respect to.  This classification is one of ‘airplane’, ‘automobile’, ‘bird’, ‘cat’, ‘deer’, ‘dog’, ‘frog’, ‘horse’, ‘ship’, or ‘truck'.

    python prepare_input.py /path/to/image.jpeg airplane input_json_file.json

### Interpreting Server Outputs

The gradients are stored in a dictionary with the keys "integrated_grads", "integrated_directional_grads", "step_sizes", and "intermediates".  They are then compressed and able to be retrieved from the saved gzip file with:

      >>> import gzip
      >>> import torch
      >>> from io import BytesIO
      >>> with gzip.open("saved_file.gzip", 'rb') as fobj:
      >>>      x = BytesIO(fobj.read())
      >>>      grad_dict = torch.load(x)


### Running on a remote server
If you want to run int-grads-server on a remote server, you can specify the hostname to be 0.0.0.0 from the command line.  Then use the `hostname` command to find out which IP address the server is running on.

       intgrads -lb /path/to/lanet.pth -h 0.0.0.0 -p 8008 --cuda
       hostname -I
       10.123.45.110 10.222.222.345 10.333.345.678

The first hostname result tells you which address to use in your `curl` request.

      curl http://10.123.45.110:8008/model/ --data @input_json_file.json --output saved_file.gzip -H "Content-Type:application/json; chartset=utf-8"