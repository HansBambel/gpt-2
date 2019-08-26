## Using Tensorflow
### Set-up ###
##### A) Without Docker
1. Clone this repository
2. Have python installed
3. Install the requirements of the requirements.txt
	a. This can be done with anaconda or pip (e.g.:pip install tqdm) (I used a conda environment `gpt-2` that was a clone of the basic python env) `conda create --name gpt-2 --clone base`
	b. Install tensorflow (for CPU `pip install tensorflow`, for GPU `pip install tensorflow-gpu`)
	c. If you want to use the GPU you also need to install CUDA 10.0 (Tensorflow 1.14 did not find files from CUDA 10.1) and cuDNN 7.6.1 (https://developer.nvidia.com/cuda-downloads)
		c1. On windows you also need VisualStudio 2017 (I installed 2019 first, may also work), but the CUDA installation will tell you that
		c2. I had the issue that when running 
		c3. add them to your PATH variable:
			C:\tools\cuda\bin
			C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
##### B) With Docker (more detail in [Docker Chapter](#using-docker))
1. Install Docker
2. Clone this repository
3. Build the docker image: `docker build --tag=transformers .`
4. Run an interactive and detached image: `docker run -it -d transformers`
	a: To get the running containers: `docker ps` -a shows all (also stopped containers)
	b. To copy files to the running docker image: `docker cp wordvectors/ <container-name>:/gpt-2`
	c. To copy files from the running docker image to the host: `docker cp <container-name>:/gpt-2 .`
5. To enter the running docker image: `docker exec -it <container-name>`


### Training ###
1. Go into gpt-2 directory
2. For Windows: `set PYTHONPATH=src` For Linux `PYTHONPATH=src`
	a. If using an environment (with anaconda): activate the conda environement (e.g.: `activate gpt-2`)
3. (only needs to be done once) Download model `python download_model.py 117M` 	 
4. Encode your data set as Byte-Pair Encoding (only needs to be done once per dataset) `python encode.py --model_name 117M data\<yourData>.txt data\<yourData>.npz`
	- Then only use the npz file for training
	- Note: When using another model this needs to be done again
	- Note: it is possible to give the encoding a file or a whole directory. It will go through every file in the directory then.
(The parameters for training a model are well described in train.py)
5. Train the 117M model: `python train.py --model_name 117M --run_name <yourModelName> --dataset data\<yourData>.npz --batch_size 1 --top_p 0.9 --save_every 2000 --sample_every 1000`
   - Train the 345M model: `python train.py --model_name 345M --run_name <yourModelName> --dataset data\<yourData>.npz --batch_size 1 --top_p 0.9 --save_every 2000 --sample_every 1000`
   - If training the 345M model did not work due to OOM issues it is possible to use SGD instead of ADAM:
	 - TODO try with memory_saving_gradients
   - `python train.py --model_name 345M --run_name <yourModelName> --dataset data\<yourData>.npz --optimizer sgd --learning_rate 0.001 --batch_size 1 --top_p 0.9 --save_every 2000 --sample_every 1000`
	 - To resume from the latest checkpoint (there will be a folder checkpoint) just run the line from 5. again
	 - To resume from a specific checkpoint `python train.py --restore_from path/to/checkpoint --model_name 117M --dataset data\<yourData>.npz --batch_size 1 --top_p 0.9 --save_every 2500 --sample_every 1000`
	 - To start fresh either delete the folder or run `python train.py --restore_from `fresh` --model_name 117M --dataset data\<yourData>.npz --batch_size 1 --top_p 0.9 --save_every 2500 --sample_every 1000`

### Generating Samples ###
1. Create a folder in `models` with your trained model (e.g.: `trained`)
2. Go to your checkpoints of your model and copy `checkpoint`, `model-xxx.data00000-of-00001`, `model-xxx.index` and `model-xxx.meta` into the new `trained` folder
3. Go to models/117M (or 345M if trained with it) and copy `encoder.json`, `hparams.json` and `vocab.bpe` to your `trained` folder
4. Go to gpt-2 again
	1. Generate unconditioned samples: `python src/generate_unconditional_samples.py --top_p 0.9 --model_name <yourModelName> --nsamples 3`
	2. Generate conditioned samples: `python src\interactive_conditional_samples.py --top_p 0.9 --model_name <yourModelName>`
	3. Generate conditioned samples using a text file: `python src\conditional_samples_with_input.py --top_p 0.9 --model_name 117M --nsamples 3 --length 80 < input.txt`
	- NOTE: `--length 100` limits the output of the samples to 100 tokens (not characters or words)
	- NOTE2: In input.txt the text that is used to condition the model is put.
	- NOTE3: The checkpoint of the model specified in the `model_checkpoint_path` in the checkpoint file is the one that is used.
	
---
### Using the cluster (Cluster uses SLURM) ###
1. Get access to the cluster [Link to Sonic](https://www.ucd.ie/itservices/ourservices/researchit/computeclusters/sonicuserguide/)
2. Use Putty or ssh to connect to cluster
	- Check what modules are available: `module avail` (these can be loaded in the script `module load <module-name>`)
3. Create a .sh script to submit a job to the cluster with specifications about the script
	- Submitting a job to the cluster: `sbatch myjob.sh` and gives back a jobid
		- To use GPU: `sbatch --partition=csgpu myjob.sh`
		- Also make sure that you specify `#SBATCH --gres=gpu:1` otherwise your job will end up in the queue but not start
	- Check running jobs: `squeue`
	- Cancel running job: `scancel <jobid>`
	
---
### Using Docker
Creates an image with the specified packages from the Dockerfile (the requirements) --> only needs to be done once
- `docker build --tag=transformers .`

#### Helpful Docker commands:
- `docker image ls` (lists all `installed` images)
- `docker ps -a` (shows all containers)
- `docker rm <container-name>` (container-name is at the end of docker ps command)
- `docker run <image-name>`
   - `-d` =detached/background
   - `-it` =interactive shell, 
   -`-rm` =removes container after exit, 
   - `-m 32g` =allows the container to use 32gb of RAM (Doesn't seem to work with current Docker version under Windows)
   - `-ipc=`host`` (needs this to make multiprocessing possible))
- `docker exec -it <container-name> /bin/bash` (enter running container)
- `docker cp . <container-name>:/gpt2` (copies files from host to container)
- `docker stop <container-name>`	(stops the container)
- `docker container prune` (removes all stopped containers)

##### Usual docker use
1. `docker run -it -d transformers`
2. get container name from `docker ps` command
3. Copy files from host to container: `docker cp . <container-name>:/gpt2`
4. enter running container again: `docker exec -it <container-name> /bin/bash`
5. run your script `python probabilities.py`
6. get the created wordvectors from the container `docker cp <container-name>:gpt2/wordvectors/. wordvectors\`
