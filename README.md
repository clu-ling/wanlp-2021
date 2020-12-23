# arabic-nlp

Models and utilities for Arabic NLP.


# Development

The code is organized as a Python module.  We recommend using [Docker](https://docs.docker.com/get-docker/) for development.

Please note if you need to install additional dependencies, you'll need to alter the `requirements.txt` and rebuild the docker image:

```bash
docker build -f Dockerfile -t "parsertongue/arabic-nlp:latest" .
```

If you're only using the existing dependencies, you can simply download the published Docker image:

```bash
docker pull "parsertongue/arabic-nlp:latest"
```

To run code interactively in the iPython interpreter:

```bash
docker run -it -v $PWD:/app parsertongue/arabic-nlp:latest ipython
```

To run code interactively in a Jupyter notebook, run the following command and open your browser to [localhost:8889](http://localhost:8889):

```bash
docker run -it -v $PWD:/app -p 8889:9999 parsertongue/arabic-nlp:latest launch-notebook
```

## Test

Tests are written by [extending the `TestCase` class](https://docs.python.org/3.8/library/unittest.html#unittest.TestCase) from the `unittest` module in the Python standard library.  All tests can be found in the [`tests`](./tests) directory.

All tests can be run using the following command:

```bash
docker run -it -v $PWD:/app "parsertongue/arabic-nlp:latest" test-all
```

## Test

Tests are written by [extending the `TestCase` class](https://docs.python.org/3.8/library/unittest.html#unittest.TestCase) from the `unittest` module in the Python standard library.  All tests can be foun
d in the [`tests`](./tests) directory.

All tests can be run using the following command:

```bash
docker run -it -v $PWD:/app "parsertongue/arabic-nlp:latest" test-all
```

### Unit tests

To run just the unit tests, run the following command:

```bash
docker run -it -v $PWD:/app "parsertongue/arabic-nlp:latest" green -vvv
```

### Type hints

The code makes use of Python type hints.  To perform type checking, run the following command:

```bash
docker run -it -v $PWD:/app "parsertongue/arabic-nlp:latest" mypy --ignore-missing-imports --follow-imports=skip --strict-optional /app
```
### Running an experiment

The snippet below will run an experiment using [toy data](./tests/toy-data):

```bash
docker run -it -v $PWD:/app parsertongue/arabic-nlp:latest cdd-base-model --config /app/tests/toy-data/test-experiment-config.yml
```

# Singularity

To build a Singularity image from a published Docker image, run the following command:

```bash
singularity pull cdd.sif docker://parsertongue/arabic-nlp:latest


## HPC

To test the image on the UA HPC with GPU acceleration, first request an interactive session:

```bash
qsub -I \
-N interactive-gpu \
-W group_list=mygroupnamehere \
-q standard \
-l select=1:ncpus=2:mem=16gb:ngpus=1 \
-l cput=3:0:0 \
-l walltime=1:0:0
```

The following modules are necessary:

```bash
module load singularity
module load cuda11
```

NOTE: not all clusters have the `cuda11` module installed.

```bash
singularity shell --nv --no-home /path/to/your.sif
```

To check that the GPU is found, run the following command:

```bash
nvidia-smi
```

Finally, confirm that PyTorch can locate the GPU:

```python
python -c "import torch;print(f'CUDA?:\t{torch.cuda.is_available()}');print(f'GPU:\t{torch.cuda.get_device_name(0)}')"
```

# Local Singularity build

NOTE: These instructions assume you've already installed Singularity (>= v3.7.0) locally on a Linux system.

First, we'll build a docker image:

```bash
docker build -f Dockerfile -t "parsertongue/arabic-nlp:latest" .
```

Next, we'll use the [`docker-daemon` bootstrap agent](https://sylabs.io/guides/3.7/user-guide/appendix.html#docker-daemon-archive) to build a Singularity image from this docker image:

```bash
sudo singularity build arabic-nlp.sif docker-daemon://parsertongue/arabic-nlp:latest
```

Finally, we'll test that image works:

```bash
singularity exec --nv arabic-nlp.sif train-arabert --help
