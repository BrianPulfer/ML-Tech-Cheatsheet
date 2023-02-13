# ML-Tech-Cheatsheet 📄

Personal "cheatsheet" repository for my ideal machine learning tech-stack. I use this repository to play around and
familiarize
with ML libraries, advanced git and GitHub features, virtualization and so on 🤓.

## Table of Contents 📜
[IDEs plugins 🧰](#ides-plugins)
- [VSCode](#vscode)
- [PyCharm](#pycharm)

[Machine Learning Libraries 🤖](#machine-learning-libraries)
- [The classics](#the-classics)
- [Pytorch, Lightning and W\&Bs](#pytorch-lightning-and-wbs)
- [transformers](#transformers)
- [DeepSpeed](#deepspeed)
- [spaCy](#spacy)
- [nvidia-ml-py3](#nvidia-ml-py3)
- [albumentations](#albumentations)
- [einops](#einops)
- [gradio and streamlit](#gradio-and-streamlit)
  
[Environments 🌎](#environments)
  - [conda](#conda)
  - [Docker](#docker)

[CLI Utilities 👨‍💻](#cli-utilities)

[High Performance Computing 🦾](#high-performance-computing-)
- [slurm](#slurm)

[Git 🐱](#git)
- [Protected Branches](#protected-branches)
- [Tags and Releases](#tags-and-releases)
- [LFS](#lfs)
- [Hidden Directory](#hidden-directory)
- [GitHub Actions](#github-actions)
- [GitHub Pages](#github-pages)
- [Others](#others)

## IDEs plugins

### VSCode

* Python
* RainbowCSV
* Remote
* CoPilot
* GitLens
* Docker
* Jupiter
* Gitignore
* vscode-pdf

### PyCharm

* GitToolBox
* CoPilot
* Docker

## Machine Learning Libraries

### The classics

* **NumPy** - Math operations, manipulations, linear algebra and more.
* **Pandas** - Tabular data management.
* **MatplotLib** and **Seaborn** - All sorts of plots.
* **OpenCV2**, **Pillow**, and **Sci-Kit Image** - Image manipulation

### Pytorch, Lightning and W&Bs

[PyTorch](https://pytorch.org/) is currently the reference ML framework for Python.

[Weights and Biases](https://wandb.ai/) (W&B) allows to easily track experiments, performances, parameters and so on in
a single place.

[PyTorch Lightning](https://www.pytorchlightning.ai/) gets rid of most of the usual PyTorch boilerplate code, like
train/val/test loops, backward and optim steps and so on. It also allows to easily use powerful pytorch features and
other libraries (like W&B) by inserting just few optional parameters here and there.

### transformers
[HuggingFace🤗](https://huggingface.co/) allows to easily download, fine-tune and deploy pre-trained transformer models across a multitude of applications.
It is also possible to share models and datasets on the platform, as well as "spaces" which are interactive live demos of the capabilities of the created models.

Related libraries:
 - [Datasets](https://pypi.org/project/datasets/) provides efficient loading of custom or common dataset samples (even online).
 - [Diffusers](https://pypi.org/project/diffusers/) is HuggingFace🤗 package for diffusion models specifically. It comes with pre-trained SOTA model for vision and audio generation.
 - [Safetensors](https://huggingface.co/docs/safetensors/index) is HuggingFace🤗 package which allows storing tensors in a safe way (unlike with pickle files).
 - [accelerate](https://pypi.org/project/accelerate/) takes care of automatically finding the best available device for training (PyTorch).
 - [optimum](https://pypi.org/project/optimum/) provides multiple features to accelerate training and inference
 - [tokenizers](https://pypi.org/project/tokenizers/) provides features to simply carry-out popular tokenizations.
 - [evaluate](https://pypi.org/project/evaluate/) allows to evaluate and compare trained models.

### DeepSpeed
[DeepSpeed](https://www.deepspeed.ai/) allows for distributed high-performance and efficient training. 
DeepSpeed is [supported in PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed).

### spaCy
[Spacy](https://spacy.io/) offers a multitude of features and pre-trained pipelines for NLP tasks (like huggingface, but just for NLP).

### nvidia-ml-py3
This [library](https://pypi.org/project/nvidia-ml-py3/) allows to access information about NVIDIA GPUs directly in python code.

### albumentations

All sorts of popular [image augmentations](https://github.com/albumentations-team/albumentations#list-of-augmentations),
like ColorJitter, ZoomBlur, Gaussian Noise... are implemented by [albumentations](https://albumentations.ai/).

### einops

Manipulation of tensors (reshaping, concatenating, ...) with [einops](https://github.com/arogozhnikov/einops) is
extremely intuitive and time-saving.

### gradio and streamlit

To quickly create interactive apps based on trained machine learning models, [gradio](https://gradio.app/)
and [streamlit](https://streamlit.io/) are among the most popular frameworks.

## Environments

### conda

[Conda](https://conda.io/) allows to easily create and share virtual environments. The
command `conda env export > environment.yml` creates a .yml file that can be used to create an identical virtual
environment.

### Docker

[Docker](https://docker.com) allows to emulate a whole operating system.

## CLI Utilities

* `~/.ssh/config` and `~/.ssh/authorized_keys` files to define known host names and authorized ssh keys.
* `nvidia-smi` ➡️ Check NVIDIA Cards current status
* `ps`, `top`, `htop` ➡️ Check currently running processes
* [`nvitop`](https://github.com/XuehaiPan/nvitop) ➡️ Like `nvidia-smi`, but better.
* [`tmux`](https://github.com/tmux/tmux/wiki) ➡️ Terminal multiplexer, allows to easily detach jobs.
* [Fig](https://fig.io/) ➡️ Intellisense (and much more) for command line commands.

## High Performance Computing 🦾

### slurm

HPC clusters typically use a cluster management and job scheduling tool. [Slurm](https://slurm.schedmd.com/) allows to
schedule jobs, handle priorities, design partitions and much more. Cheatsheet files for slurm are under
the [/slurm ](/slurm) folder.

## Git

Taking the time to go through most of [GitHub's Documentation](https://docs.github.com/) at least once is very
important. Here's a few features to keep in mind.

### Protected Branches

[Protected branches](https://docs.github.com/en/rest/branches/branch-protection) prevent code to be pushed onto custom branches.

### Tags and Releases

Important commits can be [tagged](https://git-scm.com/book/en/v2/Git-Basics-Tagging). Then, jumping to a tagged commit is easy as:

```git checkout $tag-name```

### LFS

[Git Large File System](https://git-lfs.github.com/) allows to push bigger files to the GitHub repository. **Careful**:
There is a global usage quota per GitHub account that goes across repositories.

### Hidden Directory

The `.github` directory allows to keep the landing page of the GitHub repository "clean" and includes:

* **[CONTRIBUTING.md](CONTRIBUTING.md)** ➡️ Guidelines to contribute to the repository.
* **[ISSUE_TEMPLATE.md](ISSUE_TEMPLATE.md)** ➡️ Template for issues.
* **[PULL_REQUEST_TEMPLATE.md](PULL_REQUEST_TEMPLATE.md)** ➡️Template for pull requests.
* **[README.md](README.md)** ➡️Repository's README (i.e. this) file.
* **[workflows](workflows)** ➡️Directory which contains .yaml files for GitHub actions.

### GitHub Actions

[GitHub Actions](https://docs.github.com/en/actions) allows to execute custom actions automatically upon some triggers
by some events (pull requests, pushes, issues opened, ...).

### GitHub Pages

[GitHub Pages](https://docs.github.com/en/pages) allows to host a webpage for each GitHub repository.

### Others
[GitBook](https://www.gitbook.com/) allows to simply create a documentation starting from a GitHub repository.

[Pre-commit](https://pypi.org/project/pre-commit/) allows to create customized pre-commit hooks to, e.g., run formatting or testing before committing.

[Black](https://pypi.org/project/black/) formats Python files compliantly to PEP 8.