# ML-Tech-Cheatsheet 📄
<a href="https://github.com/BrianPulfer/ML-Tech-Cheatsheet/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/BrianPulfer/ML-Tech-Cheatsheet?style=social)</a>

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
- [Colossal-AI](#colossal-ai)
- [spaCy](#spacy)
- [nvidia-ml-py3](#nvidia-ml-py3)
- [albumentations](#albumentations)
- [augly](#augly)
- [einops](#einops)
- [bitsandbytes](#bitsandbytes)

[Scientific Libraries ](#scientific-libraries)
- [Hydra](#hydra)
- [SciencePlots](#scienceplots)
  
[Environments 🌎](#environments)
  - [libraries](#libraries)
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
- [CircleCI](#circleci)
- [GitHub Actions](#github-actions)
- [GitHub Pages](#github-pages)
- [Others](#others)

[Web development 🌐](#web-development)
- [Prototyping](#prototyping)
- [Frontend](#frontend)
- [Backend](#backend)
- [APIs](#apis)
- [Database](#database)
- [Devops](#devops)

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
 - [timm](https://pypi.org/project/timm/) provides a multitude of pre-trained and vanilla image models.
 - [Safetensors](https://huggingface.co/docs/safetensors/index) is HuggingFace🤗 package which allows storing tensors in a safe way (unlike with pickle files).
 - [accelerate](https://pypi.org/project/accelerate/) takes care of automatically finding the best available device for training (PyTorch).
 - [optimum](https://pypi.org/project/optimum/) provides multiple features to accelerate training and inference
 - [tokenizers](https://pypi.org/project/tokenizers/) provides features to simply carry-out popular tokenizations.
 - [evaluate](https://pypi.org/project/evaluate/) allows to evaluate and compare trained models.
 - [peft](https://github.com/huggingface/peft) (Parameter-Efficient Fine-Tuning) provides implementations of algorithms like LORA, which allow to speed up fine-tuning while saving memory consumption.
 - [xformers](https://huggingface.co/docs/diffusers/optimization/xformers) provides optimized implementation of all operations carried-out in transformers (e.g. Memory Efficient Attention).

### DeepSpeed
[DeepSpeed](https://www.deepspeed.ai/) allows for distributed high-performance and efficient training. 
DeepSpeed is [supported in PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/advanced/model_parallel.html#deepspeed).

### Colossal AI
[Colossal-AI](https://colossalai.org/) is a framework that improves the efficiency and speed of large model training, especially for HPC clusters.

### spaCy
[Spacy](https://spacy.io/) offers a multitude of features and pre-trained pipelines for NLP tasks (like huggingface, but just for NLP).

### nvidia-ml-py3
This [library](https://pypi.org/project/nvidia-ml-py3/) allows to access information about NVIDIA GPUs directly in python code.

### albumentations

All sorts of popular [image augmentations](https://github.com/albumentations-team/albumentations#list-of-augmentations),
like ColorJitter, ZoomBlur, Gaussian Noise... are implemented by [albumentations](https://albumentations.ai/).

### augly

[Data augmentation library](https://pypi.org/project/augly/) for text, sound, image and video.

### einops

Manipulation of tensors (reshaping, concatenating, ...) with [einops](https://github.com/arogozhnikov/einops) is
extremely intuitive and time-saving.

### bitsandbytes
[bitsandbytes](https://pypi.org/project/bitsandbytes/) allow to run training using 8-bit precision. Particularly useful to fine-tune very large models.

## Scientific libraries
### Hydra
[Hydra](https://hydra.cc/docs/intro/) allows to set multiple configurations smoothly.

### SciencePlots
[SciencePlots](https://github.com/garrettj403/SciencePlots) allows to plot much nicer plots than classic matplotlib and seaborn.


## Environments

### libraries

[python-dotenv](https://github.com/theskumar/python-dotenv) allows to define and read environmental variables from a file.

[yacs](https://github.com/rbgirshick/yacs) allows to manage configurations such as hyperparameters for experiments.

### conda

[Conda](https://conda.io/) allows to easily create and share virtual environments. The
command `conda env export > environment.yml` creates a .yml file that can be used to create an identical virtual
environment.

### Docker

[Docker](https://docker.com) allows to emulate a whole operating system.

## CLI Utilities

### Terminals
[Hyper.js](https://hyper.is/) allows for high customization and is available on MacOS, Windows and Linux. Interesting plugins are documented in this nice [repo](https://github.com/bnb/awesome-hyper).

[iTerm2](https://iterm2.com/) is a MacOS only terminal emulator with lots of functionalities.

[Oh My Zsh](https://www.github.com/ohmyzsh/ohmyzsh) is available on Unix-like machines. It provides terminal plug-ins 
and themes.

### Commands and utils
* `~/.ssh/config` and `~/.ssh/authorized_keys` files to define known host names and authorized ssh keys.
* `nvidia-smi` ➡️ Check NVIDIA Cards current status
* `ps`, `top`, `htop` ➡️ Check currently running processes
* [`bpytop`](https://github.com/aristocratos/bpytop) - Like `htop`, but better.
* [`nvitop`](https://github.com/XuehaiPan/nvitop) ➡️ Like `nvidia-smi`, but better.
* [`tmux`](https://github.com/tmux/tmux/wiki) ➡️ Terminal multiplexer, allows to easily detach jobs.
* [Fig](https://fig.io/) ➡️ Intellisense (and much more) for command line commands.

## High Performance Computing 🦾

### slurm

HPC clusters typically use a cluster management and job scheduling tool. [Slurm](https://slurm.schedmd.com/) allows to
schedule jobs, handle priorities, design partitions and much more. Cheatsheet files for slurm are under
the [/slurm ](/slurm) folder. The library [submitit](https://pypi.org/project/submitit/) allows to switch seamlessly between executing on Slurm or locally.

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

### CircleCI

[CircleCI](https://circleci.com/) hosts CI/CD pipelines and workflows, similarly to [GitHub Actions](https://docs.github.com/en/actions).

### GitHub Actions

[GitHub Actions](https://docs.github.com/en/actions) allows to execute custom actions automatically upon some triggers
by some events (pull requests, pushes, issues opened, ...).

### GitHub Pages

[GitHub Pages](https://docs.github.com/en/pages) allows to host a webpage for each GitHub repository.

### Others
[GitBook](https://www.gitbook.com/) allows to simply create a documentation starting from a GitHub repository.

[Pre-commit](https://pypi.org/project/pre-commit/) allows to create customized pre-commit hooks to, e.g., run formatting or testing before committing. Some nice things to include there:
 - [Black](https://pypi.org/project/black/) formats Python files compliantly to PEP 8.
 - [autopep8](https://pypi.org/project/autopep8/) allows to automatically format files to be compliant with PEP 8.
 - [yapf](https://github.com/google/yapf) is like autopep8, but with a search algorithm for the best possible formatting.
 - [isort](https://pycqa.github.io/isort/) automatically sorts order of import instructions in python files.
 - [flake8](https://flake8.pycqa.org/en/latest/) uses other tools to check for python errors (pyflakes), correct use of PEP conventions and others.
 - [pylint](https://pypi.org/project/pylint/), similarly to flake, analyzes the code and checks for errors without actually running it.

[Shields.io](https://shields.io/) allows to put neat banner in README files, such as the number of <a href="https://github.com/BrianPulfer/ML-Tech-Cheatsheet" rel="nofollow"><img src="https://img.shields.io/github/stars/BrianPulfer/ML-Tech-Cheatsheet" alt="Downloads" style="max-width: 100%;"></a> of the repository.


## Web development
I find it extremelly satisfying to build an actual prototype or product out of a Machine Learning project. Here's my favourite options:

### Prototyping
To quickly create interactive apps based on trained machine learning models, [gradio](https://gradio.app/)
and [streamlit](https://streamlit.io/) are among the most popular frameworks. While it is easy to prototype using these frameworks, more complex applications are better built with a more complete stack. [Figma](https://www.figma.com/) is currently the best tool I could find to design an app / website.

### Frontend
On the frontend, [NextJS](https://nextjs.org/) is one of the most popular frameworks. It builds on top of the [React](https://react.dev/) framework and provides additional functionalities and optimizations.
[Tailwindcss](https://tailwindcss.com/) allows for easy styling without the need for css style sheets.
[Chakra-UI](https://chakra-ui.com/) comes with pre-built and nice looking components. It also offers support for dark mode.

### Backend
Since we are interested in Machine Learning applications, it makes sense to pick a python backend.

[FastAPI](https://fastapi.tiangolo.com/) is a python backend extremelly simple to set-up and highly optimized for speed. [Django](https://www.djangoproject.com/) and [Flask](https://flask.palletsprojects.com/en/2.3.x/) are more popular frameworks. Django is a full-stack meant for big projects with a clearly defined structure, whereas flask is lightweight and meant for smaller projects.

### APIs
[Auth0](https://auth0.com/) allows for authentication and authorization. [Stripe](https://stripe.com/) is a popular tool to deal with payments.


### Database
[MySQL](https://www.mysql.com/), [PostgreSQL](https://www.postgresql.org/about/), [Redis](https://redis.io/) and [MongoDB](https://www.mongodb.com/it-it) and are all very valid and popular databases.

[PostgreSQL](https://www.postgresql.org/about/) is preferable over [MySQL](https://www.mysql.com/) for its better support for JSON data. [Redis](https://redis.io/) is a key-value database, which is very fast and useful for caching. [MongoDB](https://www.mongodb.com/it-it) is a document-oriented database, which is very flexible and easy to use.

[Prisma](https://www.prisma.io/) is a nodejs database toolkit compatible with MySQL, PostgreSQL, SQLite and SQL server. It allows to easily create and manage databases.

### Devops
Applications can be hosted on a number of services. [Heroku](https://www.heroku.com/), [DigitalOcean](https://www.digitalocean.com/), [AWS](https://aws.amazon.com/), [Google Cloud](https://cloud.google.com/) and [Microsoft Azure](https://azure.microsoft.com/en-us) are among the most popular solutions.
