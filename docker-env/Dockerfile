FROM vadikalp/jupyter-pcl

USER root

ENV CONDA_DIR=/opt/conda

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install torch

RUN pip install torchvision

USER jovyan
