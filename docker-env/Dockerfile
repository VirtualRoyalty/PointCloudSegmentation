FROM vadikalp/jupyter-pcl

USER root

ENV CONDA_DIR=/opt/conda

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

RUN pip install --no-cache-dir torch

RUN pip install --no-cache-dir torchvision

USER jovyan
