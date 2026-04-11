FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Dépendances système pour GDAL/rasterio
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python — EcoHealth-ShrubMap pipeline
RUN pip install --no-cache-dir \
    rasterio \
    geopandas \
    shapely \
    pyproj \
    fiona \
    pystac-client \
    planetary-computer \
    laspy \
    torch \
    torchvision \
    scikit-learn \
    numpy \
    scipy \
    matplotlib \
    seaborn \
    folium \
    tqdm \
    pandas \
    Pillow \
    jupyterlab \
    ipykernel \
    ipywidgets

WORKDIR /home/jovyan/work

EXPOSE 8888
