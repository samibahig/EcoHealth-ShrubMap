FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Dépendances système GDAL complètes
RUN apt-get update && apt-get install -y \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    python3-gdal \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Variables GDAL
ENV GDAL_VERSION=3.4.1
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Base packages
RUN pip install --no-cache-dir numpy pandas scipy matplotlib seaborn tqdm Pillow

# ML
RUN pip install --no-cache-dir scikit-learn

# Geo packages avec GDAL système
RUN pip install --no-cache-dir \
    GDAL==$(gdal-config --version) \
    pyproj \
    shapely \
    fiona \
    rasterio \
    geopandas

# Autres
RUN pip install --no-cache-dir pystac-client laspy folium

# Jupyter
RUN pip install --no-cache-dir jupyterlab ipykernel ipywidgets

WORKDIR /home/jovyan/work
EXPOSE 8888
