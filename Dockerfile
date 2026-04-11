FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# Dépendances système
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libproj-dev \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Installer pip packages un par un pour isoler les erreurs
RUN pip install --no-cache-dir numpy pandas scipy matplotlib seaborn tqdm Pillow
RUN pip install --no-cache-dir scikit-learn
RUN pip install --no-cache-dir torch==1.13.1 torchvision==0.14.1
RUN pip install --no-cache-dir rasterio fiona pyproj shapely geopandas
RUN pip install --no-cache-dir pystac-client
RUN pip install --no-cache-dir laspy
RUN pip install --no-cache-dir folium
RUN pip install --no-cache-dir jupyterlab ipykernel ipywidgets

WORKDIR /home/jovyan/work
EXPOSE 8888
