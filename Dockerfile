FROM jupyter/scipy-notebook:python-3.10
 
USER root
 
# GDAL + système
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*
 
# Packages géospatiaux
RUN pip install --no-cache-dir \
    rasterio>=1.3.0 \
    geopandas>=0.13.0 \
    shapely>=2.0.0 \
    pyproj>=3.4.0 \
    fiona \
    pystac-client \
    laspy>=2.3.0 \
    folium>=0.15.0 \
    geemap>=0.30.0 \
    earthengine-api>=0.1.370 \
    geoai>=0.1.0
 
# PyTorch + deep learning
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    segmentation-models-pytorch>=0.3.3 \
    timm>=0.9.0 \
    albumentations==1.3.1
 
# ML + science
RUN pip install --no-cache-dir \
    scikit-learn>=1.2.0 \
    scikit-image>=0.20.0 \
    scipy>=1.10.0 \
    numpy>=1.24.0 \
    xgboost>=1.7.0 \
    imbalanced-learn>=0.10.0 \
    shap>=0.41.0
 
# Visualisation + utilitaires
RUN pip install --no-cache-dir \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    plotly>=5.14.0 \
    pandas>=2.0.0 \
    tqdm>=4.65.0 \
    requests>=2.28.0 \
    joblib>=1.2.0 \
    psutil>=5.9.0 \
    gdown>=4.7.0
 
USER jovyan
WORKDIR /home/jovyan/work
EXPOSE 8888
