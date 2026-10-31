import rasterio
import numpy as np
from os import path

features = rasterio.open('data\\raw\\gems-geodawn-numerical-features.tif')

def load_data(path):
    raster = rasterio.open(path)
    print(type(raster.read(1)))
    bands = [raster.read(i) for i in range(1, raster.count + 1)]
    bands = [np.where(band != raster.nodata, band, np.nan) for band in bands] # Remove nodata values from each band
    bands_array = np.array(bands)   
    print(f"load {raster.name}:", bands_array.shape)
    return bands_array

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    data_features = load_data('data\\raw\\gems-geodawn-numerical-features.tif')
    submission_example = load_data('data\\raw\\example_submission.tif')
    plt.imshow(submission_example[0])
    plt.show()
    existing_faults = load_data('data\\raw\\existing_faults.tif')