import rasterio


def load_raster(path):

    with rasterio.open(path) as src:

        data = src.read([1, 4, 5])
        profile = src.profile
        profile.update(count=3)
        return data, profile


def write_raster(data, output_path, profile, **kwargs):

    depth, width, height = data.shape
    profile["height"] = height
    profile["width"] = width
    profile["count"] = depth

    with rasterio.open(output_path, "w", **profile, **kwargs) as dst:

        dst.write(data)
