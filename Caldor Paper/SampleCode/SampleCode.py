def read_gpkg(fnm,layer='perimeter'):
    ''' read gpkg data
    op: 'perimeter', 'fireline', 'newfirepix'
    '''
    import geopandas as gpd
    import pandas as pd

    gdf = gpd.read_file(fnm,layer=layer)

    return gdf

def read_pickle(fnm):
    import pickle

    # load data
    with open(fnm,'rb') as f:
        data = pickle.load(f)
    return data

def read_netcdf(fnm):
    import xarray as xr

    ds = xr.open_dataset(fnm)

    return ds

def read_csv(fnm):
    import pandas as pd

    flist = pd.read_csv(fnm)

    return flist


if __name__ == "__main__":

    # read snapshot
    gdf = read_geojson('../2020/Snapshot/20200929AM.gpkg',layer='perimeter')
    gdf_FL = read_geojson('../2020/Snapshot/20200929AM.gpkg',layer='fireline')
    gdf_NFP = read_geojson('../2020/Snapshot/20200929AM.gpkg',layer='newfirepix')

    # read largefire
    gdf_lf = read_geojson('../2020/Largefire/LargeFires_2020.gpkg',layer='perimeter')
    gdf_lf_FL = read_geojson('../2020/Largefire/LargeFires_2020.gpkg',layer='fireline')
    gdf_lf_NFP = read_geojson('../2020/Largefire/LargeFires_2020.gpkg',layer='newfirepix')

    # read summary
    summary_ts = read_netcdf('../2020/summary/fsummary_20201231PM.nc')
    flist_heritage = read_csv('../2020/summary/Flist_heritage_2020.csv')
    flist_large = read_csv('../2020/summary/Flist_large_2020.csv')

    # read serialization (pickle file)
    allfire_obj = read_pickle('../2020/Serialization/20200929AM.pkl')  # read the allfire object
    number_of_activefires = allfire_obj.number_of_activefires  # extract sample attribute of allfire object
    fid = 3519
    fire = allfire_obj.fires[fid]  # extract a fire object from the allfire object
    t_st,ted = fire.t_st, fire.t_ed  # extract sample attributes of the fire object
    pixel = fire.pixels[10]    # extract a pixel object from the fire object
    p_lat,p_lon = pixel.loc    # extract sample attributes from the pixel object
