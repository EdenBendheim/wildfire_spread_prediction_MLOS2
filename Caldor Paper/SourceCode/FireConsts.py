""" FireConsts
This is the module containing all constants used in this project
"""

# project directories
import os
dirhome = os.environ.get('HOME')   # home directory
dirpjcode = dirhome + '/Code/'     # project code directory
dirpjdata = dirhome + '/Data/'     # output data directory
dirextdata = dirhome + '/ExtData/' # exterior input data directory

# parameters used for fire pixel clustering
EARTH_RADIUS_KM = 6371.0  # earth radius, km
SPATIAL_THRESHOLD_KM = 1  # threshold of spatial distance (between pixels, km) used to do fire initial clustering
MAX_THRESH_CENT_KM = 50   # threshold of spatial distance (between clusters centroid, km) used to filter nearby clusters

# fire type and visualization parameters
FTYP = {0:'Other', 1:'Urban', 2:'Forest wild', 3:'Forest manage', 4:'Shrub wild', 5:'Shrub manage', 6:'Agriculture'}      # fire type names
FTYPCLR = {0:'grey', 1:'rosybrown', 2:'darkolivegreen', 3:'olive', 4:'saddlebrown', 5:'sandybrown', 6:'darkviolet'}        # colors used for each fire type

# temporal and spatial distances for fire object definition
maxoffdays = 5   # fire becomes inactive after this number of consecutive days without active fire detection
CONNECTIVITY_THRESHOLD_KM = [1, 1, 5, 5, 5, 5, 1] # km, corresponding to fire type (forest/shrub: 5km; other: 1km)

# alpha parameter
valpha = 1000      # 1000 m (default)
stralpha = '1km'   # string of alpha parameter

# lengths and areas related to VIIRS pixl data
VIIRSbuf = 187.5   # fire perimeter buffer (deg), corresponding to 375m/2 at lat=30
fpbuffer = 200     # buffer use to determine fire line pixels (deg), ~200m
flbuffer = 500     # buffer for fire line pixels (radius) to intersect fire perimeter (deg), ~500m
extbuffer = 1000   # buffer to define interior/exterior region, 1000 m
area_VI = 0.141    # km2, area of each 375m VIIRS pixel

# parameter used for determine large fires
falim = 4    # large fire area threshold (in km2)

# diagnostic data name and types saved in geojson files (in addition to geometries)
dd = {'fid':'int',                  # id
      'clat':'float',               # centroid latitude   -> centroid[0]
      'clon':'float',               # centroid longitude  -> centroid[1]
      'ftype':'int',                # fire type
      'isactive':'int',             # active status
      'invalid':'int',              # invalid status
      'n_pixels':'int',             # number of total pixels
      'n_newpixels':'int',          # number of new pixels
      'farea':'float',              # fire size
      'fperim':'float',             # fire perimeter length
      'flinelen':'float',           # active fire front line length
      'duration':'float',           # fire duration
      'pixden':'float',             # fire pixel density
      'meanFRP':'float',            # mean FRP of the new fire pixels
      'tst_year':'int',             # t_st[0]
      'tst_month':'int',
      'tst_day':'int',
      'tst_ampm':'str',
      'ted_year':'int',             # t_ed[0]
      'ted_month':'int',
      'ted_day':'int',
      'ted_ampm':'str',
      }
