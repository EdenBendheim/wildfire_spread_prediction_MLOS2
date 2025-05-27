
'''
Author: Yang Chen (yang.chen@uci.edu)
This code is open-source and can be freely used/adapted for research purposes
'''

# ------------------------------------------------------------------------------
# 1. FUNCTIONS AND CONSTANTS
# ------------------------------------------------------------------------------

# temporal and spatial distances for fire object definition
maxoffdays = 5   # fire becomes inactive after this number of consecutive days without active fire detection
fpbuffer = 200     # buffer use to determine fire line pixels (deg), ~200m
flbuffer = 500     # buffer for fire line pixels (radius) to intersect fire perimeter (deg), ~500m
area_VI = 0.141    # km2, area of each 375m VIIRS pixel
FTYP = {0:'Other', 1:'Urban', 2:'Forest wild', 3:'Forest manage', 4:'Shrub wild', 5:'Shrub manage', 6:'Agriculture'}      # fire type names

# some functions
def cal_centroid(data):
    ''' Calculate the centroid of a list of points
    Parameters
    ----------
    data : list of [lat,lon]
        point location

    Returns
    -------
    xct : float
        lat of centroid
    yct : float
        lon of centroid
    '''
    x, y = zip(*(data))
    l = len(x)
    xct, yct =  sum(x) / l, sum(y) / l

    return xct,yct

def calConcHarea(hull):
    ''' calculate area given the concave hull (km2)

    Parameters
    ----------
    hull : geometry, 'Polygon' | 'MultiPoint'
        the hull for the fire

    Returns
    -------
    farea : float
        the area (km2) of the polygon enclosed by the vertices
    '''
    import math
    EARTH_RADIUS_KM = 6371.0  # earth radius, km

    farea = hull.area   # area in deg**2

    if farea > 0:
        lat = hull.bounds[1]  # latitude in deg

        # convert deg**2 to km2
        farea = farea*EARTH_RADIUS_KM**2*math.cos(math.radians(lat))*math.radians(1)**2

    return farea

def addbuffer(geom_ll,vbuf):
    ''' add a geometry in geographic projection with a buffer in meter

    Parameters
    ----------
    geom_ll : shapely geometry
        the geometry (in lat /lon)
    vbuf : float
        the buffer value (in meter)
    '''
    # the buffer in geographic projection is calculated using the centroid lat value
    EARTH_RADIUS_KM = 6371.0  # earth radius, km
    import numpy as np
    lat = geom_ll.centroid.y
    ldeg = (EARTH_RADIUS_KM*np.cos(np.deg2rad(lat))*1000*2*np.pi/360)
    vbufdeg = vbuf/ldeg    # convert vbuf in m to degs
    geom_ll_buf = geom_ll.buffer(vbufdeg)

    return geom_ll_buf

def doMultP(locs):
    ''' deirve a MultipPolygon (bufferred MultiPoint) shape from given fire locations

    Parameters
    ----------
    locs : list (nx2)
        latitude and longitude values of all fire pixels

    Returns
    -------
    multP : MultiPolygon object
        calculated shape
    '''
    from shapely.geometry import MultiPoint
    VIIRSbuf = 187.5   # fire perimeter buffer (deg), corresponding to 375m/2 at lat=30


    # MultiPoint shape
    multP = MultiPoint([(lon,lat) for lat,lon in locs])

    # Add buffer to form MultiPolygon shape
    # multP = multP.buffer(VIIRSbuf)
    multP = addbuffer(multP, VIIRSbuf)

    return multP

# b. Object-realted utility functions
# The time step in the module is now defined as a list (year, month, day, ampm).
#   The following functions are used to convert times between different formats.
#   t : time steps, tuple (year,month,day,ampm)
#   d : date, datetime.date()
#   ampm : ampm, str()
#   dt : time steps, datetime.datetime()
def t_nb(t,nb='next'):
    ''' Calculate the next or previous time step (year, month, day, ampm)
    Parameters
    ----------
    t : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM' for present time
    nb : str, 'next'|'previous'
        option to extract next or previous time step

    Returns
    -------
    t_out : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM' for next/previous time
    '''
    from datetime import date, timedelta

    # the next time step
    if nb == 'next':
        # if current time is 'AM', set next time as the current day and 'PM'
        if t[-1] == 'AM':
            t_out = list(t[:-1])
            t_out.append('PM')
        # if current time is 'PM', set next time as the following day and 'AM'
        else:
            d = date(*t[:-1])
            d_out = d + timedelta(days=1)
            t_out = [d_out.year,d_out.month,d_out.day,'AM']

    # the previous time step
    elif nb == 'previous':
        # if current time is 'PM', set previous time as the current day and 'AM'
        if t[-1] == 'PM':
            t_out = list(t[:-1])
            t_out.append('AM')
        # if current time is 'AM', set previous time as the previous day and 'PM'
        else:
            d = date(*t[:-1])
            d_out = d + timedelta(days=-1)
            t_out = [d_out.year,d_out.month,d_out.day,'PM']
    return t_out

def t_dif(t1,t2):
    ''' calculate the time difference between two time steps
    Parameters
    ----------
    t1 : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM' for time 1
    t2 : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM' for time 2

    Returns
    -------
    dt : float
        time difference in days (t2-t1), half day as 0.5
    '''
    from datetime import date

    # calculate the day difference
    d1 = date(*t1[:-1])
    d2 = date(*t2[:-1])
    dt = (d2-d1).days

    # adjust according to ampm difference
    if t1[-1] != t2[-1]:
        if t1[-1] == 'PM':
            dt -= 0.5
        else:
            dt += 0.5
    return dt

def t2d(t):
    ''' convert a t tuple to date and ampm
    Parameters
    ----------
    t : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM'

    Returns
    -------
    d : datetime date
        date
    ampm : str, 'AM'|'PM'
        ampm indicator
    '''
    from datetime import date

    d = date(*t[:-1])     # current date, datetime date
    ampm = t[-1]          # current ampm, 'AM'|'PM'

    return d, ampm

def t2dt(t):
    ''' convert a t tuple to datetime
    Parameters
    ----------
    t : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM'

    Returns
    -------
    dt : datetime datetime
        datetime
    '''
    from datetime import datetime
    dlh = {'AM':0,'PM':12}
    dhl = {0:'AM',12:'PM'}

    dt = datetime(*t[:-1],dlh[t[-1]])

    return dt

def d2t(year,month,day,ampm):
    ''' convert year, month, day, ampm to a t tuple
    Parameters
    ----------
    year : int
        year
    month : int
        month
    day : int
        day
    ampm : str, 'AM'|'PM'
        ampm indicator

    Returns
    -------
    t : list, (int,int,int,str)
        the year, month, day and 'AM'|'PM'
    '''
    t = [year,month,day,ampm]
    return t

def dt2t(dt):
    ''' convert datetime to a t tuple
    Parameters
    ----------
    dt : datetime datetime
        datetime
    Returns
    -------
    t : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM'
    '''
    dlh = {'AM':0,'PM':12}
    dhl = {0:'AM',12:'PM'}

    t = [dt.year,dt.month,dt.day,dhl[dt.hour]]
    return t

def ftrange(firstday,lastday):
    ''' get datetime range for given first and last t tuples (both ends included)

    Parameters
    ----------
    firstday : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM'
    lastday : tuple, (int,int,int,str)
        the year, month, day and 'AM'|'PM'

    Returns
    -------
    trange : pandas date range
        date range defined by firstday and lastday
    '''
    import pandas as pd

    trange = pd.date_range(t2dt(firstday),t2dt(lastday),freq='12h')
    return trange

# ------------------------------------------------------------------------------
# 2. LAYERS OF OBJECTS
# ------------------------------------------------------------------------------

# a. Object - Allfires
class Allfires:
    """ Class of allfire events at a particular time step
    """

    # initilization
    def __init__(self,t):
        ''' Initiate the object with current time
        Parameters
        ----------
        t : tuple, (int,int,int,str)
            the year, month, day and 'AM'|'PM'
        '''
        # time
        self.t = t_nb(t,nb='previous') # itialize the object with previous time step

        # a list of Fire objects
        self.fires = []

        # list of fire ids which has changes at the current time step
        self.fids_expanded = []   # a list of ids for fires with expansion at current time step
        self.fids_new = []        # a list of ids for all new fires formed at current time step
        self.fids_merged = []     # a list of ids for fires with merging at current time step
        self.fids_invalid = []    # a list of ids for fires invalidated at current time step

        # cumulative recordings
        self.heritages = []       # a list of fire heritage relationships (source, target)

    # properties
    @property
    def cday(self):
        ''' Datetime date of current time step
        '''
        from datetime import date
        return date(*self.t[:-1])

    @property
    def ampm(self):
        ''' Ampm indicator of current time step
        Parameters
        ----------
        ampm : str, 'AM'|'PM'
           the ampm option calculated from t
        '''
        return self.t[-1]

    @property
    def number_of_fires(self):
        ''' Total number of fires (active and inactive) at this time step
        '''
        return len(self.fires)

    @property
    def fids_active(self):
        ''' List of active fire ids
        '''
        return [f.id for f in self.fires if f.isactive is True]

    @property
    def number_of_activefires(self):
        ''' Total number of active fires at this time step
        '''
        return len(self.fids_active)

    @property
    def activefires(self):
        ''' List of active fires
        '''
        return [self.fires[fid] for fid in self.fids_active]

    @property
    def fids_valid(self):
        ''' List of valid (non-invalid) fire ids
        '''
        return [f.id for f in self.fires if f.invalid is False]

    @property
    def number_of_validfires(self):
        ''' Total number of valid fires at this time step
        '''
        return len(self.fids_valid)

    @property
    def validfires(self):
        ''' List of valid fires
        '''
        return [self.fires[fid] for fid in self.fids_valid]

    @property
    def fids_updated(self):
        ''' List of fire id which is updated at this time step
            (expanded, new, merged, invalid)
        '''
        fids_updated = list(set(self.fids_expanded+self.fids_new+
                                self.fids_merged+self.fids_invalid))
        return fids_updated

    @property
    def fids_ne(self):
        ''' List of fire id which is newly formed or expanded
               at this time step
        '''
        fids_ne = sorted(set(self.fids_expanded+self.fids_new))
        return fids_ne

# b. Object - Fire
class Fire:
    """ Class of a single fire event at a particular time step
    """


    # properties
    @property
    def cday(self):
        ''' Current day (datetime date)
        '''
        from datetime import date
        return date(*self.t[:-1])
#
    @property
    def ampm(self):
        ''' Current ampm flag, 'AM'|'PM'
        '''
        return self.t[-1]

    @property
    def duration(self):
        ''' Time difference between first and last active fire detection
        '''
        duration = t_dif(self.t_st,self.t_ed) + 0.5
        return duration

    @property
    def t_inactive(self):
        ''' Time difference between current time and the last active fire detection
        '''
        t_inactive = t_dif(self.t_ed,self.t)
        return t_inactive

    @property
    def isactive(self):
        ''' Fire active status
        '''
        # invalidated fires are always inactive
        if self.invalid:
            return False
        # otherwise, set to True if no new pixels detected for 5 consecutive days
        return (self.t_inactive <= maxoffdays)

    @property
    def locs(self):
        ''' List of fire pixel locations (lat,lon)
        '''
        return [p.loc for p in self.pixels]

    @property
    def n_pixels(self):
        ''' Total number of fire pixels'''
        return len(self.pixels)

    @property
    def newlocs(self):
        ''' List of new fire pixels locations (lat,lon)
        '''
        return [p.loc for p in self.newpixels]

    @property
    def n_newpixels(self):
        ''' Total number of new fire pixels
        '''
        return len(self.newpixels)

    @property
    def extlocs(self):
        ''' List of exterior fire pixel locations (lat,lon)
        '''
        return [p.loc for p in self.extpixels]

    @property
    def n_extpixels(self):
        ''' Total number of exterior fire pixels
        '''
        return len(self.extpixels)

    @property
    def ignlocs(self):
        ''' List of fire pixel locations (lat,lon) at ignition time step
        '''
        return [p.loc for p in self.ignpixels]

    @property
    def n_ignpixels(self):
        ''' Total number of ignition fire pixels
        '''
        return len(self.ignpixels)

    @property
    def farea(self):
        ''' Fire spatail size of the fire event (km2)
        '''
        # get hull
        fhull = self.hull

        # If no hull, return area calculated from number of pixels
        if fhull is None:
            return self.n_pixels * area_VI
        # otherwise, use calConcHarea to calculate area,
        #   but no smaller than area_VI (sometimes calculated hull area is very mall)
        else:
            return max(calConcHarea(fhull),area_VI)

    @property
    def pixden(self):
        ''' Fire pixel density (number of pixels per km2 fire area)
        '''
        farea = self.farea
        if farea > 0:
            return self.n_pixels/farea
        else:
            return 0

    @property
    def meanFRP(self):
        ''' Mean FRP of the new fire pixels
        '''
        frps = [p.frp for p in self.newpixels]
        if len(frps) > 0:
            m = sum(frps)/len(frps)
        else:
            m = 0
        return m

    @property
    def centroid(self):
        ''' Centroid of fire object (lat,lon)
        '''
        # get hull
        fhull = self.hull

        if fhull is not None: # centroid of the hull
            cent = (fhull.centroid.y,fhull.centroid.x)
        else: # when no hull, use the centroid of all pixels
            cent = cal_centroid(self.locs)
        return cent

    @property
    def ftype(self):
        ''' Fire type (as defined in FireConsts) derived using LCTmax and stFM1000
        '''
        # get the dominant land cover type
        LCTmax = self.LCTmax

        # determine the fire type using the land cover type and stFM1000
        if LCTmax in [11,31]:   # 'Water', 'Barren' -> 'Other'
            return 0
        elif LCTmax in [23]:    # 'Urban' -> 'Urban'
            return 1
        elif LCTmax in [82]:    # 'Agriculture' -> 'Agriculture'
            return 6
        elif LCTmax in [42]:    # 'Forest' ->
            stFM1000 = self.stFM1000
            if stFM1000 > 12:        # 'Forest manage'
                return 3
            else:                  # 'Forest wild'
                return 2
        elif LCTmax in [52,71]:    # 'Shurb', 'Grassland' ->
            stFM1000 = self.stFM1000
            if stFM1000 > 12:        # 'Shrub manage'
                return 5
            else:                  # 'Shrub wild'
                return 4

    @property
    def ftypename(self):
        ''' Fire type name
        '''
        return FTYP[self.ftype]

    @property
    def fperim(self):
        ''' Perimeter length of fire hull
        '''
        # get hull
        fhull = self.hull

        if fhull is None:  # if no hull, return zero
            perim = 0
        else:  # otherwise, use the hull length
            perim = fhull.length
        return perim

    @property
    def flinepixels(self):
        ''' List of all fire pixels near the fire perimeter (fine line pixels)
        '''
        from shapely.geometry import Point, MultiLineString

        # get hull
        fhull = self.hull

        if fhull is None: # if no hull, return empty list
            return []
        else:  # otherwise, extract the pixels nearl the hull
            # if hull is a polygon, return new pixels near the hull
            if fhull.type == 'Polygon':
                # lr = fhull.exterior.buffer(fpbuffer)
                lr = addbuffer(fhull.exterior, fpbuffer)
                nps = self.newpixels
                return [p for p in nps if lr.contains(Point(p.loc[1],p.loc[0]))]

            # if hull is a multipolygon, return new pixels near the hull
            elif fhull.type == 'MultiPolygon':
                nps = self.newpixels
                # mlr = MultiLineString([x.exterior for x in fhull]).buffer(fpbuffer)
                mlr = MultiLineString([x.exterior for x in fhull])
                mlr = addbuffer(mlr,fpbuffer)
                return [p for p in nps if mlr.contains(Point(p.loc[1],p.loc[0]))]

    @property
    def fline(self):
        ''' Active fire line MultiLineString shape (segment of fire perimeter with active fires nearby)
        '''
        from shapely.geometry import Polygon,Point,MultiPoint,MultiLineString

        if len(self.flinepixels)==0:
            return None

        # get fireline pixel locations
        flinelocs = [p.loc for p in self.flinepixels]
        flinelocsMP = FireVector.doMultP(flinelocs)

        # get the hull
        fhull = self.hull

        # calculate the fire line
        if fhull is None: # if no hull, return None
            return None
        else:  # otherwise, create shape of the active fire line
            if fhull.type == 'MultiPolygon':
                # extract exterior of fire perimeter
                mls = MultiLineString([plg.exterior for plg in fhull])
                # return the part which intersects with  bufferred flinelocsMP
                # return mls.intersection(flinelocsMP.buffer(flbuffer))
                flinelocsMP_buf = addbuffer(flinelocsMP,flbuffer)
                return mls.intersection(flinelocsMP_buf)

            elif fhull.type == 'Polygon':
                mls = fhull.exterior
                # return mls.intersection(flinelocsMP.buffer(flbuffer))
                flinelocsMP_buf = addbuffer(flinelocsMP,flbuffer)
                return mls.intersection(flinelocsMP_buf)
            else:  # if fhull type is not 'MultiPolygon' or 'Polygon', return flinelocsMP
                return flinelocsMP

    @property
    def flinelen(self):
        ''' The length of active fire line
        '''
        try:
            flinelen = self.fline.length
        except:
            flinelen = 0

        return flinelen

# # c. Object - FirePixel
class FirePixel:
    """ class of an acitve fire pixel, which includes
        loc : location (lat, lon)
        frp : fire radiative power
        t : time (y,m,d,ampm) of record
        origin : the fire id originally recorded (before merging)
    """
