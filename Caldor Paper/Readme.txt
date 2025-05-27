========================== Fire Events Data Suite (FEDS) =======================

This is the FEDS dataset for the 2012-2020 fire seasons in California.

--------------------------------------------------------------------------------

The dataset provides four compressed files for each year:

(i) [Serialization.tar.gz]
Contains all Pickle (.pkl) files that store serializations of the half-daily
allfires objects;

(ii) [Snapshot.tar.gz]
Contains GeoPackage (.gpkg) files that store major fire attributes and geometries
at half-daily time step, as well as a GeoPackage file that store the final
geometry and attributes of all fires;

(iii) [Largefire.tar.gz]
Contains GeoPackage files of large fire time series;

(iv) [Summary.tar.gz]
Include several year-end summary files in the formats of NetCDF and CSV.

--------------------------------------------------------------------------------

Two sets of accompanying python scripts are also included in the package:

(i) [SourceCode.tar.gz]
The source code used for tracking fire event spreading and deriving FEDS.

(ii) [SampleCode.tar.gz]
Sample python code to read the data files in FEDS.

--------------------------------------------------------------------------------

[COPYRIGHT]: This is an open-source dataset and can be free used for research
purposes

[CITATION]: Chen et al., California wildfire spread derived using VIIRS satellite
observations and an object-based tracking system, Scientific Data, 2022

[CONTACTS]: yang.chen@uci.edu

=================================== End ========================================
