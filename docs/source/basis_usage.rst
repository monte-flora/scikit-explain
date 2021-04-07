How To Use
============

Introduction
---------------

The five most commonly used routines can be summarized as:

- :meth:`wrf.getvar` - Extracts WRF-ARW NetCDF variables and 
  computes diagnostic variables that WRF does not compute (e.g. storm 
  relative helicity). This is the routine that you will use most often.
  
- :meth:`wrf.interplevel` - Interpolates a three-dimensional field to a 
  horizontal plane at a specified level using simple (fast) linear 
  interpolation (e.g. 850 hPa temperature).
  
- :meth:`wrf.vertcross` - Interpolates a three-dimensional field to a vertical 
  plane through a user-specified horizontal line (i.e. a cross section).
  
- :meth:`wrf.interpline` - Interpolates a two-dimensional field to a 
  user-specified line.
  
- :meth:`wrf.vinterp` - Interpolates a three-dimensional field to 
  user-specified  'surface' levels (e.g. theta-e levels). This is a smarter, 
  albeit slower, version of :meth:`wrf.interplevel`. 
