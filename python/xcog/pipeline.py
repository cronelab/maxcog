#!/usr/bin/env python

#==============================================================================#
# xcog.pipeline
# Based on maxcog.pipeline
# 2018
#==============================================================================#
# maxcog.pipeline
# v. 0.0.1
# 
# Functions for common tasks performed on LabeledArrays.
# 
# 2016 - Maxwell J. Collard
#==============================================================================#

"""Functions for common tasks performed on xarrays."""

import time
import unittest
import collections
import itertools
import functools

from collections import OrderedDict

import numpy                        as np
import numpy.lib.recfunctions       as recfunctions
import scipy.signal                 as sig
import scipy.stats.distributions    as dist

import pandas as pd
import xarray as xr


## === MAIN === ##


# TODO Add unit tests

if __name__ == '__main__':
    main()

def main ():
    pass


def transform( f, x, **kwargs ):
    '''Transforms x by f, leaving the axes in place
    
    Inputs:
    f - ndarray function to apply to x
    x - xarray containing the data
    **kjwargs - (Optional) Keyword arguments to pass to f
    '''

    ret_array = f( x, **kwargs )
    
    return xr.DataArray( ret_array,
                         coords = x.coords,
                         dims = x.dims )

def filtfilt( b, a, x, **kwargs ):
    '''xarray analogue of scipy.signal.filtfilt; runs along x's 'time' axis.
    
    Inputs:
    b, a - Filter coefficients passed to filtfilt
    x - xarray containing the data; must have a 'time' axis
    **kwargs - (Optional) Keyword arguments to pass to filtfilt
    
    Output is a xarray with the same axes as x.
    '''
    
    # Make sure we don't override axis in the kwargs
    kwargs.pop( 'axis', None )
    time_axis = x.get_axis_num( 'time' )
    
    def ff( x_ff, **kwargs ):
        return sig.filtfilt( b, a, x_ff,
                             axis = time_axis,
                             **kwargs )
                            
    return transform( ff, x, **kwargs )




def timefreq_fft( x, **kwargs ):
    '''xarray analogue of scipy.signal.spectrogram; runs along x's 'time' axis.
    
    Inputs:
    x - xarray containing the data
    **kwargs - (Optional) Keyword arguments passed to spectrogram
    
    Output is a xarray with the same axes as x except for:
        'time' - Determined by the window properties; uses the values returned
            by spectrogram
        'frequency' - Added based on the values returned by spectrogram
    '''
    
    # Make sure we don't override axis in the kwargs
    kwargs.pop( 'axis', None )
    time_axis = x.get_axis_num( 'time' )
    iter_axes = [ dim for dim in x.dims
                  if dim != 'time' ]
    
    # Test with first sample along each non-time axis
    test_slice = { dim: 0 for dim in iter_axes }
    f_spec, t_spec, ret_test = sig.spectrogram( x[test_slice], **kwargs )
    
    # Allocate return value
    ret_coords = dict( x.coords )
    ret_coords['time'] = t_spec + np.array( x.time[0] )
    ret_coords['frequency'] = f_spec
    
    ret_dims = x.dims + ('frequency',)
    
    ret = xr.DataArray( np.zeros( tuple( len( ret_coords[dim] ) for dim in ret_dims ) ),
                        coords = ret_coords,
                        dims = ret_dims )
    
    # Iterate over non-time axes
    for cur_slice in dict( zip( iter_axes,
                                itertools.product( [ range( len( x.get_index( dim ) ) )
                                                     for dim in iter_axes ] ) ) ):
        
        f_cur, t_cur, ret_cur = sig.spectrogram( x[cur_slice], **kwargs )
        ret[cur_slice] = ret_cur
    
    return ret

def decimate ( x, q, **kwargs ):
    '''Labeled analogue of scipy.signal.decimate; runs along x's 'time' axis.

    Inputs:
    x - xarray containing the data; must have a 'time' axis
    q - Decimation factor passed to decimate
    **kwargs - (Optional) Keyword arguments passed to decimate

    Output is a xarray with the same axes as x except for 'time', which
    is subsampled accordingly.
    '''
           
    # Make sure we don't override axis in the kwargs
    kwargs.pop( 'axis', None )
    time_axis = x.get_axis_num( 'time' )
    
    def deci (x_deci, **kwards):
        return sig.decimate (x_deci.data, q, 
                            axis = time_axis,
                            **kwargs)
    return transform ( deci, x, **kwards)


def map_x ( f, x ):
    '''Maps each element of the xarray x through a function f.

    Inputs:
    x - xarray containing the data to act on
    f - A one-argument function that works elementwise on a numpy ndarray

    Output is a xarray with the same axes as x.
    '''

    ret_array = f(x.data)
    
    return xr.DataArray( data = ret_array,
                         coords = x.coords,
                         dims = x.dims)


def mean_x ( x, axis = None ):
    '''x analogue of np.mean, running along the specified axis(es).

    Inputs:
    x - xarray containing the data to average
    axis - (Optional) A string or sequence of strings, specifying the name(s) of
        the axis(es) to average over
        Default: Runs over the last axis in x.axes.keys()

    Output is a xarray with the same axes as x except for 'time', which
    is subsampled accordingly.
    '''
    
    # Default: Use last axis
    if axis is None:
        axis = x.dims[-1]
        
    # Strip out the axes we average over from the return value
    # Calculate the mean over such axes
    return x.mean(dim = axis)

def std_x ( x, axis = None ):
    '''x analogue of np.std, running along the specified axis(es).

    Inputs:
    x - LabeledArray containing the data
    axis - (Optional) A string or sequence of strings, specifying the name(s) of
        the axis(es) to take the standard deviation over
        Default: Runs over the last axis in x.axes.keys()

    Output is a xarray with the same axes as x except for 'time', which
    is subsampled accordingly.
    '''
    
    # Default: Use last axis
    if axis is None:
        axis = x.dims[-1]

    # Strip out the axes we average over from the return value
    # Calculate the std over such axes
    return x.std(dim = axis)

def hilbert_labeled ( x, **kwargs ):
    '''x analogue of scipy.signal.hilbert; runs along x's 'time' axis.

    Inputs:
    x - xarray containing the data; must have a 'time' axis, and probably
        shouldn't have a 'frequency' axis
    **kwargs - (Optional) Keyword arguments passed to hilbert

    Output is a xarray with the same axes as x, but containing the analytic signal
    '''
    # Make sure we don't override axis in the kwargs
    kwargs.pop( 'axis', None )
    time_axis = x.get_axis_num( 'time' )
    
    ## Compute analytic signal
    def hilb(x_hilb, **kwargs):
        return sig.hilbert(x_hilb, axis = time_axis,
                          **kwargs)
    
    return transform(hilb, x, **kwargs)
