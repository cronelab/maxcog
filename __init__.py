#!/usr/bin/env python

#==============================================================================#
# xcog
# v. 0.0.1
# 
# ECoG analysis with xarrays.
# 
# 2018 - Maxwell J. Collard
#==============================================================================#

"""ECoG analysis with xarrays."""

__appname__ = "xcog"
__author__  = "Max Collard (itsthefedora)"
__version__ = "0.0.1"
__license__ = "CC BY-NC-SA"


## === IMPORTS === ##


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


# TODO Kludgey way to make things work without dependencies for testing

try:
    from pbar import pbar
except:
    def pbar( x, task = None ): return x

try:
    from h5eeg import H5EEGEvents
    EVENT_DTYPE = H5EEGEvents.EVENT_DTYPE
except:
    EVENT_DTYPE = [('name', '<U32'), ('start_idx', '<i4'), ('duration', '<i4')]
    

## === MAIN === ##


# TODO Add unit tests

if __name__ == '__main__':
    main()

def main ():
    pass


## === MISCELLANEOUS DOODADS === ##


def make_event ( name = None, start_idx = 0, duration = 0 ):
    '''Conveniently creates an event compatible with H5EEG's format'''
    if name == None:
        return np.array( [], dtype = EVENT_DTYPE )
    return np.array( [(name, start_idx, duration)], dtype = EVENT_DTYPE )[0]

def label_eeg ( eeg ):
    '''Convert an H5EEGDataset directly into a LabeledArray.
    CAUTION: Resulting array may be 'UGE!'''
    
    ret_axes = {}
    
    ret_axes['time'] = eeg.samp_to_sec( np.arange( len( eeg ) ) )
    ret_axes['channel'] = eeg.get_labels()
    
    n_time = len( ret_axes['time'] )
    n_channel = len( ret_axes['channel'] )

    # This actually causes the drive read!
    ret_array = eeg[:, :]
    
    ret = xr.DataArray( np.zeros( (n_time, n_channel ) ),
                        coords = ret_axes,
                        dims = ('time', 'channel') )
    return ret


## === CLASSES === ##


class EventFrame ( object ):
    '''Provides an abstraction of the notion of "locking to an event"
    EventFrame relies upon a H5EEGDataset to serve as a data source. It has two
    primary modes of operation: trials, which returns a generator that extracts
    each trial individually, and extract, which pulls 
    '''

    def __init__ ( self, eeg, events, window ):
        '''Initializes an EventFrame for given data, locking events, and
        extraction window
        Inputs:
        eeg - H5EEGDataset to serve as the data source
        events - ndarray of h5eeg event records (i.e.,
            dtype = H5EEGEvents.EVENT_DTYPE) to serve as t=0 for the frame
        window - extraction window of form (start, stop), in seconds
        '''

        self.eeg        = eeg
        self.events     = events
        self.window     = window

    def _t_slice ( self, x ):
        '''Convenience method to generate a slicing array for a window centered
        around a particular sample, x
        '''
        return range( int( round( x + self.eeg.get_rate() * self.window[0] ) ),
                      int( round( x + self.eeg.get_rate() * self.window[1] ) ) )

    def _axes ( self, multiple_trials = True, channels = None ):
        '''Convenience method to create an axes object compatible with
        LabeledArray
        Inputs:
        multiple_trials - (Optional) If False, the trial axis is omitted
            Default: True
        '''

        ret_coords = {}
        ret_dims = []
        
        ret_coords['time'] = np.linspace( self.window[0],self.window[1], len( self._t_slice( 0 ) ) )
        ret_coords['channel'] = channels if channels != None else self.eeg.get_labels()
        ret_dims = ['time', 'channel']
        
        if multiple_trials:
            # TODO Reinstate this when extra metadata (added below) becomes available
            # ret_coords['trial'] = np.array( np.arange( 1, len( self.events ) + 1 ), dtype = [ ( 'val', int ) ] )
            ret_coords['trial'] = np.arange( 1, len( self.events ) + 1 )
            ret_dims += ['trial']
        
        return ret_coords, ret_dims


    def trials ( self, channels = slice( None ) ):
        '''Returns a generator that pulls out the data from each event into a
        LabeledArray
        Inputs:
        channels - (Optional) Subset of channels to be extracted, in a format
            compatible with H5EEGDataset's slicing
            Default: All channels; i.e., slice(None)
        Output is a generator; each returned item is a LabeledArray of signature
        ('time', 'channel')
        '''

        # TODO Figure out how to get length caching to work
        # Cache length
        #ret.__length_hint__ = lambda : len( self.events )

        ret_coords, ret_dims = self._axes( multiple_trials = False )
        
        return ( xr.DataArray( self.eeg[self._t_slice( e['start_idx'] ), channels],
                               coords = ret_coords,
                               dims = ret_dims )
                 for e in self.events )

    def extract ( self ):
        '''Pulls out the data from all events into a LabeledArray. Each event is
        placed into its own entry along the 'trial' axis of the returned array.
        Output is a LabeledArray of signature ('time', 'channel', 'trial')
        '''

        # TODO Get this to work without copying ...

        # Preallocate return value
        ret_coords, ret_dims = self._axes()
        ret_array = np.zeros( tuple( len( ret_coords[axis] ) for axis in ret_dims ) )
        i = 0
        event_names = []
        for event in pbar( self.events ):
            # TODO implicitly assumes that trial is last axis for internal representation
            ret_array[:, :, i] = self.eeg[self._t_slice( event['start_idx'] ), :]
            event_names.append( event['name'] )
            i += 1
        
        # TODO This breaks something right now because it returns a MaskedArray
        # ret_coords[ 'trial' ] = recfunctions.append_fields( ret_coords['trial'], 
        #                                                     'event_name', 
        #                                                     np.array( event_names ) )
        
        return xr.DataArray( ret_array,
                             coords = ret_coords,
                             dims = ret_dims )


#