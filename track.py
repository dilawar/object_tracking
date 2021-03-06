#!/usr/bin/env python

from __future__ import print_function 

"""
Locate mouse by template tracking.

"""
    
__author__           = "Me"
__copyright__        = "Copyright 2016, Me"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Me"
__email__            = ""
__status__           = "Development"

import cv2
import math
from collections import defaultdict
import numpy as np
import gnuplotlib as gpl

trajectory_ = [ ]
curr_loc_ = (100, 100)
static_features_ = defaultdict( int )
static_features_img_ = None
distance_threshold_ = 200
trajectory_file_ = None

# To keep track of template coordinates.
bbox_ = [ ]

frame_ = None # Current frame.
nframe_ = 0   # Index of currnet frame
fps_ = 1      # Frame per seocond

# global window with callback function
window_ = "Mouse tracker"

# This is our template. Use must select it to begin with.
template_ = None
template_size_ = None


def onmouse( event, x, y, flags, params ):
    global curr_loc_, frame_, window_ 
    global bbox_
    global template_, template_size_

    if template_ is None:
        # Draw Rectangle. Click and drag to next location then release.
        if event == cv2.EVENT_LBUTTONDOWN:
            bbox_ = []
            bbox_.append((x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            bbox_.append((x, y))

        if len( bbox_ ) == 2:
            print( 'bbox_ : %s and %s' % (bbox_[0], bbox_[1]) )
            cv2.rectangle( frame_, bbox_[0], bbox_[1], 100, 2)
            ((x0,y0),(x1,y1)) = bbox_ 
            template_size_ = (y1-y0, x1-x0)
            template_ = frame_[y0:y1,x0:x1]
            cv2.imshow( window_, frame_ )

    # Else user is updating the current location of animal.
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            curr_loc_ = (x, y)
            # print( '[INFO] Current location updated to %s' % str( curr_loc_ ) )


def toGrey( frame ):
    return cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY )


def display_frame( frame, delay = 40 ):
    global window_ 
    try:
        cv2.imshow( window_, frame )
        cv2.waitKey( delay )
    except Exception as e:
        print( '[warn] could not display frame' )
        print( '\t Error was %s' % e )


def clip_frame( frame, box ):
    (r1, c1), (r2, c2 ) = box
    return frame[c1:c2,r1:r2]


def initialize_template( ):
    global window_, frame_
    global template_, bbox_
    cv2.setMouseCallback(window_, onmouse)
    if template_ is None:
        while True:
            cv2.imshow( window_, frame_ )
            key = cv2.waitKey( 1 ) & 0xFF
            if key == ord( 'n' ):
                # print( '[INFO] Dropping this frame' )
                frame_ = fetch_a_good_frame( )
            elif key == ord( 'r' ):
                bbox_ = []
                template_ = None
            elif key == ord( 'q' ):
                break

def initialize_global_window( ):
    global window_ 
    cv2.namedWindow( window_ )

def is_a_good_frame( frame ):
    if frame.max( ) < 100 or frame.min() > 150:
        # print( '[WARN] not a good frame: too bright or dark' )
        return False
    if frame.mean( ) < 50 or frame.mean() > 200:
        # print( '[WARN] not a good frame: not much variation' )
        return False
    return True

def fetch_a_good_frame( drop = 0 ):
    global cap_
    global nframe_
    for i in range( drop ):
        ret, frame = cap_.read()
        nframe_ += 1
    ret, frame = cap_.read()
    if ret:
        if is_a_good_frame( frame ):
            return toGrey( frame )
        else:
            return fetch_a_good_frame( )
    else:
        print( "Can't fetch anymore. All done" )
        return None

def distance( p0, p1 ):
    x0, y0 = p0
    x1, y1 = p1
    return ((x0 - x1)**2 + (y0 - y1)**2) ** 0.5

def draw_point( frame, points, thickness = 2):
    for p in points:
        (x, y) = p.ravel()
        cv2.circle( frame, (x,y), 2, 30, thickness )
    return frame

def update_template( frame ):
    global curr_loc_ 
    global template_, template_size_
    h, w = template_size_ 
    c0, r0 = curr_loc_
    h = min( c0, r0, h, w)
    template_ = frame[ r0-h:r0+h, c0-h:c0+h ]
    # cv2.imshow( 'template', template_ )
    # cv2.waitKey( 1 )


def fix_current_location( frame ):
    """We have a hint of mouse location, now fix it by really locating the
    aninal
    """
    global curr_loc_, nframe_
    global template_
    global trajectory_
    try:
        update_template( frame )
        res = cv2.matchTemplate( frame, template_, cv2.TM_SQDIFF_NORMED )
        minv, maxv, (y,x), maxl = cv2.minMaxLoc( res )
        c0, r0 = curr_loc_
        w, h = template_.shape
        maxMatchPoint = (y+w/2, x+h/2)
        # cv2.circle( frame, curr_loc_, 5, 100, 5)
        curr_loc_ = maxMatchPoint
        cv2.circle( frame, curr_loc_, 10, 255, 3)
        trajectory_.append( curr_loc_ )
        print( '- Time %.2f, Current loc %s', ( nframe_/fps_, str(curr_loc_)))
        time = nframe_ / float( fps_ )
        # Append to trajectory file.
        # done, totalF, fps = get_cap_props( )
        with open( trajectory_file_, 'a' ) as trajF:
            c0, r0 = curr_loc_
            trajF.write( '%g %d %d\n' % (time, c0, r0) )

    except Exception as e:
        print( 'Failed with %s' % e )
        return 


def update_mouse_location( points, frame ):
    global curr_loc_
    global static_features_img_
    global distance_threshold_

    c0, r0 = curr_loc_ 
    res = {}
    newPoints = [ ]
    if points is None:
        return None, None
    sumC, sumR = 0.0, 0.0

    for p in points:
        (x,y) = p.ravel( )
        x, y = int(x), int(y)

        # We don't want points which are far away from current location.
        if distance( (x,y), curr_loc_ ) > distance_threshold_:
            continue 

        # if this point is in one of static feature point, reject it
        if static_features_img_[ y, x ] > 1.5:
            continue
        newPoints.append( (x,y) )
        sumR += y
        sumC += x

    newPoints = np.array( newPoints )
    ellipse = None
    try:
        if( len(newPoints) > 5 ):
            ellipse = cv2.fitEllipse( newPoints )
    except Exception as e:
        pass
    if len( newPoints ) > 0:
        curr_loc_ = ( int(sumC / len( newPoints )), int(sumR / len( newPoints)) )
        

    ## Fix the current location
    fix_current_location( frame )
    
    res[ 'ellipse' ] = ellipse 
    res[ 'contour' ] = newPoints

    return res

def insert_int_corners( points ):
    """Insert or update feature points into an image by increasing the pixal
    value by 1. If a feature point is static, its count will increase
    drastically.
    """
    global static_features_img_
    global distance_threshold_
    if points is None:
        return 
    for p in points:
        (x,y) = p.ravel()
        static_features_img_[ y, x ] += 1

def smooth( vec, N = 10 ):
    window = np.ones( N ) / N
    return np.correlate( vec, window, 'valid' )

def track( cur ):
    global curr_loc_ 
    global static_features_img_
    global trajectory_
    # Apply a good bilinear filter. This will smoothen the image but preserve
    # the edges.
    cur = cv2.bilateralFilter( cur, 5, 50, 50 )
    p0 = cv2.goodFeaturesToTrack( cur, 200, 0.1, 5 )
    insert_int_corners( p0 )
    draw_point( cur, p0, 1 )
    res = update_mouse_location( p0, cur )
    p1 = res[ 'contour' ]
    ellipse = res[ 'ellipse' ]
    # if p1 is not None:
        # for p in p1:
            # (x, y) = p.ravel()
            # cv2.circle( cur, (x,y), 10, 20, 2 )
    # if ellipse is not None:
        # cv2.drawContours( cur, [p1], 0, 255, 2 )
        # cv2.ellipse( cur, ellipse, 1 )
        # pass

    display_frame( cur, 1 )
    # Plot the trajectory
    # toPlot = zip(*trajectory_[-100:]) 
    if len( trajectory_ ) % 20 == 0:
        y, x = zip( *trajectory_ )
        # Smooth them
        cols, rows = [ smooth( a, 20 ) for a in [y,x] ]
        gpl.plot( cols, rows 
                , terminal = 'x11', _with = 'line' 
                # To make sure the origin is located at top-left. 
                , cmds  = [ 'set yrange [:] reverse' ]
                )
    return 


def get_cap_props( ):
    global cap_
    nFrame = 0
    try:
        nFrames = cap_.get( cv2.cv.CV_CAP_PROP_FRAME_COUNT )
    except Exception as e:
        nFrames = cap_.get( cv2.CAP_PROP_FRAME_COUNT )
    fps = 0.0
    try:
        fps = float( cap_.get( cv2.cv.CV_CAP_PROP_FPS ) )
    except Exception as e:
        fps = float( cap_.get( cv2.CAP_PROP_FPS ) )

    totalFramesDone = 0
    try:
        totalFramesDone = cap_.get( cv2.cv.CV_CAP_PROP_POS_FRAMES ) 
    except Exception as e:
        totalFramesDone = cap_.get( cv2.CAP_PROP_POS_FRAMES ) 

    return totalFramesDone, nFrames, fps 

def process( args ):
    global cap_
    global box_
    global curr_loc_, frame_, fps_
    global nframe_
    global static_features_img_ 

    nframe_, totalFrames, fps = get_cap_props( )
    print( '[INFO] FPS = %f' % fps )
    if fps > 1: fps_ = fps

    static_features_img_ = np.zeros( frame_.shape )
    while True:
        nframe_ += 1
        if nframe_ + 1 >= totalFrames:
            print( '== All done' )
            break

        frame_ = fetch_a_good_frame( ) 
        if frame_ is None:
            break

        assert frame_.any()
        track( frame_ )

        # After every 3 frame, Divide the static_features_img_ by its maximum
        # value. This way we don't over-estimate the static point. Sometime
        # animal may not move at all and if we don't do this, we will ignore all
        # good corners on the mouse.
        if nframe_ % 5 == 0:
            static_features_img_ /=  5.0

        print( '[INFO] Done %d frames out of %d' % ( nframe_, totalFrames ))
    print( '== All done' )

def main(args):
    # Extract video first
    global cap_, frame_
    global trajectory_file_ 

    trajectory_file_ = '%s_traj.csv' % args.file 
    with open( trajectory_file_, 'w' ) as f:
        f.write( 'time column row\n' )

    initialize_global_window( )
    cap_ = cv2.VideoCapture( args.file )
    assert cap_

    frame_ = fetch_a_good_frame( )

    # Let user draw rectangle around animal on first frame.
    initialize_template( )
    process( args )

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Detect eye blinks in given recording.'''
    parser = argparse.ArgumentParser(description=description)
    class Args: pass 
    args = Args()
    parser.add_argument('--file', '-f'
        , required = True
        , help = 'Path of the video file or camera index. default camera 0'
        )
    parser.add_argument('--verbose', '-v'
        , required = False
        , action = 'store_true'
        , default = False
        , help = 'Show you whats going on?'
        )
    parser.add_argument('--template', '-t'
        , required = False
        , default = None
        , type = str
        , help = 'Template file'
        )

    parser.add_argument('--col', '-c'
            , required = False 
            , type = int
            , help = 'Column of mouse'
            )
    parser.add_argument('--row', '-r'
            , required = False 
            , type = int
            , help = 'Row index of mouse'
            )
    parser.parse_args(namespace=args)
    main( args )

