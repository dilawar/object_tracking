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

trajectory_ = [ ]
curr_loc_ = (100, 100)
static_features_ = defaultdict( int )
static_features_img_ = None
distance_threshold_ = 200

# To keep track of template coordinates.
bbox_ = [ ]

# Current frame
frame_ = None

# global window with callback function
window_ = "Mouse tracker"


# This is our template. Use must select it to begin with.
template_ = None


def onmouse( event, x, y, flags, params ):
    global curr_loc_, frame_, window_ 
    global bbox_
    global template_

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
            template_ = frame_[y0:y1,r0:r1]
            cv2.imshow( window_, frame_ )

    # Else user is updating the current location of animal.
    else:
        if event == cv2.EVENT_LBUTTONDOWN:
            curr_loc_ = (x, y)
            print( '[INFO] Current location updated to %s' % str( curr_loc_ ) )


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
                print( '[INFO] Dropping this frame' )
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
        print( '[WARN] not a good frame: too bright or dark' )
        return False
    if frame.mean( ) < 50 or frame.mean() > 200:
        print( '[WARN] not a good frame: not much variation' )
        return False
    return True

def fetch_n_frames_averaged( n = 1 ):
    global cap_ 
    frames = []
    for i in range( n ):
        ret, frame = cap_.read()
        frames.append( cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY) )
    meanF = np.uint8( np.mean( np.dstack(frames), axis = 2 ))
    print( meanF.shape )
    return meanF

def fetch_a_good_frame( drop = 0 ):
    global cap_
    for i in range( drop ):
        ret, frame = cap_.read()
    ret, frame = cap_.read()
    if ret:
        if is_a_good_frame( frame ):
            return toGrey( frame )
        else:
            return fetch_a_good_frame( )
    else:
        print( '[Warn] Failed to fetch a frame' )
        return None

def threshold_frame( frame ):
    mean, std = frame.mean(), frame.std( )
    thres = max(0, mean - 2*std)
    frame[ frame > thres ] = 255
    return frame

def find_edges( frame ):
    u, std = frame.mean(), frame.std()
    return cv2.Canny( frame, u, u + std, 11 )

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
    global template_
    h = 50
    c0, r0 = curr_loc_
    h = min( c0, r0, h)
    template_ = frame[ r0-h:r0+h, c0-h:c0+h ]
    cv2.imshow( 'template', template_ )
    cv2.waitKey( 1 )


def fix_current_location( frame ):
    """We have a hint of mouse location, now fix it by really locating the
    aninal
    """
    global curr_loc_
    global template_
    try:
        update_template( frame )
        res = cv2.matchTemplate( frame, template_, cv2.TM_SQDIFF_NORMED )
        minv, maxv, (y,x), maxl = cv2.minMaxLoc( res )
        c0, r0 = curr_loc_
        w, h = template_.shape
        maxMatchPoint = (y+w/2, x+h/2)
        # print( "loc", curr_loc_, " minl", maxMatchPoint )
        # cv2.rectangle( frame, (y,x), (y+w,x+h), 100, 2 )
        curr_loc_ = maxMatchPoint
        cv2.circle( frame, curr_loc_, 10, 255, 3)
        print( 'Successfully updated current loc', curr_loc_ )
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
    sumC = 0.0
    sumR = 0.0

    for p in points:
        (x,y) = p.ravel( )
        x, y = int(x), int(y)

        # We don't want points which are far away from current location.
        if distance( (x,y), curr_loc_ ) > distance_threshold_:
            continue 

        # if this point is in one of static feature point, reject it
        if static_features_img_[ y, x ] > 1:
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


def track( cur ):
    global curr_loc_ 
    global static_features_img_
    # Apply a good bilinear filter. This will smoothen the image but preserve
    # the edges.
    cur = cv2.bilateralFilter( cur, 5, 50, 50 )
    p0 = cv2.goodFeaturesToTrack( cur, 200, 0.1, 5 )
    insert_int_corners( p0 )
    draw_point( cur, p0, 1 )
    res = update_mouse_location( p0, cur )
    p1 = res[ 'contour' ]
    ellipse = res[ 'ellipse' ]
    if p1 is not None:
        for p in p1:
            (x, y) = p.ravel()
            cv2.circle( cur, (x,y), 10, 20, 2 )
    if ellipse is not None:
        # cv2.drawContours( cur, [p1], 0, 255, 2 )
        cv2.ellipse( cur, ellipse, 1 )

    display_frame( cur, 1 )
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

    return nFrames, fps 

def process( args ):
    global cap_
    global box_
    global curr_loc_, frame_
    global static_features_img_ 

    nFrames, fps = get_cap_props( )
    print( '[INFO] FPS = %f' % fps )

    static_features_img_ = np.zeros( frame_.shape )
    while True:
        totalFramesDone = -1
        try:
            totalFramesDone = cap_.get( cv2.cv.CV_CAP_PROP_POS_FRAMES ) 
        except Exception as e:
            totalFramesDone = cap_.get( cv2.CAP_PROP_POS_FRAMES ) 

        if totalFramesDone + 1 >= nFrames:
            print( '== All done' )
            break
        # prev = cur
        frame_ = fetch_a_good_frame( ) 
        assert frame_.any()
        # cur = threshold_frame( cur )
        track( frame_ )

        # After every 5 frame, Divide the static_features_img_.
        if totalFramesDone % 5 == 0:
            static_features_img_ = static_features_img_ / 5

def main(args):
    # Extract video first
    global cap_, frame_
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
        , required = False
        , default = 0
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

