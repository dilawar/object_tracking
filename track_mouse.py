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

# global window with callback function
window_ = "Mouse tracker"

def onmouse( event, x, y, flags, params ):
    global curr_loc_
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

def generate_box( (c,r), width, height ):
    if width < 0: 
        width = 10
    if height < 0 : 
        height = 10
    leftCorner = ( max(0,c - width / 2), max(0, r - height / 2 ) )
    rightCorner = (leftCorner[0] + width, leftCorner[1] + height)
    return leftCorner, rightCorner 

def apply_template( frame, tmp ):
    tr, tc = tmp.shape    # Rows and cols in template.
    res = cv2.matchTemplate( frame, tmp, cv2.TM_SQDIFF_NORMED )
    minmax = cv2.minMaxLoc( res )
    minv, maxv, minl, maxl = minmax
    return minl

def initialize_global_window( ):
    global window_ 
    cv2.namedWindow( window_ )
    cv2.setMouseCallback(window_, onmouse)

def is_far_from_last_good_location( loc ):
    global lastGoodLocation_ 
    if lastGoodLocation_ is None:
        return False
    x1, y1 = lastGoodLocation_ 
    x2, y2 = loc 
    dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5 
    if dist < 5.0:
        return True
    return False


def track_using_templates( frame ):
    global lastGoodTemplate_
    global lastGoodLocation_ 

    # if last good location of mice is not known, then we apply the first
    # template and get it.
    if lastGoodTemplate_ is not None:
        indx = lastGoodTemplate_[0]
        ranges = np.mod( 
                np.arange( indx, indx + len( templates_ ), 1 )
                , len( templates_ ) 
                )
    else:
        ranges = np.arange( 0, len( templates_ ) )

    for i in ranges:
        tmp = templates_[i]
        tr, tc = tmp.shape    # Rows and cols in template.
        loc = apply_template( frame, tmp )
        if is_far_from_last_good_location( loc ):
            continue 
        else:
            lastGoodLocation_ = loc
            lastGoodTemplate_ = (i, tmp)
            break
    r, c = lastGoodLocation_
    print( "Last known good location is %s" % str( lastGoodLocation_ ) )
    return frame[c-100:c+100,r-100:r+100]


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

def on_the_mouse( p, frame ):
    w = 20
    c, r = p
    rect = frame[c-w:c+w,r-w:r+w]

def distance( p0, p1 ):
    x0, y0 = p0
    x1, y1 = p1
    return ((x0 - x1)**2 + (y0 - y1)**2) ** 0.5

def draw_point( frame, points, thickness = 2):
    for p in points:
        (x, y) = p.ravel()
        cv2.circle( frame, (x,y), 2, 30, thickness )
    return frame


def update_mouse_location( points ):
    global curr_loc_
    global static_features_
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
        if distance( (x,y), curr_loc_ ) > 100:
            continue 

        # if this point is in one of static feature point, reject it
        if static_features_[ (x,y) ] > 1:
            print( 'x', end = '')
            continue
        newPoints.append( (x,y) )
        sumR += y
        sumC += x

    newPoints = np.array( newPoints )
    ellipse = None
    try:
        ellipse = cv2.fitEllipse( newPoints )
    except Exception as e:
        pass
    if len( newPoints ) > 0:
        curr_loc_ = ( int(sumC / len( newPoints )), int(sumR / len( newPoints)) )
    
    res[ 'ellipse' ] = ellipse 
    res[ 'contour' ] = newPoints
    return res

def insert_int_corners( points ):
    """Insert or update feature points into an image by increasing the pixal
    value by 1. If a feature point is static, its count will increase
    drastically.
    """
    global static_features_img_
    global static_features_
    if points is None:
        return 
    for p in points:
        (x,y) = p.ravel()
        static_features_[ (x,y) ] += 1
        # static_features_img_[ y, x ] += 1


def track( cur ):
    global curr_loc_ 
    global static_features_img_
    # Apply a good bilinear filter. This will smoothen the image but preserve
    # the edges.
    cur = cv2.bilateralFilter( cur, 3, 50, 50 )
    p0 = cv2.goodFeaturesToTrack( cur, 200, 0.01, 5 )

    insert_int_corners( p0 )
    draw_point( cur, p0, 1 )

    res = update_mouse_location( p0 )
    p1 = res[ 'contour' ]
    ellipse = res[ 'ellipse' ]
    if p1 is not None:
        for p in p1:
            (x, y) = p.ravel()
            cv2.circle( cur, (x,y), 10, 20, 2 )
    if ellipse is not None:
        cv2.drawContours( cur, [p1], 0, 255, 2 )
        # cv2.ellipse( cur, ellipse, 1 )
    cv2.circle( cur, curr_loc_, 10, 255, 3)
    display_frame( cur, 1 )
    # cv2.imshow( 'static features', static_features_img_ )
    return 
    # Find a contour
    prevE = find_edges( prev )
    curE = find_edges( cur )
    img = curE - prevE
    cnts, hier = cv2.findContours( img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE )
    cnts = filter( ismouse, cnts )
    cv2.drawContours( img, cnts, -1, 255, 3 )
    display_frame( img, 1)
    return 
    p1, status, err = cv2.calcOpticalFlowPyrLK( prev, cur, p0 )
    mat = cv2.estimateRigidTransform( p0, p1, False )
    # print cv2.warpAffine( curr_loc_, mat, dsize=(2,1) )
    if mat is not None:
        dx, dy = mat[:,2]
        da = math.atan2( mat[1,0], mat[0,0] )
        trajectory_.append( (dx, dy, da) )
        print( "Transformation", dx, dy, da )
        curr_loc_ = (curr_loc_[0] - int(dy), curr_loc_[1] - int(dx))

def process( args ):
    global cap_
    global box_, templates_
    global curr_loc_ 
    global static_features_img_ 
    cap_ = cv2.VideoCapture( args.file )

    nFrame = 0
    try:
        nFames = cap_.get( cv2.cv.CV_CAP_PROP_FRAME_COUNT )
    except Exception as e:
        nFames = cap_.get( cv2.CAP_PROP_FRAME_COUNT )
    fps = 0.0
    try:
        fps = float( cap_.get( cv2.cv.CV_CAP_PROP_FPS ) )
    except Exception as e:
        fps = float( cap_.get( cv2.CAP_PROP_FPS ) )
    print( '[INFO] FPS = %f' % fps )
    cur = fetch_a_good_frame( )
    static_features_img_ = np.zeros( cur.shape )
    # cur = threshold_frame( cur )
    while True:
        totalFramesDone = -1
        try:
            totalFramesDone = cap_.get( cv2.cv.CV_CAP_PROP_POS_FRAMES ) 
        except Exception as e:
            totalFramesDone = cap_.get( cv2.CAP_PROP_POS_FRAMES ) 

        if totalFramesDone + 1 >= nFames:
            print( '== All done' )
            break
        # prev = cur
        cur = fetch_a_good_frame( ) 
        assert cur.any()
        # cur = threshold_frame( cur )
        track( cur )

def main(args):
    # Extract video first
    initialize_global_window( )
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

