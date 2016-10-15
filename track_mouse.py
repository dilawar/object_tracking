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
curr_loc_ = None
static_features_ = defaultdict( int )
static_features_img_ = None
distance_threshold_ = 200
mouse_color_ = 10

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

def fetch_a_good_frame( color = False ):
    global cap_
    ret, frame = cap_.read()
    if ret:
        if is_a_good_frame( frame ):
            if color:
                return frame
            else:
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

def drop_frames( n = 1 ):
    global cap_
    for i in range( n ):
        cap_.read()

def get_rect( frame, point, d = 100 ):
    r0, c0 = point 
    patch = frame[max(0,c0-d):c0+d,max(0,r0-d):r0+d ]
    return patch

def isOnTheMouse( point, frame ):
    global mouse_color_
    return True
    # if color around the point is near to mouse color, we are good to go.
    mousePatch = get_rect( frame, point, 20 )
    print( mousePatch )
    meanColor = np.mean( mousePatch )
    print("Color", meanColor, mouse_color_ )
    if meanColor > mouse_color_ + 20:
        return False
    if meanColor < mouse_color_ - 20:
        return False
    return True
    

def update_mouse_location( flow, frame ):
    global curr_loc_
    global static_features_img_
    global distance_threshold_

    # get a patch of flow near current location.
    r0, c0 = curr_loc_
    print( "Curr location %s" % str(curr_loc_ ))
    patch = get_rect( flow, curr_loc_, 100 )
    patch = flow 
    # Find the place where maximum change in flow intestity and draw a circle
    # there. Thats probably is the mouse
    minV, maxV, minL, maxL = cv2.minMaxLoc( patch )
    while( not isOnTheMouse( maxL, frame ) ):
        print( 'New point', maxL )
        patch[maxL[1], maxL[0]] = maxV - 2
        minV, maxV, minL, maxL = cv2.minMaxLoc( patch )

    curr_loc_ = maxL
    cv2.circle( patch, maxL, 10, 255, 4 )

    cv2.imshow( 'patch', patch + frame )
    cv2.waitKey( 1 )


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


def track( cur, prev ):
    global curr_loc_ 
    global static_features_img_
    global hsv_
    # Apply a good bilinear filter. This will smoothen the image but preserve
    # the edges.
    cur = cv2.bilateralFilter( cur, 5, 50, 50 )
    prev = cv2.bilateralFilter( prev, 5, 50, 50 )
    flow = cv2.calcOpticalFlowFarneback( 
            prev, cur, None, 0.5, 2, 10, 3, 5, 1.2, 0
            )
    mag, ang = cv2.cartToPolar( flow[...,0], flow[...,1] )
    hsv_[...,0] = ang * 180 / np.pi / 2
    hsv_[...,2] = cv2.normalize( mag, None, 0, 255, cv2.NORM_MINMAX ) 
    hsvImg = toGrey( cv2.cvtColor( hsv_, cv2.COLOR_HSV2RGB ))
    update_mouse_location( hsvImg, cur )
    display_frame( cur, 1 )
    return 

def get_cap_props( ):
    global cap_
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

    return nFames, fps 

def process( args ):
    global cap_
    global box_, templates_
    global curr_loc_ 
    global static_features_img_ 
    global hsv_
    cap_ = cv2.VideoCapture( args.file )
    assert cap_
    nFames, fps = get_cap_props( )

    print( '[INFO] FPS = %f' % fps )
    cur = fetch_a_good_frame( color = True )
    hsv_ = np.zeros_like( cur )
    hsv_[...,1] = 255
    cur = toGrey( cur )
    curr_loc_ = cur.shape[1]/2, cur.shape[0] / 2
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
        prev = cur
        # drop_frames( 4 )
        cur = fetch_a_good_frame( ) 
        track( cur, prev )


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

