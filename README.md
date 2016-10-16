# Object Tracking 

This is an experimental project designed to track a moving animal in an arena.
This may not work out of the box in your case but I'd be happy to help.

# Demo 

See a demo [here](https://youtu.be/Eng82w9g9-w) This demo was created with version v0.0.3 of this script.


# Dependencies 

- python-opencv
- python-numpy 
- gnuplot 

# How to use 

When you run the command `python track.py -f /path/to/recorded_video`, you will be presented the 
first frame to locate the animal. You can drop as many frame as you like by pressing `n`. Once 
you see the animal in frame, use the mouse to draw a rectangle around the animal: press left mouse button
and drag it to another corner then release the button. Press `q` to start tracking.

During tracking, you can click on the animal to fix the location. A white cirlce 
represents the approximate location of animal (see the demo). 

When all frames are done, check the directory of video file, there must be a data file in csv format. First 
column represent the time, second the column index and third the row index of the animal.

# Help

Drop and email to dilawars@ncbs.res.in . Though you should raise an issue on this repo. Make sure you 
attack a fragment of video for me to test. 
