# Lane-Following
Use OpenCV for lane detection and steering

* lanes.py implements simplistic lane detection, but will not work for lanes that curve, hence is impractical.
* laneDetection.py implements a more general case that accommodates for lane curvature, and returns the magnitude of curvature as a value between -1 and 1 (capped).
