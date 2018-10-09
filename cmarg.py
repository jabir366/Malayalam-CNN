#!/usr/bin/env python
import argparse
import os
import cv2
parser = argparse.argumentparser()
#parser.add_argument("square", help="display a square of a given number",
 #                   type=int)
parser.add_argument("file",help="just echoing")
args = parser.parse_args()
#print(args.square**2)
#print(args.echo)
def mplot(img,img2=none):

    cv2.namedwindow('img',cv2.window_normal)
    cv2.movewindow('img', 600,300)
    cv2.imshow('img',img)
    if img2 is not none:
        cv2.namedwindow('img2',cv2.window_normal)
        cv2.movewindow('img', 600,600)
        cv2.imshow('img2',img2)
    cv2.waitkey(0)
    cv2.destroyallwindows()
img=cv2.imread(args.file)
mplot(img)
