import cv2
import numpy as np
import pandas as pd
import sys
import os

# new_frame is the variable with all the frames that need to be sent across
# though it is being written into a video to be sent to save bandwidth usage

def decode_fourcc(v):
  v = int(v)
  return "".join([chr((v >> 8 * i) & 0xFF) for i in range(4)])

scale = float(sys.argv[2])

inp = cv2.VideoCapture(sys.argv[1])
fold = os.path.splitext(os.path.basename(sys.argv[1]))[0]
ok, new_frame = inp.read()
size = (new_frame.shape[1], new_frame.shape[0])
fps = inp.get(cv2.CAP_PROP_FPS)
cod = decode_fourcc(inp.get(cv2.CAP_PROP_FOURCC))
ss = (int(size[0]/scale), int(size[1]/scale))
print(ss)
print(fps)
out = cv2.VideoWriter(sys.argv[3]+"/res."+os.path.splitext(os.path.basename(sys.argv[1]))[1],cv2.VideoWriter_fourcc(*cod), fps, ss)
fc = 1
while ok:
	new_frame = cv2.resize(new_frame, ss, cv2.INTER_AREA)
	out.write(new_frame)
	ok, new_frame = inp.read()
	fc += 1

out.release()
inp.release()