import sys
import os
import cv2

tes = ["1 N", "1 Y", "2 N", "2 Y", "4 N", "4 Y"]
comp = [0.01, 0.02, 0.5, 0.35, 0.8, 0.65]
netw = [1.02, 2.96, 3.3, 3.75, 13.95, 15.3]

comcost = float(sys.argv[2])	#Cost per hour of compute in dollars
netwcost = float(sys.argv[3])	#Cost per MB of transfer in dollars
# memcost = float(sys.agrv[4])

vid = sys.argv[1]

inp = cv2.VideoCapture(vid)
frame_count = float(inp.get(cv2.CAP_PROP_FRAME_COUNT))
siz = os.stat(vid).st_size

n = []

for i in range(6):
	co = (siz * netwcost / netw[i])/(1024*1024) + (frame_count * comp[i] * comcost)/3600 
	n.append(co)

strat = tes[n.index(min(n))]

print("2 Y")

os.system("python client.py " + vid + " 2 Y")