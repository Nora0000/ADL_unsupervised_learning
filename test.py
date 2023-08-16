import os.path

from helper import datetime_from_str
import numpy as np
from scipy.signal import find_peaks
import datetime
from datetime import timedelta, datetime


def function(ts_file, minutes_start, seconds_start, minutes_stop, seconds_stop, fr=24):
	"""
return the start datetime and stop datetime of each activity, in local timezone,
given the start time and stop time of the activity in the video and the frame timestamps (rgb_ts.txt) of the video.
	Args:
		ts_file:
		minutes_start:
		seconds_start:
		minutes_stop:
		seconds_stop:
		fr: frame rate of video
	"""
	# dur = timedelta(minutes=minutes, seconds=seconds)
	frame_num_start = (minutes_start * 60 + seconds_start) * fr
	frame_num_stop = (minutes_stop * 60 + seconds_stop) * fr
	with open(ts_file) as file:
		ts = file.readlines()
		if frame_num_start < 1:
			frame_num_start = 1
		if frame_num_stop > len(ts):
			print("frame_num_stop (= {}) > len(ts) (= {})".format(frame_num_stop, len(ts)))
			frame_num_stop = len(ts)
		dt_start = datetime.fromtimestamp(float(ts[int(frame_num_start) - 1]))
		dt_stop = datetime.fromtimestamp(float(ts[int(frame_num_stop) - 1]))
		print("start:{},stop:{}".format(str(dt_start), str(dt_stop)))
		# return dt


if __name__ == "__main__":
	segs = [
		[5, 39, 5, 47],
		[5, 49, 5, 59],
		[6, 16, 7, 1],
		[7, 2, 7, 9],
		[7,13,7,22],
		[7,26,7,33]
		
	]
	
	for seg in segs:
		function("/home/mengjingliu/ADL_unsupervised_learning/ADL_data/YhsHv0_ADL_1/rgb_ts.txt", seg[0], seg[1], seg[2], seg[3])

