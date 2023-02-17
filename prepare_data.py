import numpy as np
import os
# from test_modules.check_data_loss import datetime_from_str
import datetime
import math
from visualize import visualize

def re_write_data(data_path):
	"""
	load data in .txt and save in .npy after alignment. efficient.
	Args:
		data_path:
	"""
	data_imag = np.loadtxt(os.path.join(data_path, 'frame_buff_imag.txt'))
	data_real = np.loadtxt(os.path.join(data_path, 'frame_buff_real.txt'))
	times = np.loadtxt(os.path.join(data_path, 'times.txt'))
	times_dt = [datetime.datetime.fromtimestamp(time) for time in times]

	number = min(min(len(data_imag), len(data_real)), min(times_dt))
	print("imaginary file: {} lines, real file: {} lines, times: {} lines".format(len(data_imag), len(data_real), len(times_dt)))
	data_imag = data_imag[0:number]
	data_real = data_real[0:number]
	data_complex = data_real + 1j * data_imag  # compute complex number
	np.save(os.path.join(data_path, "data_complex.npy"), np.array(data_complex))


def load_seg_file(file, flag=False):
	"""
	load segment files from my android app
	Args:
		file:
		flag: whether it's mixed. like take on/off, sit down/stand up

	Returns: 2-d list of paris [datetime_begin, datetime_end]

	"""
	if not os.path.exists(file):
		print("{} doesn't exist.".format(file))
		return []
	with open(file) as f:
		lines = f.readlines()

	if len(lines) == 1:  # start:xxx,start:xxx,
		dt_str = lines[0].split(',')[:-1]
		date_times = [datetime.datetime.strptime(dt[6:], '%Y-%m-%d %H:%M:%S.%f') for dt in dt_str]
		if not flag:
			seg = [[date_times[i], date_times[i + 1]] for i in range(len(date_times) - 1)]
			return seg, []
		else:
			seg_1 = [[date_times[i], date_times[i + 1]] for i in range(0, len(date_times) - 1, 2)]
			seg_2 = [[date_times[i], date_times[i + 1]] for i in range(1, len(date_times) - 1, 2)]
			return seg_1, seg_2
	else:  # start:xxx, stop:xxx\n
		if flag:
			seg_1 = [[datetime.datetime.strptime(line.strip('\n').split(',')[0][6:], '%Y-%m-%d %H:%M:%S.%f'),
			        datetime.datetime.strptime(line.strip('\n').split(',')[1][5:], '%Y-%m-%d %H:%M:%S.%f')] for line in lines]
			seg_2 = [[datetime.datetime.strptime(lines[i].strip('\n').split(',')[1][5:], '%Y-%m-%d %H:%M:%S.%f'),
			          datetime.datetime.strptime(lines[i+1].strip('\n').split(',')[0][6:], '%Y-%m-%d %H:%M:%S.%f')] for i in range(len(lines)-1)]
			return seg_1, seg_2
		else:
			seg = [[datetime.datetime.strptime(line.split(',')[0][6:], '%Y-%m-%d %H:%M:%S.%f'),
			          datetime.datetime.strptime(line.split(',')[1][5:], '%Y-%m-%d %H:%M:%S.%f')] for line in lines]
			return seg, []


seg_label = {
	'2023-02-09-15-51-33.txt': 1,   # wash hands
	'2023-02-09-16-00-14.txt': 2,   # use toilet
	'2023-02-09-16-26-20.txt': 3,   # take on/off clothes
	'2023-02-09-16-42-48.txt': 3,
	'2023-02-09-17-56-33.txt': 1,
	'2023-02-09-18-03-16.txt': 2,
	'2023-02-09-18-16-49.txt': 3
}


def load_segments(seg_path='/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/segmentation'):
	"""
	load multiple segment files from my android app and organize them by labels
	Args:
		data_path:
		seg_path:

	Returns: segments of type [[datetime_start, datetime_stop]] for each class

	"""
	seg1 = []
	seg2 = []
	seg3 = []
	seg4 = []

	# load segment files
	# zh 1
	seg_file = os.path.join(seg_path, '2023-02-09-15-51-33.txt')
	seg1, _ = load_seg_file(seg_file, False)

	# zh 2
	seg_file = os.path.join(seg_path, '2023-02-09-16-00-14.txt')
	seg2, _ = load_seg_file(seg_file, False)

	# zh 3, 4
	seg_file = os.path.join(seg_path, '2023-02-09-16-26-20.txt')
	seg3, seg4 = load_seg_file(seg_file, True)

	seg_file = os.path.join(seg_path, '2023-02-09-16-42-48.txt')
	seg33, seg44 = load_seg_file(seg_file, True)
	seg3.extend(seg33)
	seg4.extend(seg44)

	# lmj 1
	seg_file = os.path.join(seg_path, '2023-02-09-17-56-33.txt')
	seg, _ = load_seg_file(seg_file, False)
	seg1.extend(seg)

	# lmj 2
	seg_file = os.path.join(seg_path, '2023-02-09-18-03-16.txt')
	seg, _ = load_seg_file(seg_file, False)
	seg2.extend(seg)

	# lmj 3, 4
	seg_file = os.path.join(seg_path, '2023-02-09-16-26-20.txt')
	seg33, seg44 = load_seg_file(seg_file, True)
	seg3.extend(seg33)
	seg4.extend(seg44)

	return seg1, seg2, seg3, seg4


def find_segment_indices(seg_dt_start, seg_dt_end, times_dt):
	"""
	traverse datetime of each data frame to find the segment fitting in seg_dt_start and seg_dt_end. return the indices.s

	Args:
		seg_dt_start: datetime
		seg_dt_end:
		times_dt:
		augment:

	Returns: indices of frames which is collected between datetime seg_dt_start, seg_dt_end

	"""
	i = 0
	data_indices = []
	while i < len(times_dt) and times_dt[i] < seg_dt_start:
		i += 1
	while i < len(times_dt) and times_dt[i] <= seg_dt_end:
		data_indices.append(i)
		i += 1
	return data_indices


# def augment(indices, length_of_times):
# 	gap = length_of_times - indices[-1]
# 	return [indices+i for i in range(1, gap)]


def downSample(indices, m):
	"""
	downsample and augmentation for one segemnt
	Args:
		indices: indices of one segment got from data_segment()
		m: frame rate in one segment

	Returns: 2-d list of indices

	"""
	ave_duration = math.floor(len(indices) / m)
	offset = [[math.floor(i * ave_duration) + indices[k] for i in range(m)] for k in range(0, len(indices)-ave_duration*(m-1))]
	# offset = [[math.floor(i * ave_duration) + indices[k] for i in range(m)] for k in range(0, 20)]
	return offset


def upSample(indices, m):
	diff = m - len(indices)
	suffix = [indices[-1]] * diff
	indices.extend(suffix)
	return indices


def segment_data(data_path, seg_path='/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/segmentation', sampleRate=50, augment_flag=False):
	"""
	load data frame and organize them by segments.
	Args:
		data_path:
		seg_path:
		sampleRate:
		augment_flag:

	Returns:

	"""
	data_complex = np.load(os.path.join(data_path, 'data_complex.npy'))
	times = np.loadtxt(os.path.join(data_path, 'times.txt'))
	times_dt = [datetime.datetime.fromtimestamp(time) for time in times]

	seg_all = [[], [], [], []]
	seg_all[0], seg_all[1], seg_all[2], seg_all[3] = load_segments(seg_path)

	data = []
	label = []
	for i in range(4):
		indices_i = []
		labels_i = []
		for segg in seg_all[i]:
			seg_start, seg_end = segg[0], segg[1]
			indices = find_segment_indices(seg_start, seg_end, times_dt)
			if len(indices) == 0:
				print("{}, {}".format(str(seg_start), str(seg_end)))
				continue
			if len(indices) >= sampleRate:
				indices_t = downSample(indices, sampleRate)
				indices_i.extend(indices_t)
				labels_i.extend([int(i + 1)] * len(indices_t))
			elif len(indices) > 0:
				indices_t = upSample(indices, sampleRate)
				indices_i.append(indices_t)
				labels_i.append(int(i+1))
		# np.save(os.path.join(data_path, "data_complex_segmented_{}.npy".format(i)), np.array(data_complex[tuple(indices_i), :]))
		# np.save(os.path.join(data_path, "label_segmented_{}.npy".format(i)), np.array(labels_i))

		data.extend(data_complex[tuple(indices_i), :])
		label.extend(labels_i)
	print(len(data))
	np.save(os.path.join(data_path, "data_complex_segmented.npy"), np.array(data))
	np.savetxt(os.path.join(data_path, "label_segmented.txt"), np.array(label))

	return np.array(data), np.array(label).astype(int)


def compute_range_doppler(data_path="/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/46-d7/"):
	"""
	compute range doppler for data organized by segments
	Args:
		data_path:
	"""
	# data, label = load_data('/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/46-d7/')

	data_complex = np.load(data_path + 'data_complex_segmented.npy')
	# label = np.loadtxt(data_path + 'label_segmented.txt')

	doppler_original = []
	doppler_gt = []
	for dc in data_complex:
		d_fft_ = np.abs(np.fft.fft(dc, axis=0)) # FFT
		d_fft = np.fft.fftshift(d_fft_, axes=0) # shift
		doppler_original.append(d_fft)

		# left, right = chose_central_rangebins(d_fft)
		# fft_gt = np.copy(d_fft[:, left:right])  # several central range-bins
		#
		# DISCARD_BINS=[23, 34, 25, 26]
		#
		# for j in DISCARD_BINS:  # normalize the central 3 rows
		# 	mean_j = np.mean(fft_gt[j, :])
		# 	fft_gt[j, :] -= mean_j
		#
		# # sum over range
		# fft_gt = np.sum(d_fft[:, left:right], axis=1)
		# doppler_gt.append(fft_gt)
	# doppler_gt = np.array(doppler_gt)
	doppler_original = np.array(doppler_original)
	np.save(os.path.join(data_path, 'doppler_original_segmented.npy'), doppler_original)
	# np.save(os.path.join(data_path, 'doppler_gt.npy'), doppler_gt)
	print(np.max(doppler_original))
	print(np.min(doppler_original))
	np.savetxt(os.path.join(data_path, "statistics.txt"), [np.max(doppler_original), np.min(doppler_original)])
	# print(np.max(doppler_gt))
	# print(np.min(doppler_gt))



if __name__ == "__main__":
	data_path = "/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/d0-0b/"
	seg_path = '/Users/liumengjing/Documents/HAR/ADL_data_collection/2023-02-09/segmentation'
	segment_data(data_path, seg_path)
	compute_range_doppler(data_path)
	# visualize(data_path)









