#!/usr/bin/env python 

import skimage as sk
import skimage.io as sk_io
import skimage.transform as sk_trans
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sci_nd

def main():
	scale = 5
	padding_in_blocks = 200
	oahu = make_oahu(scale, padding_in_blocks)
	oahu = quantize(oahu)
	sk_io.imsave('oahu_mc_{}.png'.format(scale), oahu)
	plt.imshow(oahu, cmap=plt.get_cmap('prism'))
	plt.show()
	exit(0)
	#sk_io.show()
	#sk_io.imshow(quantize(oahu))
	#


	# Resize
	oahu_small = sk_trans.resize(oahu, (np.array(oahu.shape) * 0.02).astype(np.int64))

	print('saving extra extra small')
	sk_io.imsave('oahu_mc_extra_small.png', quantize(oahu_small))

	print('saving normal')
	sk_io.imsave('oahu_mc.png', quantize(oahu))

	# Resize
	oahu_big = sk_trans.resize(oahu, (np.array(oahu.shape) * horizontal_scale).astype(np.int64))

	print('saving big')
	sk_io.imsave('oahu_mc_big.png', quantize(oahu_big))

	# Display
	#sk_io.imshow(oahu_horizontal_scaled)
	#sk_io.show()



def make_oahu(lol=5, padding_in_blocks=100, sea_depths_o=10, sea_level_o=19, asdf=236):

	sea_depths_o -= 0.05 # Fixing rounding-related problems
	max_limit_o = 255.999
	vertical_scale = 1/lol
	horizontal_scale = 10/lol
	padding = padding_in_blocks / horizontal_scale
	padding = int(padding)

	# Read in raw values
	oahu = sk_io.imread('oahu.tif')
	# Add padding 
	oahu = np.pad(oahu, padding, mode='constant', constant_values=np.min(oahu))
	sea_pixels = oahu == np.min(oahu)
	land_pixels = np.logical_not(sea_pixels)

	# Compute distance field

	sea_falloff = np.zeros(oahu.shape)
	sea_falloff[sea_pixels] = 1
	sea_falloff = sci_nd.morphology.distance_transform_edt(sea_falloff)
	sea_falloff = np.minimum(sea_falloff, padding)
	sea_falloff /= padding
	sea_falloff = 1-sea_falloff # 1 is land, 0 is deepest part
	
	oahu[sea_pixels] = -1 # Replace sea with -1
	oahu_equalized = sk.exposure.equalize_hist(oahu)

	# Find the soft cutoff range
	# Divide into linear pixels and equalized pixels based off soft cutoff
	linear_pixels = (oahu * vertical_scale + sea_level_o) <= asdf
	equalized_pixels = np.logical_not(linear_pixels)
	linear_pixels = np.logical_and(land_pixels, linear_pixels)
	equalized_pixels = np.logical_and(land_pixels, equalized_pixels)

	# Scale the linear pixels
	oahu *= vertical_scale
	# Add sea level requirement
	oahu += sea_level_o
	oahu[sea_pixels] = sea_falloff[sea_pixels] * (sea_level_o - 0.05 - sea_depths_o) + sea_depths_o

	# Compute the remaining space
	remaining_space = max_limit_o - np.max(oahu[linear_pixels])
	print('Remaining space: ', remaining_space)

	# Equalize the remaining parts, if necessary
	if np.any(equalized_pixels):
		oahu_equalized -= np.min(oahu_equalized[equalized_pixels])
		oahu_equalized /= np.max(oahu_equalized[equalized_pixels])
		oahu_equalized *= max_limit_o - np.max(oahu[linear_pixels])
		oahu_equalized += np.max(oahu[linear_pixels])
		oahu[equalized_pixels] = oahu_equalized[equalized_pixels]
	else:
		print("No equalization necessary!")

	# Show new min and max
	print(np.min(oahu))
	print(np.max(oahu))



	oahu_big = sk_trans.resize(oahu, (np.array(oahu.shape) * horizontal_scale).astype(np.int64))
	return oahu_big
	
# Quantize
def quantize(img):
	
	shape = np.array(img.shape, dtype=np.float64)
	shape /= 128
	shape = np.ceil(shape)
	shape = shape * 128
	shape = shape.astype(np.int64)
	
	new_img = np.zeros(shape)
	new_img[:,:] = np.min(img)
	
	new_img[:img.shape[0],:img.shape[1]] = img
	new_img = np.floor(new_img)
	
	new_img = new_img.astype(np.uint8)
	return new_img
main()
