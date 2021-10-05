import cv2
import numpy as np
import os

from .contrast_types import ContrastTypes

class Fingerprint:
	def __init__(self, path, contrast_type):
		self.image = cv2.imread(path)
		self.image_grayscaled = self.image2greyscale()
		self.image_grey_contrasted = self.computeContrast(type=contrast_type)

		self.mask, self.masked_image = self.segmentation(self.image_grey_contrasted)
		
		self.computeContrast(type=ContrastTypes.MICHELSON)
		self.computeContrast(type=ContrastTypes.RMS)

	def segmentation(self, image):
		
		img_edge = cv2.Canny(image, 100, 200)
		cv2.imshow('Canny', img_edge)      

		# Morphology
		kernel = np.ones((25, 25), np.uint8)
		img_morph = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
		
		cv2.imshow('Morphology', img_morph)   

		# Contours 
		contours, _ = cv2.findContours(img_morph.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

		# Sort Contors by area -> remove the largest frame contour
		n = len(contours) - 1
		contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

		copy = image.copy()


		# Just for testing
		# Iterate through contours and draw the convex hull
		thres = 750
		for c in contours:
			if cv2.contourArea(c) < thres:
				continue
			hull = cv2.convexHull(c)
			cv2.drawContours(copy, [hull], 0, (0, 255, 0), 2)

		cv2.imshow('Convex Hull', copy)    

		return img_morph, None

	def computeContrast(self, type):
		result = None

		if type == ContrastTypes.CLAHE:	
			clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			result = clahe.apply(self.image_grayscaled)
			cv2.imshow('CLAHE contrast', result)

		elif type == ContrastTypes.MICHELSON:
			img_yuv = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)[:,:,0]
			min = np.min(img_yuv)
			max = np.max(img_yuv)

			contrast = (max-min)/(max+min)
			print(min,max,contrast)

		elif type == ContrastTypes.WEBER:
			pass

		else:
			contrast = self.image2greyscale().std()
			print(contrast)	

		return result

	def image2greyscale(self):
		return cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
		
	def show(self, name, scale=1.0):
		if scale != 1.0:
			width = int(self.image.shape[1] * scale)
			height = int(self.image.shape[0] * scale)
			dimensions = (width, height)
			self.image = cv2.resize(self.image, dimensions, interpolation=cv2.INTER_AREA)

		cv2.imshow(name, self.image)
