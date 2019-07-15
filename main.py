import argparse

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from socket import create_connection

from os import path
from os import sys

from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage



# class isOnline(argparse._StoreTrueAction):
#     _checked = False

#     def __call__(self, parser, namespace, values, option_string=None):
#         super().__call__(parser, namespace, values)
#         if not isOnline._checked:
#             try:
#                 create_connection(("www.google.com", 80))
#                 isOnline._checked = True
#             except OSError:
#                 msg = 'There is no internet connection!'
#                 raise argparse.ArgumentTypeError(msg)


def isOnline():
	checked = False
	if not checked:
		try:
			create_connection(("www.google.com", 80))
			isOnline = True
			checked  = True
		except OSError:
			isOnline = False

	return isOnline



def doesImageExist(image):
	if path.isfile(image):
		return image
	else:
		msg = 'Image does not exist!'
		raise argparse.ArgumentTypeError(msg)


def getNumberOfFaces(image):
	face_cascade     = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_alt.xml')

	cvImage          = cv.imread(image)
	gray             = cv.cvtColor(cvImage, cv.COLOR_BGR2GRAY)

	# facesCoordinates = face_cascade.detectMultiScale(gray, 1.31, 6)
	facesCoordinates = face_cascade.detectMultiScale(gray, 1.30, 6)

	numberOfFaces    = len(facesCoordinates)

	cv.destroyAllWindows()

	return numberOfFaces, facesCoordinates, cvImage


def createImageOfFaces(cvImage, facesCoordinates, fileName):
	for (x,y,w,h) in facesCoordinates:
		cv.rectangle(cvImage, (x, y), (x+w, y+h), (255,255,255), 3)

	plt.imsave(fileName, cv.cvtColor(cvImage, cv.COLOR_BGR2RGB))
	cv.destroyAllWindows()
	return



def getNamesFromResponse(response):
	regions = response['outputs'][0]['data']['regions']
	names   = []
	for region in regions:
		name  = region['data']['face']['identity']['concepts'][0]['name']
		name  = ' '.join(map(lambda x: x.capitalize(), name.split())) # Capitalizing the initials
		value = region['data']['face']['identity']['concepts'][0]['value']
		if value < 0.6: # Case of not accurate prediction
			name = 'May not be accurate: ' + name
		names.append(name)
	return names



def getCelebrityName(image):
	app      = ClarifaiApp(api_key='f4206dba9ebf43009b40b41fa9dce5ef')
	model    = app.models.get('celeb-v1.3')
	clImage  = ClImage(filename=image)
	response = model.predict([clImage])
	names    = getNamesFromResponse(response)
	return names


def getInfo(image):
	app      = ClarifaiApp(api_key='f4206dba9ebf43009b40b41fa9dce5ef')
	model    = app.models.get('demographics')
	clImage  = ClImage(filename=image)
	response = model.predict([clImage])
	regions  = response['outputs'][0]['data']['regions']
	infos    = []
	for region in regions:
		age       = region['data']['face']['age_appearance']['concepts'][0]['name']
		gender    = region['data']['face']['gender_appearance']['concepts'][0]['name']
		ethnicity = region['data']['face']['multicultural_appearance']['concepts'][0]['name']
		infos.append((age, gender, ethnicity))
	return infos



argumentParser  = argparse.ArgumentParser(
	prog        = 'Face recognition',
	description = 'A simple CLI face recognition app'
	)

argumentParser.add_argument(
	'-c', '--count',
	action  = 'store_true',
	default = False,
	help    = 'prints the number of faces in the image, and creates an image with rectangles surrounding the faces detected'
	)

argumentParser.add_argument(
	'-w', '--who',
	default = False,
	action  = 'store_true',
	help    = 'tries to guess the name of the celebrity using the Clarifai API, requires internet connection'
	)

argumentParser.add_argument(
	'-i', '--info',
	default = False,
	action  = 'store_true',
	help    = 'analyzes the demographics of the person using the Clarifai API (beta), requires internet connection'
	)

argumentParser.add_argument('image', type=doesImageExist)

try:
	args = argumentParser.parse_args()
except argparse.ArgumentTypeError as ex:
	print(ex)
	sys.exit()

if args.count:
	numberOfFaces, facesCoordinates, cvImage = getNumberOfFaces(args.image)
	print('Faces found:', numberOfFaces)
	if numberOfFaces != 0:
		fileName = path.splitext(args.image)
		fileName = fileName[0] + ' FACE_DETECT' + fileName[1]
		createImageOfFaces(cvImage, facesCoordinates, fileName)


if args.who:
	if isOnline():
		try:
			names = getCelebrityName(args.image)
			for name in names:
				print(name)
		except KeyError:
			print('Cannot find any faces')
	else:
		print('There is no internet connection!')


if args.info:
	if isOnline():
		try:
			infos = getInfo(args.image)
			for info in infos:
				(age, gender, ethnicity) = info
				print('Age      :', age)
				print('Gender   :', gender)
				print('Ethnicity:', ethnicity)
		except KeyError:
			print('Cannot find any faces')


	else:
		print('There is no internet connection!')