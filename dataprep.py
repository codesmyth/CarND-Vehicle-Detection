import os
import glob

basedir = 'vehicles/'

image_types = os.listdir(basedir)

cars = []
for imtype in os.listdir(basedir):
    cars.extend(glob.glob(basedir+imtype+'/*'))

print('Number of Vehicle Images found: ', len(cars))
with open( "cars.txt", 'w') as f:
    for fn in cars:
        f.write(fn+'\n')

basedir = 'non-vehicles/'
image_types = os.listdir(basedir)

notcars = []
for imtype in os.listdir(basedir):
    notcars.extend(glob.glob(basedir + imtype + '/*'))

print('Number of Non-vehicle Images found: ', len(notcars))
with open("notcars.txt", 'w') as f:
    for fn in notcars:
        f.write(fn + '\n')




