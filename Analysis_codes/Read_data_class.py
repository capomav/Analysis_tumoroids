import numpy as np
import os
import time
import glob
from numba import jit, njit, float64
import pandas
import gzip


## Taken form the SAMoSA -Analysis library developed by Silke Henkes
## Refer for further details https://github.com/silkehenkes/SAMoSA


class ReadData:
  
	def __init__(self, filename,dialect):
		if filename.split('.')[-1] == 'gz':
			names = ['id', 'type', 'flag', 'radius', 'x','y','z', 'vx', 'vy', 'vz', 'nx', 'ny', 'nz']
			self.datafile = gzip.open(filename, newline='') # type: ignore
			line = self.datafile.readline()
			self.header_names = 0
			if line.startswith("#"): # type: ignore
				self.header_names = line[1:].strip().split()
			self.datafile.seek(0)	

		else:
			self.datafile = open(filename, newline='')
			line = self.datafile.readline()
			self.header_names = 0
			names = ['id', 'type', 'flag', 'radius', 'x','y','z', 'vx', 'vy', 'vz', 'nx', 'ny', 'nz']
			if line.startswith("#"):
				self.header_names = line[1:].strip().split()
			else:
				self.header_names = names
			self.datafile.seek(0)	
		self.dialect = dialect
		self.__read_data()

	

	# Read data using pandas. Simplify data structure for Configuration
	def __read_data(self):
		if self.dialect == "SAMoS":
			
			self.data = pandas.read_csv(self.datafile, sep=r"\s+", comment= "#", names = self.header_names)
			#
			# temp = self.data.columns
			#colshift = {}
			#for u in range(len(temp)-1): 
			#	colshift[temp[u]] = temp[u+1]
			#self.data.rename(columns = {temp[len(temp)-1]: 'garbage'},inplace=True)
			#self.data.rename(columns = colshift,inplace=True,errors="raise")
			#print(self.data.columns)
		elif self.dialect == "CCCPy":
			self.data = pandas.read_csv(self.datafile,header=0)
			# look of the header
			# currTime,xPos,yPos,xVel,yVel,polAngle,polVel,xPol,yPol,rad,glued
			# We need to muck about with the headers to distil this to a unified format
			# Classical samos header:
			#  id  type  flag  radius  x  y  z  vx  vy  vz  nx  ny  nz 
			self.data.rename(columns={"xPos": "x", "yPos": "y", "xVel": "vx", "yVel": "vy", "xPol": "nx", "yPol": "ny", "rad":"radius", "glued":"type"}, inplace=True,errors="raise")
			#print(self.data.columns)
		elif self.dialect == "CAPMD":
			self.data = pandas.read_csv(self.datafile,header=0)
		else:
			print("Unknown data format dialect!")
		

class extract:
	
    def __init__(self, filespath):
        self.filespath = filespath
	
    def extract_positions(self, tp):
        # extracts the positions from the .dat file and filter for the given  type

        files = sorted(glob.glob(self.filespath))

        positions = []

        for file in files:   
            read_file = ReadData(file, "SAMoS")
            read_data= read_file.data
            data1 = read_data[read_data['type']==tp]
            cur_positions = np.stack((data1['x'], data1['y'], data1['z'])).T
            positions.append(cur_positions)
        
        positions = np.asarray(positions)
		
        return positions


    def extract_vel(self, tp):
        # extracts the velocities from the .dat file and filter for the given type

        files = sorted(glob.glob(self.filespath))

        velocities = []

        for file in files:   
            read_file = ReadData(file, "SAMoS")
            read_data= read_file.data
            data1 = read_data[read_data['type']==tp]
            cur_velocities = np.stack((data1['vx'], data1['vy'], data1['vz'])).T
            velocities.append(cur_velocities)
        
        velocities = np.asarray(velocities)
        return velocities


    def extract_directors(self, tp):
        # extracts the directors/orientation from the .dat file and filter for the given type

        files = sorted(glob.glob(self.filespath))

        directors = []

        for file in files:   
            read_file = ReadData(file, "SAMoS")
            read_data= read_file.data
            data1 = read_data[read_data['type']==tp]
            cur_directors = np.stack((data1['nx'], data1['ny'], data1['nz'])).T
            directors.append(cur_directors)
        
        directors = np.asarray(directors)
        return directors
	
	
    def extract_particle_radii(self, tp):
        # extracts the particle radii from the .dat file and filter for the given type

        files = sorted(glob.glob(self.filespath))
        radii = []
        for file in files:   
            read_file = ReadData(file, "SAMoS")
            read_data= read_file.data
            data1 = read_data[read_data["type"] == tp]
            cur_radii = data1['radius']
            radii.append(cur_radii)
        
        radii = np.asarray(radii)
        return radii
	
