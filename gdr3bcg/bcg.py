import numpy as np
import pandas as pd
import os
import sys
import pdb
import math
from pkg_resources import resource_filename

"""

"""
class BolometryTable:
		
	'''
	'''
	def main(self):
		if len(sys.argv) < 5 or len(sys.argv) > 6:
			print('bad number of arguments : should be teff,logg,metal,alpha (and optionally offset)')
			exit()
		
		value=sys.argv[1:5]
		param = np.zeros(len(value))
		
		for i in range(len(param)):
			param[i]=float(value[i])
		
		if len(sys.argv)==6:
			offset=float(sys.argv[5])
			print(self.computeBc(param,offset))
		else:
			print(self.computeBc(param))
		exit()
    	
	"""
	   constructor. Take as argument the ascii table file.
	"""
	def __init__(self,file=None):
		
		if file is None:
			data_dir = resource_filename('gdr3bcg','data')
			file=os.path.join(data_dir, 'bc_dr3_feh_all.dat')

		table=pd.read_table(file, delim_whitespace=True, names=["teff", "logg","metal","alpha","bc"],dtype={'teff':np.float64,'logg':np.float64,'metal':np.float64,'alpha':np.float64,'bc':np.float64})
		self.__bolometry=table.sort_values(by=['teff', 'logg','metal','alpha'])
		self.__param=self.__bolometry.values[:,0:4]
		self.__bc=self.__bolometry.values[:,4]
		self.__paramkeys=[]
	    
		for i in range(4):
			self.__paramkeys.append(np.unique(self.__param[:,i]))

	
	"""
	   main method: uses 'value' as input : [teff, logg, metal, alphaFe] and an optional offset
	   returns the bolometric correction at this point
           the value is shifted by '0.2', which is an offset applied for DR3
           This implies that the absolute magnitude if the sun in G band is equals 4.66 mag
           see : creevey et al, 2022, sect 4,3
	"""
	def computeBc(self,value,offset=0):

		try:
			bc=self.interpolate(value)
		except ValueError:
			index = self.nearestIndex(value)
			bc=self.__bc[self.nearestIndex(value)]
		return bc+0.2+offset
				
	"""
	   find the index location of pouint g (array) by dichotomy
	"""
	def where(self,g):
		bas=0
		haut=len(self.__param)-1
		result=-1
		
		while True:
			milieu =(int)((bas+haut)//2)
			elem = self.__param[milieu]

			if self.equals(elem,g):
				result=milieu
			elif self.compareTo(elem,g)==-1:
				bas = milieu + 1
			else:
				haut = milieu -1

			if self.equals(g,elem) or (bas > haut):
				break
		return result

	"""
		checks equality of 2 points
	"""
	def equals(self,p1,p2):
		tol=1e-5
		
		if abs(round(p1[0]-p2[0],6))<tol and abs(round(p1[1]-p2[1],6))<tol and abs(round(p1[2]-p2[2],6))<tol and abs(round(p1[3]-p2[3],6))<tol:
			return True
		return False
		
	"""
		define comparaison between 2 points (-1 if before, 1 after)
	"""
	def compareTo(self,p1,p2):
		tol=1e-4
		
		if p1[0] > p2[0]:
			return 1
			
		if (p1[0] < p2[0]):
			return -1
		
		if abs(round(p1[0] -p2[0],5)) < tol:
		
			if p1[1] > p2[1]:
				return 1				
			if p1[1] < p2[1]:
				return -1				
			if abs(round(p1[1] -p2[1],5)) < tol:			
			
				if p1[2] > p2[2]:
					return 1				
				if p1[2] < p2[2]:
					return -1			
						
				if abs(round(p1[2] -p2[2],5)) < tol:
				
					if p1[3] > p2[3]:
						return 1		
					if p1[3] < p2[3]:
						return -1
		return 0
		
	"""
       find the nearest (predecessor) point
    """
	def nearestIndex(self,value):
		delta=[100,1,1,1]
		nearest=np.zeros(len(value))
		params_lower=np.zeros(len(value))
		params_upper=np.zeros(len(value))
		
		for i in range(len(value)):
			val_g=round(value[i],4)
			
			if val_g >= max(self.__bolometry.values[:,i]):
				nearest[i]=max(self.__bolometry.values[:,i])
			elif val_g <= min(self.__bolometry.values[:,i]):
				nearest[i]=max(self.__bolometry.values[:,i])
			else:
			
				for j in range(len(self.__paramkeys[i])):
					u = self.__paramkeys[i][j]
					
					if abs(u-value[i]) < 1e-5:
						nearest[i]=u
						params_lower[i]=u
						params_upper[i]=u
						break
					elif u>value[i]:
						params_lower[i]=self.__paramkeys[i][j-1]
						params_upper[i]=u
						
						if abs(self.__paramkeys[i][j-1]-value[i]) > abs(u-value[i]):
							nearest[i]=u
						else:
							nearest[i]=self.__paramkeys[i][j-1]
						break
		
		index= self.where(nearest)
		
		if index!=-1:
			return index
		else:
			id_min=self.where(params_lower)
			
			if id_min==-1:
			
				for axis in range(len(params_lower)):
					id_min=self.search(params_lower,axis,-1)
					
					if id_min!=-1:
						break
			
			if id_min==-1:
				id_min=0
			id_max=self.where(params_upper)
			
			if id_max==-1:
			
				for axis in range(len(params_upper)):
					id_max=self.search(params_upper,axis,1)
					
					if id_max!=-1:
						break
						
			if id_max==-1:
				id_max=len(self.__bolometry)-1
			d_max=math.inf
			
			for l in range(id_min, id_max+1,1):
				distance=0
				
				for k in range(len(value)):
					x = self.__param[l][k]
					y = value[k]
					u = round(x-y,14)	
					distance = distance+(u*u)/(delta[k]*delta[k])
					
				if distance < d_max:
					d_max=distance
					index=l
			
			return index
		
	"""
	   performs a quick interpolation into the grid
	"""
	def interpolate(self,value):
		
		pos = self.where(value)
		x=value.copy()
		nParams=len(value)
		
		if pos!=-1:
			return self.__bc[pos]
			
		index = self.nearestIndex(value)
		
		nearGridNode = self.__param[index].copy()
		
		for i in range(nParams):
			ww=round(value[i]*1e7)/1e7
			xx=round(nearGridNode[i]*1e7)/1e7
			
			if ww < xx:
				prevIndex=self.previousNode(nearGridNode,i)
			
				if prevIndex==-1:
					raise ValueError()
				nearGridNode[i]=self.__param[prevIndex][i]
		
		nElems = 2**nParams
		coeff=np.zeros((nElems,nParams), dtype = int)
		v={}	
		hypercubeNodes=[]
		hypercubeBc=[]
		
		for i in range(nElems):
			val=i
			nodeCopy=nearGridNode.copy()
			
			try:
				for j in range(nParams):
					
					if val%2==1:
						coeff[i][j]=1
						
						if not j in v:
							nextNode=self.nextNode(nearGridNode,j)
				
							if nextNode==-1:
								nextNode=self.previousNode(nearGridNode,j)
								
								if nextNode==-1:
									raise ValueError()
									
							v[j]=nextNode
						nodeCopy[j]=self.__param[v[j]][j]
						
					val=val//2	
				
				index=self.where(nodeCopy)
					
				if index==-1:
					raise ValueError()
					
				hypercubeBc.append(self.__bc[index])
				hypercubeNodes.append(nodeCopy)
				
			except ValueError:
				index=self.where(nearGridNode)
				
				if index==-1:
					raise ValueError()
				hypercubeNodes.append(nearGridNode)
				hypercubeBc.append(self.__bc[index])
			
		interpolated=0
		x=np.zeros(nParams, dtype = float)
		
		for i in range(nParams):
				
			if hypercubeNodes[2**i][i]==nearGridNode[i]:
				x[i]=0
			else:
				x[i]=(value[i]-nearGridNode[i]) / (hypercubeNodes[2**i][i]-nearGridNode[i])
		
		for i in range(coeff.shape[0]):
			c=1
			
			for j in range(coeff.shape[1]):
				c=c*((1-x[j]) + coeff[i][j]* (2*x[j]-1))
				
			interpolated=interpolated+hypercubeBc[i]*c
		return interpolated


	"""
	   find the kth previous or kth next value in one given dimension
	"""
	def find_value(self,value,axis,k):
		count=0
		
		for i in self.__paramkeys[axis]:
			
			if abs(value-i) < 1e-7:
				if count+k < 0:
					return None
				try :
					return self.__paramkeys[axis][count+k]
				except IndexError:
					return None
			count=count+1
		return None
		
	"""
  	   check bounds : False if (previous or next) point from 'val' is outside grid
  	"""
	def check(self,val,axis,direction):
	
		if direction>0 and val>=self.__paramkeys[axis].max():
			return False
			
		if direction<0 and val <=self.__paramkeys[axis].min():
			return False
			
		return True

	def search(self, val, axis, direction):
		k=0
		index=-1

		while True:
			x=val.copy()
			
			if direction<0:
				k=k-1
			else:
				k=k+1
				
			paramValSearch=self.find_value(val[axis],axis,k)
			
			if paramValSearch is None:
				return -1
				
			x[axis]=paramValSearch

			index=self.where(x)
		
			if index !=-1 or not self.check(paramValSearch,axis,k):
				return index

	def previousNode(self,val,axis):
		return self.search(val,axis,-1)

	def nextNode(self,val,axis):
		return self.search(val,axis,1)
		
if __name__ == "__main__":
	bcg = BolometryTable()
	bcg.main()
			
			


		
