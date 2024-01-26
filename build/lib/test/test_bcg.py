import gdr3bcg.bcg as bcg
import unittest
import pandas as pd
import numpy as np
import os
from pkg_resources import resource_filename


class TestBcg(unittest.TestCase):

	def setUp(self):
		self.__data_test = resource_filename('gdr3bcg','data/test')
		file=os.path.join(self.__data_test, 'ascii_bcg.dat')
		self.__bc=bcg.BolometryTable(file)
		self.__testVal=[2600.0,5.0,-0.50,0.2]
		
	def test_where(self):
		print('test where method')
		expected = 358
		result = self.__bc.where(self.__testVal)
		self.assertEqual(expected,result)
		
		
	def test_previous(self):
		print('test previous node search')
		expectedIndexes = [171,342,-1,357]
		
		for i in range(len(self.__testVal)):
			self.assertEqual(expectedIndexes[i],self.__bc.previousNode(self.__testVal,i))
	
	def test_next(self):
		print('test next node search')
		expectedIndexes = [746,-1,-1,-1]
		
		for i in range(len(self.__testVal)):
			self.assertEqual(expectedIndexes[i],self.__bc.nextNode(self.__testVal,i))
			
	def test_interpolate(self):
		print('test interpolate 1')
		result = self.__bc.interpolate([2550.,5.0 ,-0.5  ,0.2])
		expected = -2.1935
		self.assertAlmostEqual(expected, result, places=4)
		
	def test_interpolate2(self):
		print('test interpolate 2')
		result = self.__bc.interpolate([4200,1,1,0.1])
		expected = -0.5282
		self.assertAlmostEqual(expected, result, places=4)
		
	def test_computeBc(self):
		print('test computeBc')
		offset = 0.12
		test=[2550.,5.0 ,-0.5  ,0.2]
		expected = -1.8735
		result = self.__bc.computeBc(test,offset)
		self.assertAlmostEqual(expected, result, places=4)
		
	def test_nearestIndex(self):
		print('test nearest index search')
		test=[2871.180,   -0.095,   -4.595,    0.0277664]
		expected=581
		result=self.__bc.nearestIndex(test)
		self.assertEqual(expected,result)

	def test_random(self):
		print('test java vs python on random parameters')
		file=os.path.join(self.__data_test, 'bcgran.dat')
		table=pd.read_table(file, delim_whitespace=True, names=["teff", "logg","metal","alpha","bc"],dtype={'teff':np.float64,'logg':np.float64,'metal':np.float64,'alpha':np.float64,'bc':np.float64})
		params=table.values[:,0:4]
		expected=table.values[:,4]
		
		for i in range(params.shape[0]):
			result=self.__bc.computeBc(params[i])
			self.assertAlmostEqual(expected[i],result,places=3)
			
		
		
	if __name__ == '__main__':
		unittest.main()
