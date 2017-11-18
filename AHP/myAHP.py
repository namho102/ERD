import csv
import numpy as np
from decimal import Decimal as D

RI = (0, 0, D('0.58'), D('0.9'), D('1.12'),
      D('1.24'), D('1.32'), D('1.41'), D('1.45'), D('1.49'),
      D('1.51'), D('1.48'), D('1.56'), D('1.57'), D('1.59')
)

def load_criteria(file):
	with open(file, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		headers = next(reader)
		matrix = []
		for row in reader:
			arr = [ float(i) for i in row ]
			matrix.append(arr)
		
		return (headers, matrix)

def load_alternatives(file):
	with open(file, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		return next(reader)

def load_alt_comparsion_matrix(file):
	with open(file, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		matrix = []
		for row in reader:
			arr = [ float(i) for i in row ]
			matrix.append(arr)

		return matrix	

def get_weights(matrix):
   evas, eves = np.linalg.eig(matrix)

   eva = max(evas)
   eva_idx = evas.tolist().index(eva)
   eve = eves.take((eva_idx,), axis=1)

   normalized = eve / sum(eve)

   vector = [abs(e[0]) for e in normalized]

   return vector

def get_consistency(matrix):
	eva = max(np.linalg.eig(matrix)[0]).real
	n = len(matrix)
	CI = (eva-n) / (n-1)
   
	CR = CI / float(RI[n])
	return CR

if __name__ == '__main__':
	criteria, criteria_matrix = load_criteria('criteria.csv')
	alternatives = load_alternatives('alternatives.csv')
	ranking_matrix = []
	for criteria_name in criteria:
		alt_comparsion_matrix = load_alt_comparsion_matrix(criteria_name + '.csv')
		ranking = get_weights(alt_comparsion_matrix)
		ranking_matrix.append(ranking)

	priority_vector = get_weights(criteria_matrix)
	priority_vector = np.array(priority_vector)
	ranking_matrix = np.array(ranking_matrix).transpose()

	product_matrix = np.multiply(ranking_matrix, priority_vector)
	sum_matrix = np.sum(product_matrix, axis=1)

	for i in range(len(alternatives)):
		print(alternatives[i], sum_matrix[i])
	

