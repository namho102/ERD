from __future__ import division
import csv
from scipy import spatial, stats
import numpy as np


def load_data(file):
	with open(file, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		items = next(reader)
		items = items[1:len(items)]
		users = []
		matrix = []
		for row in reader:
			users.append(row[0])
			arr = [ int(i) for i in row[1:len(row)] ]
			matrix.append(arr)
		
		return (items, users, np.array(matrix))

def get_cosine_similarity(vector_A, vector_B):
	sim = 1 - spatial.distance.cosine(vector_A, vector_B)
	return sim

def get_pearson_similarity(vector_A, vector_B):
	a = average(vector_A)
	b = average(vector_B)
	
	# s = sum([vector_A[i] - vector_B[i] for i in range(len(vector_A))])
	# print([vector_A[i] - vector_B[i] for i in range(len(vector_A))])

	p1 = p2 = 0
	s = 0
	for i in range(len(vector_A)):
		if vector_A[i] != 0 and vector_B[i] != 0:
			t1 = vector_A[i] - a
			t2 = vector_B[i] - b
			s += t1*t2
			p1 += t1*t1
			p2 += t2*t2

	sim = s/(np.sqrt(p1)*np.sqrt(p2))	
	return sim


def get_sim_matrix(matrix, method):
	length = len(matrix)
	sim_matrix = np.zeros((length, length))

	for i in range(length):
		for j in range(i, length):
			if method == 'cosine':
				sim = get_cosine_similarity(matrix[i], matrix[j])
			else:
				sim = get_pearson_similarity(matrix[i], matrix[j])
			sim_matrix[i][j] = sim_matrix[j][i] =  sim
	
	return np.array(sim_matrix)	

def average(arr):
	s = 0
	length = 0
	for el in arr:
		s += el
		if el != 0 : length += 1 
	
	return s/length

def predict(user_index, item_index, matrix, sim_matrix, k):
	item_col = matrix[:,item_index]
	# print(col)
	users_rated_item = [i for i, x in enumerate(item_col) if x != 0 and i != user_index]

	# print(users_rated_item)
	sim_col = sim_matrix[:,user_index]
	sim_col = [x for i, x in enumerate(sim_col) if i in users_rated_item]
	nearest_neighbours = np.array(sim_col).argsort()[-k:][::-1]

	# average_rating = [{i: average(matrix[i])} for i in nearest_neighbours]
	# print(average_rating)

	s = 0
	t = 0
	for i in nearest_neighbours:
		sm = sim_matrix[user_index][i]
		# print (sm, matrix[i][item_index], average(matrix[i]))
		s += sm*(matrix[i][item_index] - average(matrix[i]))
		t += abs(sm)

	# print(s, t)	

	r = average(matrix[user_index]) + s/t
	# print average(matrix[user_index])
	print("predicted rating: %f" %(s/t))
	print r

	return (r, s/t)


if __name__ == '__main__':
	items, users, matrix = load_data('rating.csv')

	sim_matrix = get_sim_matrix(matrix, 'pearson')
	# print(sim_matrix)
	predict(3, 0, matrix, sim_matrix, 2)

