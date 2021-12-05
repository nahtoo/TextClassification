import sys
import os
import re

from unigram import Unigram_Baseline
from bigram import Bigram_Baseline

lineNumberMatch = re.compile(r"Lines:\s(\d+)")
headerLineMatch = re.compile(r".+:\s\S+")

def extract_data(path_to_folder):
	classes = []
	data = []
	class_num = 0
	for item in os.listdir(path_to_folder):
		item_path = os.path.join(path_to_folder,item)
		# print(item)
		# print(item_path)
		if os.path.isdir(item_path):
			for subitem in os.listdir(item_path):
				subitem_path = os.path.join(item_path, subitem)
				if os.path.isfile(subitem_path):
					# print(subitem)
					# print(subitem_path)
					extract_segments_from_file(subitem_path,data,classes,class_num)
			class_num += 1
	return (classes,data)

def extract_segments_from_file(file,data,classes,class_num):
	max_lines = -1
	num_lines = 0
	body = False
	with open(file,'r') as fread:
		while True:
			if max_lines != -1 and num_lines == max_lines:
				break
			try:
				line = fread.readline()
			except UnicodeDecodeError:
				continue
			if not line:
				break
			if not body:
				match = lineNumberMatch.match(line)
				if match:
					max_lines = int(match.group(1))
				else:
					match = headerLineMatch.match(line)
					# print(match)
					if match is None and max_lines != -1:
						body = True
			if body:
				if not line.isspace():
					data.append(line)
					classes.append(class_num)
			num_lines += 1
	return data



if __name__ == '__main__':
	if len(sys.argv) != 5:
		# trainingclasses,trainingdata = extract_data("Selected 20NewsGroup/Training")
		# testclasses,testdata = extract_data("Selected 20NewsGroup/Evaluation")
		# print("Unigram")
		# UBtest = Unigram_Baseline(trainingdata,trainingclasses,testdata,testclasses)
		# UBtest.display_LC()
		# print("Bigram")
		# BItest = Bigram_Baseline(trainingdata,trainingclasses,testdata,testclasses)
		sys.exit("python UBBB.py <trainset> <evalset> <output> <displayLC>")
	else:
		trainingclasses,trainingdata = extract_data(sys.argv[1])
		testclasses,testdata = extract_data(sys.argv[2])
		UB = Unigram_Baseline(trainingdata,trainingclasses,testdata,testclasses,sys.argv[3])
		BB = Bigram_Baseline(trainingdata,trainingclasses,testdata,testclasses,sys.argv[3])
		if int(sys.argv[4]) == 1:
			UB.display_LC()
		
