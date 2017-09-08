import json
from pprint import pprint
epoch_length = 0.5

#function.py is a bad name for a file. function is a reserved word, never use reserved words as var names, class names, or file names

def minutes_in_stage(epochstage):
	#read in the epochstage and check the size
	# This should be done in a loop. Whenever you are iterating something in words, you might as well use a loop
	zero = epochstage.count(0) * epoch_length 
	first = epochstage.count(1) * epoch_length 
	second = epochstage.count(2) * epoch_length
	third =  epochstage.count(3) * epoch_length
	minuteinstage = {'0': zero, '1': first,'2': second, '3':third} #PEP8 standard variables should_be_like_this and notlikethis
	return minuteinstage #You could just return the dictionary dirrectly here and save on a variable def

#this function could calculate the proportion of the data
def proportion_in_stage(epochstage):
#These function repeat each other, remeber the golden rule of programing, "dont repeat yourself". 
# so, proportion_in_stage could just call minutes_in_stage. Feel free to add total_sleep_time as one of the dictionary keys to make this easier

	zero = epochstage.count(0)
	first = epochstage.count(1) 
	second = epochstage.count(2)
	third =  epochstage.count(3)
	total = zero + first + second + third 
	proportioninstage  = {'0': zero/(100*total), '1': first/(100*total),'2': second/(100*total), '3':third/(100*total)} #Why are you dividing by 100. Its already a proportion (range = [0,1])
	return proportioninstage


#json_file='a.json' 
json_file='CAPStudy_subjectid216_visit1.json'

json_data=open(json_file)
data = json.load(json_data)
#pprint(data)
json_data.close()

#print ( "epochstage: ", data['epochstage'][1])
minutes_in_stage(data['epochstage'])
proportion_in_stage(data['epochstage'])
