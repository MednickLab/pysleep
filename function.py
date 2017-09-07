import json
from pprint import pprint
epoch_length = 0.5

def minutes_in_stage(epochstage):
	#read in the epochstage and check the size
	zero = epochstage.count(0) * epoch_length 
	first = epochstage.count(1) * epoch_length 
	second = epochstage.count(2) * epoch_length
	third =  epochstage.count(3) * epoch_length
	minuteinstage = {'0': zero, '1': first,'2': second, '3':third}
	return minuteinstage

#this function could calculate the proportion of the data
def proportion_in_stage(epochstage):

	zero = epochstage.count(0)
	first = epochstage.count(1) 
	second = epochstage.count(2)
	third =  epochstage.count(3)
	total = zero + first + second + third 
	proportioninstage  = {'0': zero/(100*total), '1': first/(100*total),'2': second/(100*total), '3':third/(100*total)}
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