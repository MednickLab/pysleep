import json
from pprint import pprint
epoch_length = 0.5


def minutes_in_stage(epochstage):
	#read in the epochstage and check the size
	
	total = 0
	a = []
	for x in range(0, 4):
		temp = epochstage.count(x)*epoch_length
		a.append(temp)
		total += temp
	
	return  {'0': a[0], '1': a[1],'2': a[2], '3':a[3], 'total_sleep_time':total} 

#this function could calculate the proportion of the data
def proportion_in_stage(epochstage):
	temp = minutes_in_stage(epochstage)
	total = temp['total_sleep_time']
	return {'0': temp['0']/total, '1': temp['1']/total,'2': temp['2']/total, '3':temp['3']/total} 

def sleep_efficiency(epochstage):
	temp = minutes_in_stage(epochstage)
	return (temp['1']+temp['2']+temp['3'])/temp['total_sleep_time']
#json_file='a.json' 
json_file='CAPStudy_subjectid216_visit1.json'

json_data=open(json_file)
data = json.load(json_data)
#pprint(data)
json_data.close()

#print ( "epochstage: ", data['epochstage'][1])
minutes_in_stage(data['epochstage'])
proportion_in_stage(data['epochstage'])
