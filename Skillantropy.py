#process.py

import pandas as pd
import numpy as np
import nltk
from os import path
from time import time


def parse_path_data(path_data):
# split by first "/"
path_word = path_data[1].split("/",1)
if path_word[0] == "search":
# if the category is search, return key words
separated = path_word[1].split("/", 1)
if len(separated) > 1:
keywords=separated[1]
return [path_data[0], 'search', keywords, path_data[2]]
elif path_word[0] == "node":
if len(path_word) > 1:
# if the category is node, return the number
return [path_data[0], 'node', path_word[1], path_data[2]]


data_dir = "\\2016-Customer\\Skillanthropy\\data"

#
# access_log1 = pd.read_csv(path.join(data_dir, "access_logs1.csv"), sep=",", usecols=[0,1,3])
#
# # test
# path_word = access_log1.head(1).values.tolist()[0]


def process_raw_file(data_dir, in_file, out_search_file, out_node_file):

out_search = open(path.join(data_dir, out_search_file), 'w')
out_node = open(path.join(data_dir, out_node_file), 'w')

with open(path.join(data_dir, in_file)) as f:
f.next()
for line in f:
#ignore the first line
processed = line.strip("\n").split(",")
processed = [processed[i] for i in [0,1,3]]
parsed_data = parse_path_data(processed)
if parsed_data:
if parsed_data[1] == 'search':
parsed_data = [parsed_data[i] for i in [0,2,3]]
out_search.write(",".join(parsed_data) + "\n")
elif parsed_data[1] == 'node':
parsed_data = [parsed_data[i] for i in [0, 2, 3]]
out_node.write(",".join(parsed_data) + "\n")

out_node.close()
out_search.close()

process_raw_file(data_dir, "access_logs2.csv", "out_search2.txt", "out_node2.txt")

#PART 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

#data folder
dir = '//Desktop/Skillanthropy/readworks_data/'
# read passage data
passage = pd.read_csv(dir+'passage_report.csv', sep=",")
passage['Grade'].head(5)

"""
# read access 1 data
access1 = pd.read_csv(dir+'access_logs1.csv', sep=",")
access1['path'].head(5)

# read access 2 data
access2 = pd.read_csv(dir+'access_logs2.csv', sep=",")
access2['path'].head(5)
"""

#%%

#data folder
dir = '//Desktop/Skillanthropy/node_query_data/'
# read click data1
node1 = pd.read_csv(dir+'out_node1.txt', sep=",")
node1.columns = ['uid','NID','timestamp']
# read click data2
node2 = pd.read_csv(dir+'out_node2.txt', sep=",")
node2.columns = ['uid','NID','timestamp']

node = node1.append(node2)

# drop duplicates - 34M rows
node_dis = node.drop_duplicates()


#%%
# make the types consistent
passage['NID'] = passage['NID'].apply(lambda x: str(x))
# join with the passage report to get passage related infomation and remove video nodes
node_ful = node_dis.merge(passage, on='NID',how='inner')

# sort the clicked passages
node_ful2 = node_ful.join(node_ful.groupby('NID')['NID'].count(), on='NID', rsuffix='_ct')
top100 = node_ful2[['NID','NID_ct']].drop_duplicates().sort(['NID_ct'], ascending=[0]).head(100)

# join with passage report
top100_ful = top100.merge(passage, on='NID',how='inner')

""" 
test = node_dis;
test2 = test.join(test.groupby('NID')['NID'].count(), on='NID', rsuffix='_ct')
test2_10 = test2[['NID','NID_ct']].drop_duplicates().sort(['NID_ct'], ascending=[0]).head(10)
test2_10_ful = test2_10.merge(passage, on='NID',how='left')


Output:
NID NID_ct Title \
0 2059 178678 NaN 
1 7889 129540 A Baby Polar Bear Grows Up 
2 7373 101270 A Bat Mystery 
3 9242 93768 A Bigger Pond 
4 8836 83369 A Camping Trip 
5 8980 81208 A Bird Came Down the Walk 
6 7810 79118 "Seven Minutes of Terror," Eight Years of Inge... 
7 7864 74773 NaN 
8 2387 71998 NaN 
9 7877 67905 A Bad Robot 

The most popular one is node 2059, but readworks.org/node/2059 is 
http://www.readworks.org/lessons/thank-you-for-joining
which is the introduction to Readworks Curriculum

node 7864: http://www.readworks.org/books/about-question-sets
node 2387: http://www.readworks.org/lessons/aligned-standards-and-textbooks
"""

# plot histogram
names = top100_ful['Title'].head(10)
data = np.transpose(top100_ful['NID_ct'].head(10))

# Make an example plot with two subplots...
fig = plt.figure()
ax = plt.subplot(111)
width = 0.3
bins = map(lambda x: x-width/2,range(1,len(data)+1))
ax.bar(bins,data)
ax.set_ylim([40000,140000])
ax.set_xticks(map(lambda x: x, range(1,len(data)+1)))
ax.set_xticklabels(names,rotation=70, rotation_mode="anchor", ha="right")

dir = '//Desktop/Skillanthropy/'
#fig.savefig(dir+'top10Passages.jpg')

#%%
# top passages changing over time
# join with passage report
top10_time = top100_ful.head(12).merge(node_dis, on='NID',how='inner')
top10_time['month'] = top10_time['timestamp'].str[0:7]
top10_time2 = top10_time[['NID','month']]
top10_time3 = top10_time2.merge(passage, on='NID',how='inner')[['NID','month','Title']]


# top 1 passage A Baby Polar Bear Grows Up
nid = list(top10_time['NID'].drop_duplicates())
for x in nid:
test = top10_time3[top10_time3['NID']== x]
test2 = test.join(test.groupby(['month']).count(), on='month', rsuffix='_ct').drop_duplicates()

# plot the trend over time
plt.cla()
tle = list(test['Title'])[1]
plt.plot(np.linspace(1,12,12), test2['NID_ct'], linewidth=2)
plt.suptitle(tle, fontsize=15)
plt.gca().set_xticks(np.linspace(1,12,12))
plt.gca().set_xticklabels(test2['month'],rotation=40, rotation_mode="anchor", ha="right")

dir = '//Desktop/Skillanthropy/'
plt.savefig(dir+tle+'_month.png',format='png', dpi=900)

#PART 3
matplotlib.pyplot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# read the search data

dir = '//Skillantrophy/'
passage = pd.read_csv(dir+'passage_report.csv', sep=",")
log1 = pd.read_csv(dir+'out_search1.txt', sep=",")
log1.columns = ['uid','query','timestamp']
node1 = pd.read_csv(dir+'out_search2.txt', sep=",")
node1.columns = ['uid','query','timestamp']

# appending two search data
node = log1.append(node1)

# drop duplicate data
log1_dis = node.drop_duplicates()

# count the nodes per passage , calulate top 100
log1_dis2 = log1_dis.join(log1_dis.groupby('query')['query'].count(), on='query', rsuffix='_ct')
log1_dis3=log1_dis2[['query','query_ct']].drop_duplicates().sort(['query_ct'], ascending=[0]).head(100)
log1_dis4 = log1_dis3.reset_index()
log1_dis4.to_csv(dir+'topquery1.txt')

# consider all for timestamp
log1_dis3_all=log1_dis2[['query','query_ct']].drop_duplicates().sort(['query_ct'], ascending=[0])
top10_time = log1_dis3.head(10).merge(log1_dis, on='query',how='inner')
top10_time['month'] = top10_time['timestamp'].str[0:7]
top10_time2 = top10_time[['query','month']]
query = list(top10_time['query'].drop_duplicates())
for x in query:
test = top10_time2[top10_time2['query']== x]
test2 = test.join(test.groupby(['month']).count(), on='month', rsuffix='_ct').drop_duplicates()
# graphs for top queries over time
plt.cla()
tle = list(test['query'])[1]
plt.plot(np.linspace(1,12,12), test2['query_ct'], linewidth=2)
plt.suptitle(tle, fontsize=15)
plt.gca().set_xticks(np.linspace(1,12,12))
plt.gca().set_xticklabels(test2['month'],rotation=40, rotation_mode="anchor", ha="right")
plt.savefig(dir+tle+'_month.png',format='png', dpi=900)


#Passage data Analysis from passage_report.csv

#dataframe for passage
dp=pd.DataFrame(data=passage)
grd=dp.groupby('Grade').NID.nunique()
grd_df=pd.DataFrame(data=grd)
grd_df1 = grd_df.reset_index()
#grd_df1.loc[grd_df1['Grade']=='K','Grade']=0

lex=dp.groupby('Lexile').NID.nunique()
lex = lex.reset_index()
lex.to_csv(dir+'lex.csv')

fic_df=dp.groupby('Fiction / Nonfiction').NID.nunique()
fic_df1 = fic_df.reset_index()

concp_df=dp.groupby('Concepts of Comprehension').NID.nunique()
concp_df1 = concp_df.reset_index()

domain_df=dp.groupby('Domains').NID.nunique()
domain_df1= domain_df.reset_index()

title_ct=dp.groupby('Title').NID.nunique()
dp.loc[dp['Title'] == 'Frogs at Risk']

title_ct2 = title_ct.reset_index()
title_ct2.to_csv(dir+'test2.csv')

#Plots for passage report analysis

#GRADE DATA

ax = axes([0.1, 0.1, 0.8, 0.8])
# The slices will be ordered and plotted counter-clockwise.GRADE
labels = grd_df1['Grade']
fracs = grd_df1['NID']
explode=(0, 0.05, 0, 0)
cs=plt.Set1(np.arange(11)/11.)
pie(fracs, labels=labels , autopct='%1.1f%%', shadow=True, startangle=90,
colors=cs)
# The default startangle is 0, which would start
# the Frogs slice on the x-axis. With startangle=90,
# everything is rotated counter-clockwise by 90 degrees,
# so the plotting starts on the positive y-axis.
title('Grade VS NId', bbox={'facecolor':'0.8', 'pad':5})
show()
savefig(dir+'grade.jpg')

################################################################################################

#FICTION/NONFICTION DATA

ax = axes([0.1, 0.1, 0.8, 0.8])
# The slices will be ordered and plotted counter-clockwise.GRADE
labels = fic_df1['Fiction / Nonfiction']
fracs = fic_df1['NID']
explode=(0, 0.05, 0, 0)
cs=plt.Set1(np.arange(11)/11.)
pie(fracs, labels=labels , autopct='%1.1f%%', shadow=True, startangle=90,
colors=cs )
# The default startangle is 0, which would start
# the Frogs slice on the x-axis. With startangle=90,
# everything is rotated counter-clockwise by 90 degrees,
# so the plotting starts on the positive y-axis.
title('Fiction / Nonfiction VS NId', bbox={'facecolor':'0.8', 'pad':5})
show()
savefig(dir+'Fiction.jpg')

################################################################################################

#graph to show frequency of top 10 queries

log1_top10=log1_dis2[['query','query_ct']].drop_duplicates().sort(['query_ct'], ascending=[0]).head(10)
fig = plt.figure()
ax = plt.subplot(111)
width = 0.3
bins = map(lambda x: x-width/2,range(1,len(log1_top10['query_ct'])+1))
ax.bar(bins,log1_top10['query_ct'])
#ax.set_ylim([40000,140000])
ax.set_xticks(map(lambda x: x, range(1,len(log1_top10['query_ct'])+1)))
ax.set_xticklabels(log1_top10['query'],rotation=70, rotation_mode="anchor", ha="right")
fig.savefig(dir+'top10queriess.jpg')

####################################################################################


matplotlib.pyplot
weights = lr.coef_[0]


# Make figure and plot weights as a Stem plot. 
fig2, ax2 = plt.subplots(figsize=(12,6))
stemplot = plt.stem(weights, linefmt='b--', markerfmt='bo', basefmt='r-')

# Set axis labels
ax2.set_xlabel("Feature names", fontsize=20, style='italic', fontweight='bold')
ax2.set_ylabel("Weight", fontsize=20, style='italic', fontweight='bold')
ax2.set_xticklabels(model_df.columns[feature_idx], rotation=70)
ax2.set_title("Feature weights for model", fontsize=25, fontweight='bold')

# Again, make a few aesthetic adjustments to x-axis to make all data point clearly viewable
ax2.set_xlim(-0.5, len(weights)-0.5)
ax2.tick_params(axis='both', which='major', labelsize=20, length=10)
ticks = plt.xticks(np.arange(0,len(weights),1.0))