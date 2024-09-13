counts = {genre: 0 for genre in genres}
arr=[genres_dict[i] for i in ([t[1] for t in data if t[0]==270894 and t[2]==5.0])]
for m in arr:
	for gen in m:
		counts[gen]+=1

#-----------------------------------------------------
res=(load_model()[0][270894].T@load_model()[2]).T+load_model()[1][270894]+load_model()[3]
res=res.flatten()
myndict = {index: res[index] for index in movies_dict.keys()}
sorted_dict_desc = dict(sorted(myndict.items(), key=lambda item: item[1], reverse=True))
#print(list(sorted_dict_desc.keys())[:50])

#getting list of films rated by the user
ratedmovies=[]
for tup in data:
	if(tup[0]==270894):
		ratedmovies.append(tup[1])

#getting 50 most potentially liked films by the user
top50movies=[]
for i in list(sorted_dict_desc.keys()):
	if(len(top50movies)==50):
		break
	if(i not in ratedmovies):
		top50movies.append(i)
#counting of these films tagged with 'Animation'
m=0
for i in top50movies:
	if('Animation' in genres_dict[i]):
		m+=1
print(m)

#----------------------------------------------------
def similarity(model,i,j):
	return (np.dot(model[2][i].T,model[2][j])/(np.linalg.norm(model[2][i])*np.linalg.norm(model[2][j])))[0][0]
sim_dict={j: similarity(model,260,j) for j in movies_dict.keys()}
sorted_sim_dict = dict(sorted(sim_dict.items(), key=lambda item: item[1], reverse=True))
print(list(sorted_sim_dict.keys())[:11])
#---------------------------------------------------
for i in movies_dict.keys():
    for j in movies_dict.keys():
        if(i!=j):
            avg+=similarity(model,i,j)
avg=avg/(len(list(movies_dict.keys()))**2)
print(avg)
#---------------------------------------------------
result=[]
for genre in genres:
	summ=0
	counter=0
	for mov1 in movies_dict.keys():
		for mov2 in movies_dict.keys():
			if(genre in genres_dict[mov1] and genre in genres_dict[mov2] and mov1!=mov2):
				summ+=similarity(model,mov1,mov2)
				counter+=1
	result.append((genre,summ/counter))
print(result)
#---------------------------------------------------
result=[]
for genre2 in genres:
    if('Comedy'!=genre2):
        summ=0
        counter=0
        for mov1 in movies_dict.keys():
        	for mov2 in movies_dict.keys():
        		if('Comedy' in genres_dict[mov1] and genre2 in genres_dict[mov2] and mov1!=mov2):
        		  summ+=similarity(model,mov1,mov2)
        		  counter+=1
        result.append((genre2,summ/counter))
print(result)