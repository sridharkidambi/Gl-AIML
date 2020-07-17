from collections import defaultdict
from surprise import SVD
from surprise import Dataset;
data =Dataset.load_builtin('ml-100k')
trainset=data.build_full_trainset()
# print(trainset.ur)

algo=SVD()
algo.fit(trainset)

testset=trainset.build_anti_testset()
# print(testset)
predictions =algo.test(testset)
# print(predictions)

def get_top10_fun(predictions,n=1):
    top_n=defaultdict(list)
    for uid,iid,true_r,est,_ in predictions:
        top_n[uid].append((iid,est))

    for uid,user_ratings in top_n.items():
        user_ratings.sort(key=lambda x:x[1],reverse=True)
        top_n[uid]=user_ratings[:n]
    return top_n;

top_n=get_top10_fun(predictions,1)
print(top_n)