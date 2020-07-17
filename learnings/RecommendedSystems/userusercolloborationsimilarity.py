from surprise import KNNWithMeans
from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import train_test_split

data =Dataset.load_builtin('ml-100k')
print(data)
# trainset ,testset =train_test_split(data,test_size=0.15)
# # user_based=false 
# algo=KNNWithMeans(k=50,sim_options={'name':'pearson_baseline','user_based':True}) 
# algo.fit(trainset)

# uid=str(196)
# iid=str(302)

# pred=algo.predict(uid,iid,verbose=True)

# # print(pred)

# pred_test=algo.test(testset)

# # print(pred_test)

# print('user based model:Test set')
# accuracy.rmse(pred_test,verbose=True)
# print(accuracy)
