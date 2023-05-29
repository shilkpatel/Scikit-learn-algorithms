from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

dataset = pandas.read_csv("archive/amsterdam_weekdays.csv")

y=dataset.realSum
X=dataset[['room_shared','room_private','person_capacity',
           'host_is_superhost','biz','cleanliness_rating','guest_satisfaction_overall',
           'bedrooms','dist','metro_dist','attr_index','attr_index_norm','rest_index','rest_index_norm'
           ]]

clean_set = {
    'room_type':{'Private room':0,'Entire home/apt':1}

}

dataset=dataset.replace(clean_set)


train_x, validation_x, train_y, validation_y=train_test_split(X,y,random_state=1)
#26 max nodes seem optimal with mse of 181.595
for i in [i for i in range(20,30)]:
    prediction_model= RandomForestRegressor(random_state=0,max_leaf_nodes=i)
    prediction_model.fit(train_x,train_y)

    validation_prediction=prediction_model.predict(validation_x)
    validation_mse = mean_absolute_error(validation_prediction,validation_y)
    print("Max nodes are ",i,"Error:",validation_mse)


