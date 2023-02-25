
from sklearn.preprocessing import LabelEncoder

# Convert the string features into numerical features using sklearn LabelEncoder.
train_features = ["Apple", "Orange", "Potato", "Tomato", "Grape", "Apple", "Tomato", "Grape"]
test_features = ["Apple", "Orange", "Tomato"]
le = LabelEncoder()

# START EDIT
transformed_train = []
transformed_test = []
# END EDIT

print(train_features)
print(test_features)
print(transformed_train)
print(transformed_test)