from firebase import firebase
# https://i2max-project.firebaseio.com/
firebaseDatabase = firebase.FirebaseApplication('https://i2max-project.firebaseio.com/', None)
firebaseDatabaseResult = firebaseDatabase.get('/1', None)
print (firebaseDatabaseResult)