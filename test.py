# listx = ([('gender', 'Female'), ('age', '20'), ('hypertension', '1'), ('heart_disease', '1'), ('ever_married', 'Yes'), ('work_type', 'Never_worked'),
#          ('Residence_type', 'Urban'), ('avg_glucose_level', '80'), ('Height', '180'), ('weight', '50'), ('smoking_status', 'never smoked')])
# # h = listx.pop(8)
# # # w = listx.pop(8)
# # # print(listx)
# # # print(h[1])
# # # print(w[1])

# # # BMI = round(int(w[1]) / int(h[1]) / int(h[1]) * 10000)
# # # listx.insert(8, ("BMI", BMI))
# # print(listx)
# d = {'gender': ['Female'], 'age': ['20'], 'hypertension': ['0'], 'heart_disease': ['0'], 'ever_married': ['Yes'], 'work_type': ['children'],
#      'Residence_type': ['Urban'], 'avg_glucose_level': ['80'], 'Height': ['180'], 'weight': ['50'], 'smoking_status': ['never smoked']}

# X = tuple([])
# for x in d:
#     X.append(x)
# print(X)
from werkzeug.datastructures import MultiDict

orders = MultiDict([(1, 'GFG'), (1, 'Geeks')])
print(type(orders))
h = orders.pop(1)
orders.add('bMI',str(2))
print(orders)\

print(request.form)
        listx = MultiDict(request.form)
        h = listx.pop('weight')
        w = listx.pop('Height')
        # print(h[0])
        # print(w[0])
        BMI = round(int(w[0]) / int(h[0]) / int(h[0]) * 10000)
        listx.add('BMI',str(2))
        print(listx)