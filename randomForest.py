# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#veri seti okunur
dataset = pd.read_csv('pima-indians-diabetes.csv')

#öznitelikler atanır
data_values = dataset.iloc[:, 0:8]
#sınıflandırıcı atanır
data_target = dataset.iloc[:, 8]

#%80 eğitim %20 test verisi oluşturulur.
X_train, X_test, Y_train, Y_test = train_test_split(data_values, data_target, test_size=0.20, random_state=21)

#veriler arası uyumsuzluk önlenir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#RandomForest sınıflandırıcısı ormanda bulunan 50 ağaç ile oluşturulur
classifier = RandomForestClassifier(n_estimators=50, criterion='entropy', random_state=42)
classifier.fit(X_train, Y_train)

pred = classifier.predict(X_test)

count_true_pred = 0

for i in range(len(pred)):
    if pred[i] == Y_test.values[i]:
        count_true_pred = count_true_pred + 1

acc = count_true_pred / len(pred)
print("Accuracy:",acc)

con_mat = confusion_matrix(Y_test, pred)

print(con_mat)

#hata matrisi grafiği çizdirilir.
plt.matshow(con_mat)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
