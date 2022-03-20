
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.linear_model import LogisticRegression

#.txt dosyasını csv dosyasına çevrilmesi ve column isimleri eklenmesi
df = pd.read_csv ('wdbc.txt',names=["ID","Class","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"])

#sınıflandırmaya etki etmeyecek ID sütunun silinmesi
df.drop(["ID"],axis=1,inplace=True)

#Null veri olup olmadığı kontrol edildi olmadığı için işlem yapılmadı
print(df.isnull().values.any())

#Kategorik sütun olan Class sütunu nümerik veriye çevrildi.
df["Class"]= df['Class'].replace(['M'],1)
df["Class"]= df['Class'].replace(['B'],0)

#Feature lar arasındaki ilişki heatmap ile gözlemlenir.
corr = df.corr()
plt.figure(figsize = (16,16))
ax1 = sns.heatmap(corr, cbar=0,annot = True, linewidths=2,vmax=1, vmin=0, square=True, cmap='Blues')
plt.show()

#Verinin imblance olup olmadığı kontrol edilir.
numbers=[df["Class"].value_counts()[0],df["Class"].value_counts()[1]]
names=["Benign","Malign"]
plt.bar(names,numbers,color ='maroon')
plt.xlabel("Type of tumor")
plt.ylabel("Number of tumors")
plt.show()

X=df[["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"]]
y=df["Class"]

#Scale işlemi için standard scaler tercih ettim
scaler = StandardScaler()
scaler.fit(X)
scaled = scaler.fit_transform(X)
scaleddf= pd.DataFrame(scaled, columns=X.columns)

#scaled df kaydedilir
scaleddf["class"]=y
scaleddf.to_csv("data.csv")

X_train, X_test, y_train, y_test = train_test_split(scaled, y, test_size=0.2)
y_testcopy = y_test.copy(deep=True)
y_testcopy=y_testcopy.to_list()

clf=RandomForestClassifier(n_estimators=20) # 10-100 arası değerleri denedim fakat önemli ölçüde değişim olmadığından 20 de karar kıldım
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print("Random Forest Accuracy:",metrics.accuracy_score(y_test, y_pred))
report = metrics.classification_report(y_test, y_pred)
print("Random Forest other metrics :")
print(report)
matrix = metrics.confusion_matrix(y_test, y_pred)
print("Random Forest Confusion Matrix :")
print(matrix)

fn=[]
fp=[]
for i in range(len(y_test)):
  if(y_testcopy[i]!=y_pred[i]):
    if(y_pred[i]==1):
      fp.append(i)
    else:
      fn.append(i)

print("False Negative Indicies :",fn)
print("False Positive Indicies :",fp)

knn = KNeighborsClassifier(n_neighbors=5)# 3 ile 100 arasınadki değerleri denedim en yüksek sonucu veren komşu sayısana karar verdim
knn.fit(X_train, y_train)
y_pred2 = knn.predict(X_test)
print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred2))
report2 = metrics.classification_report(y_test, y_pred2)
print("KNN other metrics :")
print(report2)
matrix2 = metrics.confusion_matrix(y_test, y_pred2)
print("KNN Confusion Matrix :")
print(matrix2)

fn=[]
fp=[]
for i in range(len(y_test)):
  if(y_testcopy[i]!=y_pred2[i]):
    if(y_pred2[i]==1):
      fp.append(i)
    else:
      fn.append(i)

print("False Negative Indicies :",fn)
print("False Positive Indicies :",fp)

svmclf = svm.SVC(kernel='linear',probability=True) #  rbf linear ve poly kernel tipleri denenip linear seçilmiştir.
svmclf.fit(X_train, y_train)
y_pred3 = svmclf.predict(X_test)

print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred3))
report3 = metrics.classification_report(y_test, y_pred3)
print("SVM other metrics :")
print(report3)
matrix3 = metrics.confusion_matrix(y_test, y_pred3)
print("SVM Confusion Matrix :")
print(matrix3)

fn=[]
fp=[]
for i in range(len(y_test)):
  if(y_testcopy[i]!=y_pred3[i]):
    if(y_pred3[i]==1):
      fp.append(i)
    else:
      fn.append(i)

print("False Negative Indicies :",fn)
print("False Positive Indicies :",fp)

model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X_train,y_train)
y_pred4=model.predict(X_test)

print("Linear Regression Accuracy:",metrics.accuracy_score(y_test, y_pred4))
report4 = metrics.classification_report(y_test, y_pred4)
print("Linear Regression other metrics :")
print(report4)
matrix4 = metrics.confusion_matrix(y_test, y_pred4)
print("Linear Regression Confusion Matrix :")
print(matrix4)

fn=[]
fp=[]
for i in range(len(y_test)):
  if(y_testcopy[i]!=y_pred4[i]):
    if(y_pred4[i]==1):
      fp.append(i)
    else:
      fn.append(i)

print("False Negative Indicies :",fn)
print("False Positive Indicies :",fp)

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
y_pred5=dtree.predict(X_test)

print("Decision Tree Accuracy:",metrics.accuracy_score(y_test, y_pred5))
report5 = metrics.classification_report(y_test, y_pred5)
print("Decision Tree other metrics :")
print(report5)
matrix5 = metrics.confusion_matrix(y_test, y_pred5)
print("Decision Tree Confusion Matrix :")
print(matrix5)

fn=[]
fp=[]
for i in range(len(y_test)):
  if(y_testcopy[i]!=y_pred5[i]):
    if(y_pred5[i]==1):
      fp.append(i)
    else:
      fn.append(i)

print("False Negative Indicies :",fn)
print("False Positive Indicies :",fp)

#KNN,RF,SVM,LR algoritmalarının tümü benzer doğruluk oranları -%96-%97- verdi.
#DT algoritması ise diğerlerinden bariz bir biçimde düşük performans verdi.
#Bunun sebebi olarak DT algoritmasının overfit olmaya yatkınlığı örnek verilebilir.
#Genel anlamda diğer algoritmaların başarılı olmasının sebepleri arasında veri setinin temiz ve scale edilmiş olması verilebilir.
