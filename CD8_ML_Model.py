import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
import glob
import matplotlib.pyplot as plt
import math
pd.set_option('display.max_columns', None)
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_graphviz
import random
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer

# # dlbcl1 ['Progression'] = 0

#Set the path of where all our files are
# path = r'FCS Raw Data /Mix 1/' # use your path
path = r'FCS Raw Data /Mix 2/' # use your path

#load all the files into a list, seperated by subgroups DLBCL, HL , FL
# DLBCL_files = glob.glob(path + "/export_Mix 1_DLBCL*.csv")
# HL_files = glob.glob(path + "/export_Mix 1_HL*.csv")
# FL_files = glob.glob(path + "/export_Mix 1_FL*.csv")

#mix2
DLBCL_files = glob.glob(path + "/export_Mix 2_DLBCL*.csv")
HL_files = glob.glob(path + "/export_Mix 2_HL*.csv")
FL_files = glob.glob(path + "/export_Mix 2_FL*.csv")


#checking lists are of correct length ( # of files!!)
print(len(DLBCL_files))
print(len(HL_files))
print(len(FL_files))

#print list of files to make sure we included everyone
print(*DLBCL_files, sep ="\n")
print("==========================================")
print(*HL_files, sep = "\n")
print("==========================================")
print(*FL_files, sep = "\n")
print("==========================================")

#empty lists that will contain dataframes for each subgroup
DLBCL_0 = []
DLBCL_1 =[]
HL_0 = []
HL_1 = []
FL_0 = []
FL_1 = []

print(DLBCL_files)


#========================================================================================
#================================ DLBCL =================================================
#========================================================================================
#index positions for DLBCL file names that have no progression
#mix1
# DLBCL_no = [1,5,7,9,10,11,13]
# DLBCL_yes = [0,2,3,4,6,8,12,14]

# #mix2
DLBCL_no = [0,1,5,8,10,12,14]
DLBCL_yes = [2,3,4,6,7,9,11,13]

#for loop to seperate DLBCL files from progression to  no progression, add columns of 0 or 1's then append
for filename in DLBCL_files:
    for x in DLBCL_no:
        if filename == DLBCL_files[x]:
            # print("no prog " + filename)
            df = pd.read_csv(filename, index_col=None, header=0)
            df ['Progression'] = 0
            print(df.shape)
            DLBCL_0.append(df)
    for y in DLBCL_yes:
        if filename == DLBCL_files[y]:
            # print("yes prog " + filename)
            df1 = pd.read_csv(filename, index_col=None, header=0)
            df1['Progression'] = 1
            print(df1.shape)
            DLBCL_1.append(df1)




#combine the two lists of DLBCL dataframes and then concat into one big data frame
DLBCL_MERGE = DLBCL_1 + DLBCL_0
DLBCL_MIX1 = pd.concat(DLBCL_MERGE, axis=0, ignore_index=True)
print("DLBCL SIZE")
print(DLBCL_MIX1.shape)


#========================================================================================
#================================ HL  =================================================
#========================================================================================

#print HL files
print(HL_files)

#index positions for HL file names that have no progression
#mix 1
# HL_no = [0,2,3,4,5,6,8,9,11,13,14,16]
# HL_yes = [1,7,10,12,15,17]

#mix2
HL_no = [0,3,4,5,6,7,8,9,10,12,14,17]
HL_yes = [1,2,11,13,15,16]
#for loop to seperate HL files from progression to  no progression, add columns of 0 or 1's then append
for filename in HL_files:
    for x in HL_no:
        if filename == HL_files[x]:
            # print("no prog " + filename)
            df = pd.read_csv(filename, index_col=None, header=0)
            df ['Progression'] = 0
            print(df.shape)
            HL_0.append(df)
    for y in HL_yes:
        if filename == HL_files[y]:
            # print("yes prog " + filename)
            df1 = pd.read_csv(filename, index_col=None, header=0)
            df1['Progression'] = 1
            print(df1.shape)
            HL_1.append(df1)


#combine the two lists of DLBCL dataframes and then concat into one big data frame
HL_MERGE = HL_1 + HL_0
HL_MIX1 = pd.concat(HL_MERGE, axis=0, ignore_index=True)
print("HL SIZE")
print(HL_MIX1.shape)

# ##############################
#
# #========================================================================================
# #================================ FL  =================================================
# #========================================================================================
#
# #print FL files
# print(FL_files)
# #index positions for HL file names that have no progression
# # #mix1
# # FL_no = [0,2,3,5,6,8,9,10,12,13]
# # FL_yes = [1,4,7,11,14,15]
#
# mix2
FL_no = [1,3,6,7,8,11,12,13,14,15]
FL_yes = [0,2,4,5,9,10]

#for loop to seperate HL files from progression to  no progression, add columns of 0 or 1's then append
for filename in FL_files:
    for x in FL_no:
        if filename == FL_files[x]:
            # print("no prog " + filename)
            df = pd.read_csv(filename, index_col=None, header=0)
            df ['Progression'] = 0
            print(df.shape)
            FL_0.append(df)
    for y in FL_yes:
        if filename == FL_files[y]:
            # print("yes prog " + filename)
            df1 = pd.read_csv(filename, index_col=None, header=0)
            df1['Progression'] = 1
            print(df1.shape)
            FL_1.append(df1)

#combine the two lists of DLBCL dataframes and then concat into one big data frame
FL_MERGE = FL_1 + FL_0
FL_MIX1 = pd.concat(FL_MERGE, axis=0, ignore_index=True)
print("FL SIZE")
print(FL_MIX1.shape)

# concating all the lists into dataframes

DLBCL_MERGE = DLBCL_1 + DLBCL_0
HL_MERGE = HL_1 + HL_0
FL_MERGE = FL_1 + FL_0

merge1 = DLBCL_MERGE + HL_MERGE
merge2 = merge1 + FL_MERGE


# DLBCL_HL_FL_MIX1 = pd.concat(DLBCL_MERGE, axis=0, ignore_index=True)
DLBCL_HL_FL_MIX1 = pd.concat(merge2, axis=0, ignore_index=True)

print("TOTAL SIZE")
print(DLBCL_HL_FL_MIX1.shape)

# #save master DLBCL data frame
# writer = pd.ExcelWriter('DLBCL_SINGLECELL_MASTER.xlsx', engine='xlsxwriter')
# DLBCL_MIX1.to_excel(writer, sheet_name='Sheet1')
# writer.save()
#
#print column names so we can see what to drop
print(DLBCL_HL_FL_MIX1.columns)
#
#drop these columns
#MIX1
# DLBCL_HL_FL_MIX1 = DLBCL_HL_FL_MIX1.drop(['FSC-A','FSC-H','FSC-W','SSC-A','SSC-H','SSC-W','Comp-Alexa Fluor 700-A :: CD8','Comp-BUV395-A :: CD3',
#        'Comp-BUV737-A','Comp-BV605-A :: DEAD', 'Comp-BV711-A','Comp-V500-A :: CD4','Time','Comp-PE-CF594-A :: T-Bet','Comp-APC-Cy7-A :: CD44','Comp-PE-A :: Ly108','Comp-BV786-A :: CXCR5'],axis=1)

# DLBCL_HL_FL_MIX1 = DLBCL_HL_FL_MIX1.drop(['FSC-A','FSC-H','FSC-W','SSC-A','SSC-H','SSC-W','Comp-Alexa Fluor 700-A :: CD8','Comp-BUV395-A :: CD3',
#        'Comp-BUV737-A','Comp-BV605-A :: DEAD', 'Comp-BV711-A','Comp-V500-A :: CD4','Time','Comp-PE-CF594-A :: T-Bet'],axis=1)

#MIX2
DLBCL_HL_FL_MIX1 = DLBCL_HL_FL_MIX1.drop(['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W','Comp-Alexa Fluor 700-A :: CD8','Comp-BUV737-A :: CD3', 'Comp-BV605-A :: DEAD',
       'Comp-BV711-A','Comp-PE-A :: TOX', 'Comp-PE-CF594-A', 'Comp-V500-A :: CD4', 'Time','Comp-BV786-A :: CD27'],axis=1)

# DLBCL_HL_FL_MIX1 = DLBCL_HL_FL_MIX1.drop(['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A', 'SSC-H', 'SSC-W', 'Comp-BV605-A :: DEAD',
#        'Comp-BV711-A','Comp-PE-A :: TOX', 'Comp-PE-CF594-A', 'Time'],axis=1)

#check that everything is dropped correctly
print(DLBCL_HL_FL_MIX1.columns)

# Make sure each column is a numeric type
print(DLBCL_HL_FL_MIX1.dtypes)

print(DLBCL_HL_FL_MIX1.shape)
#create our x/y dataset
# x_dataset=DLBCL_HL_FL_MIX1.iloc[:, 0:9]
x_dataset=DLBCL_HL_FL_MIX1.iloc[:, 0:7]
y_dataset= DLBCL_HL_FL_MIX1.iloc[:,-1]



#checking the size of both dataframes to make sure they are equal in rows and offset in columns
print(x_dataset.shape)
print(y_dataset.shape)

#save feature names and y dataset
x_featurenames = x_dataset.columns
print(x_featurenames)
class_names = y_dataset.name
#
# y target name
print(class_names)

#convert dataframes to numpy arrays
x_dataset =x_dataset.to_numpy()
y_dataset=y_dataset.to_numpy()

#create a standardScaler object to feature scale our dataset
# sc = StandardScaler()


#split into training, testing, and val sets
X_train, X_test, y_train, y_test = train_test_split(x_dataset, y_dataset, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

#fit and transform the data into the scaled data
# sc.fit(X_train)
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# X_val = sc.transform(X_val)


# sc= MinMaxScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# X_val = sc.transform(X_val)

sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_val = sc.transform(X_val)

#apply PCA
# pca = PCA(n_components=2)
#
# X_pca = pca.fit_transform(X_train)
# # X_test = pca.transform(X_test)
# # X_val = pca.transform(X_val)
#
# X_pca_df = pd.DataFrame(data = X_pca , columns = ['principal component 1', 'principal component 2'])
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#
#
# target = pd.Series(data=y_train)
# target = target.rename('Progression')
#
# X_pca_df = pd.concat((X_pca_df,target), axis=1)
# pca_info = pd.DataFrame(pca.components_,columns=x_featurenames,index = ['PC-1','PC-2'])
# print (pca_info.T)
#
#
#
#
#
# plt.figure()
# plt.figure(figsize=(8,8))
# plt.xticks([])
# plt.yticks()
# plt.xlabel('Principal Component - 1',fontsize=20)
# plt.ylabel('Principal Component - 2',fontsize=20)
# plt.title("Principal Component Analysis of MIX 2",fontsize=20)
# targets = [0, 1]
# colors = ['b', 'r']
#
# for target, color in zip(targets,colors):
#     indicesToKeep = X_pca_df['Progression'] == target
#     plt.scatter(X_pca_df.loc[indicesToKeep, 'principal component 1']
#                , X_pca_df.loc[indicesToKeep, 'principal component 2'], c = color, s=0.5)
#
#
# plt.legend(targets,prop={'size': 15},markerscale=2,loc='upper right')
#
#
# plt.show()





print("------------ X DATASET POINTS ---------")
print("x train:" , X_train.size, " x test: " ,X_test.size," x_train: ", X_train.size," x val: ", X_val.size )
print("------------ Y DATASET POINTS  --------")
print("y train:" ,  y_train.size, " y test: " ,y_test.size," y_train: ", y_train.size," y val: ", y_val.size )
print("------------END ---------")



clf=RandomForestClassifier(n_estimators=200,n_jobs=-1,oob_score=True)

# all_accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=5)
#
# print(all_accuracies)
#
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000],
#     'criterion': ['gini', 'entropy']
# }
#
#
#
# grid_search = GridSearchCV(estimator = clf, param_grid = param_grid,
#                           cv = 5, n_jobs = -1)
#
# grid_search.fit(X_train, y_train)
#
# best_parameters = grid_search.best_params_
# print(best_parameters)
#
# best_result = grid_search.best_score_
# print(best_result)




clf.fit(X_train,y_train)
y_pred = clf.predict(X_val)




print("Accuracy: ", metrics.accuracy_score(y_val,y_pred))
print("Balanaced Accuracy: ", metrics.balanced_accuracy_score(y_val,y_pred))
print(metrics.classification_report(y_val,y_pred))

pred_y_test = clf.predict(X_test)

print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")
print("-------------------------------------------------------------------")
print("TESTS USING TEST SET")
print(metrics.classification_report(y_test, pred_y_test))
print("Accuracy: ", metrics.accuracy_score(y_test, pred_y_test))
print("Balanaced Accuracy: ", metrics.balanced_accuracy_score(y_test, pred_y_test))


rand = random.randint(0, 100877)
print("Index: ", rand," Y: Predicated: ", pred_y_test[rand])

print("Index: ",rand,"Y: Real: ",y_test[rand])

y_prob = clf.predict_proba(X_test)
print("AUC: ",roc_auc_score(y_test, y_prob[:, 1]))


cm=confusion_matrix(y_test, pred_y_test)
print(cm)

tn= cm[0,0]
fn= cm[1,0]
tp= cm[1,1]
fp= cm[0,1]

print("True Negatives: ", tn)
print("False Negatives: ", fn)
print("True Positives: ", tp)
print("False Positives: ", fp)

fpr, tpr, threshold = metrics.roc_curve(y_test, y_prob[:, 1])
roc_auc = metrics.auc(fpr, tpr)

acc_confusion= ((tp+tn)/(tp+tn+fp+fn))

print("Accuracy of confusion matrix: ",acc_confusion)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('Normalized MIX 2 ML-MODEL-auc - accuracy.png', dpi=400)
plt.show()

importances = clf.feature_importances_

std = np.std([tree.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")

for f in range(X_test.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(25,10))
# plt.figure()
plt.title("Feature importances")
plt.bar(range(X_test.shape[1]), importances[indices],
       color="b", yerr=std[indices], align="center", width=0.8)
plt.xticks(range(X_test.shape[1]), x_featurenames[indices])
plt.xlim([-1, X_test.shape[1]])
plt.ylabel('Feature Importance')
plt.savefig('Normalized 2 ML-MODEL-feature_importance - accuracy.png', dpi=400)
plt.show()

# clf1 = svm.SVC(gamma='scale')
# clf1.fit(X_train, y_train)
#
#
# pred = clf1.predict(y_test)
# print("Accuracy of SVM: ", metrics.accuracy_score(y_test, pred))


