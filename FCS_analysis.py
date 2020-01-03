import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
import numpy as np



#open excel file and save into Pandas dateframe
CD8_MIX1 = pd.read_excel('clean data/CD8_MIX1_CLEAN.xls')
CD8_MIX2 = pd.read_excel('clean data/CD8_MIX2_CLEAN.xls')
CD8_MIX3 = pd.read_excel('clean data/CD8_MIX3_CLEAN.xls')
CD8_MIX4 = pd.read_excel('clean data/CD8_MIX4_CLEAN.xls')

# #Print first 10 rows to make sure everything is loaded properly
# print(CD8_MIX1.head(10))
# print(CD8_MIX2.head(10))
# print(CD8_MIX3.head(10))
# print(CD8_MIX4.head(10))
#
# #merge dataframes mix 1 and mix 2
# MIX1_MIX2 = pd.merge(CD8_MIX1, CD8_MIX2, on='ID', how='inner')
#
# # find out who was excluded during the merge of mix 1 and mix 2
# exclude1 = pd.merge(CD8_MIX1, CD8_MIX2, on = 'ID', how = 'outer', indicator=True)
# exclude1 = exclude1.query('_merge != "both"')
# #
# #save merged file mix 1 and mix 2
# writer = pd.ExcelWriter('clean data/MIX1_MIX2.xlsx', engine='xlsxwriter')
# MIX1_MIX2.to_excel(writer, sheet_name='Sheet1')
# writer.save()
#
# #save  excluded merged file mix 1 and mix 2
# writer = pd.ExcelWriter('clean data/exclude MIX1_MIX2.xlsx', engine='xlsxwriter')
# exclude1.to_excel(writer, sheet_name='Sheet1')
# writer.save()
#
#
# #merge dataframes mix 3 and mix 4
# MIX3_MIX4 = pd.merge(CD8_MIX3, CD8_MIX4, on='ID', how='inner')
#
# # find out who was excluded during the merge of mix 3 and mix 4
# exclude2 = pd.merge(CD8_MIX3, CD8_MIX4, on = 'ID', how = 'outer', indicator=True)
# exclude2 = exclude2.query('_merge != "both"')
# #
# #save merged file mix 3 and mix 4
# writer = pd.ExcelWriter('clean data/MIX3_MIX4.xlsx', engine='xlsxwriter')
# MIX3_MIX4.to_excel(writer, sheet_name='Sheet1')
# writer.save()
#
# #save  excluded merged file mix 3 and mix 4
# writer = pd.ExcelWriter('clean data/exclude MIX3_MIX4.xlsx', engine='xlsxwriter')
# exclude2.to_excel(writer, sheet_name='Sheet1')
# writer.save()
#
#
# #merge dataframes of previous merges
# Master_merge = pd.merge(MIX1_MIX2, MIX3_MIX4, on='ID', how='inner')
#
# # find out who was excluded during the merge of mix 3 and mix 4
# exclude3 = pd.merge(MIX1_MIX2, MIX3_MIX4, on = 'ID', how = 'outer', indicator=True)
# exclude3 = exclude3.query('_merge != "both"')
# #
# #save merged file
# writer = pd.ExcelWriter('clean data/Master_Merge.xlsx', engine='xlsxwriter')
# Master_merge.to_excel(writer, sheet_name='Sheet1')
# writer.save()
#
# #save  excluded merged file
# writer = pd.ExcelWriter('clean data/exclude_Master.xlsx', engine='xlsxwriter')
# exclude3.to_excel(writer, sheet_name='Sheet1')
# writer.save()

#load master merge into pandas dataframe
# Merge = pd.read_excel('clean data/Master_Merge.xlsx')

#print data frame

# print(Merge.head(10))

#sort values first by progression status and then by ID
MIX1 = CD8_MIX1.sort_values(by=['Progression 0 = No, 1 = Yes, 2= NA', 'ID'])
MIX2 = CD8_MIX2.sort_values(by=['Progression 0 = No, 1 = Yes, 2= NA', 'ID'])
MIX3 = CD8_MIX3.sort_values(by=['Progression 0 = No, 1 = Yes, 2= NA', 'ID'])
MIX4 = CD8_MIX4.sort_values(by=['Progression 0 = No, 1 = Yes, 2= NA', 'ID'])

#------------------------------------------------------------------------------------------
#-----------------------------------------MIX 1--------------------------------------------
#------------------------------------------------------------------------------------------


#split mix1 into DLBCL relapse vs non relapse and progression status
DLBCL_relapse_MIX1 = MIX1.loc[(MIX1['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX1['ID'].str.match('DLBCL'))]
DLBCL_norelapse_MIX1 = MIX1.loc[(MIX1['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX1['ID'].str.match('DLBCL'))]

#split mix1 into HL relapse vs non relapse and progression status
HL_relapse_MIX1 = MIX1.loc[(MIX1['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX1['ID'].str.match('HL'))]
HL_norelapse_MIX1 = MIX1.loc[(MIX1['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX1['ID'].str.match('HL'))]

#split mix1 into FL relapse vs non relapse and progression status
FL_relapse_MIX1 = MIX1.loc[(MIX1['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX1['ID'].str.match('FL'))]
FL_norelapse_MIX1 = MIX1.loc[(MIX1['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX1['ID'].str.match('FL'))]

#------------------------------------------------------------------------------------------
#-----------------------------------------MIX 2--------------------------------------------
#------------------------------------------------------------------------------------------
#split mix2 into DLBCL relapse vs non relapse and progression status
DLBCL_relapse_MIX2 = MIX2.loc[(MIX2['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX2['ID'].str.match('DLBCL'))]
DLBCL_norelapse_MIX2 = MIX2.loc[(MIX2['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX2['ID'].str.match('DLBCL'))]

#split mix2 into HL relapse vs non relapse and progression status
HL_relapse_MIX2 = MIX2.loc[(MIX2['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX2['ID'].str.match('HL'))]
HL_norelapse_MIX2 = MIX2.loc[(MIX2['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX2['ID'].str.match('HL'))]

#split mix2 into FL relapse vs non relapse and progression status
FL_relapse_MIX2 = MIX2.loc[(MIX2['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX2['ID'].str.match('FL'))]
FL_norelapse_MIX2 = MIX2.loc[(MIX2['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX2['ID'].str.match('FL'))]

#------------------------------------------------------------------------------------------
#-----------------------------------------MIX 3--------------------------------------------
#------------------------------------------------------------------------------------------

#split mix3 into DLBCL relapse vs non relapse and progression status
DLBCL_relapse_MIX3 = MIX3.loc[(MIX3['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX3['ID'].str.match('DLBCL'))]
DLBCL_norelapse_MIX3 = MIX3.loc[(MIX3['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX3['ID'].str.match('DLBCL'))]

#split mix3 into HL relapse vs non relapse and progression status
HL_relapse_MIX3 = MIX3.loc[(MIX3['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX3['ID'].str.match('HL'))]
HL_norelapse_MIX3 = MIX3.loc[(MIX3['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX3['ID'].str.match('HL'))]

#split mix3 into FL relapse vs non relapse and progression status
FL_relapse_MIX3 = MIX3.loc[(MIX3['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX3['ID'].str.match('FL'))]
FL_norelapse_MIX3 = MIX3.loc[(MIX3['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX3['ID'].str.match('FL'))]

#------------------------------------------------------------------------------------------
#-----------------------------------------MIX 4--------------------------------------------
#------------------------------------------------------------------------------------------
#split mix3 into DLBCL relapse vs non relapse and progression status
DLBCL_relapse_MIX4 = MIX4.loc[(MIX4['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX4['ID'].str.match('DLBCL'))]
DLBCL_norelapse_MIX4 = MIX4.loc[(MIX4['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX4['ID'].str.match('DLBCL'))]

#split mix3 into HL relapse vs non relapse and progression status
HL_relapse_MIX4 = MIX4.loc[(MIX4['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX4['ID'].str.match('HL'))]
HL_norelapse_MIX4 = MIX4.loc[(MIX4['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX4['ID'].str.match('HL'))]

#split mix3 into FL relapse vs non relapse and progression status
FL_relapse_MIX4 = MIX4.loc[(MIX4['Progression 0 = No, 1 = Yes, 2= NA'] == 1) & (MIX4['ID'].str.match('FL'))]
FL_norelapse_MIX4 = MIX4.loc[(MIX4['Progression 0 = No, 1 = Yes, 2= NA'] == 0) & (MIX4['ID'].str.match('FL'))]


#
# DLBCL_MIX1 = DLBCL_relapse_MIX1.append(DLBCL_norelapse_MIX1, ignore_index=True)
# HL_MIX1 = HL_relapse_MIX1.append(HL_norelapse_MIX1, ignore_index=True)
# FL_MIX1 = FL_relapse_MIX1.append(FL_norelapse_MIX1, ignore_index=True)
#
# DLBCL_MIX1 = DLBCL_MIX1.append(HL_MIX1, ignore_index=True)
# DLBCL_MIX1 = DLBCL_MIX1.append(FL_MIX1, ignore_index=True)
#
#
#
# DLBCL_MIX1 = DLBCL_MIX1.drop(['CD8','MIX','CD38','EOMES','CD44','CXCR5','LY108','TBET'],axis=1)
# print(DLBCL_MIX1.head(10))
#
# DLBCL_MIX1 = pd.melt(DLBCL_MIX1, id_vars=["Progression 0 = No, 1 = Yes, 2= NA", "ID"], var_name="measurement")
# DLBCL_MIX1['ID'] = DLBCL_MIX1['ID'].str.replace('\d+', '')
#
#
# print(DLBCL_MIX1)
#
#
# g = sns.catplot(x="measurement", y="value", hue="ID", data=DLBCL_MIX1,
#                 height=8, kind="swarm", palette="bright",legend = True,ci=None, legend_out=True)
# g.despine(left=True)
# g.set_ylabels("Values")
#
# plt.legend(['No Progression','Progression'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()


#
# #-----------------------------------------------------------------------------------
#
# DLBCL_MIX2 = DLBCL_relapse_MIX2.append(DLBCL_norelapse_MIX2, ignore_index=True)
#
# DLBCL_MIX2 = DLBCL_MIX2.drop(['ID','CD8','MIX'],axis=1)
# print(DLBCL_MIX2.head(10))
#
# DLBCL_MIX2 = pd.melt(DLBCL_MIX2, "Progression 0 = No, 1 = Yes, 2= NA", var_name="measurement")
#
#
# g = sns.catplot(x="measurement", y="value", hue="Progression 0 = No, 1 = Yes, 2= NA", data=DLBCL_MIX2,
#                 height=8, kind="swarm", palette="Pastel1",legend = False,ci=None)
# g.despine(left=True)
# g.set_ylabels("Values")
#
# plt.legend(['No Progression','Progression'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#
# plt.show()
#
# #-----------------------------------------------------------------------------------
#
#
# HL_MIX1 = HL_relapse_MIX1.append(HL_norelapse_MIX1, ignore_index=True)
#
# HL_MIX1 = HL_MIX1.drop(['ID','CD8','MIX','CD38','EOMES','CD44','CXCR5','LY108','TBET'],axis=1)
# print(HL_MIX1.head(10))
#
# HL_MIX1 = pd.melt(HL_MIX1, "Progression 0 = No, 1 = Yes, 2= NA", var_name="measurement")
#
#
# g = sns.catplot(x="measurement", y="value", hue="Progression 0 = No, 1 = Yes, 2= NA", data=HL_MIX1,
#                 height=8, kind="swarm", palette="Pastel1",legend = False,ci=None,)
# g.despine(left=True)
# g.set_ylabels("Values")
#
# plt.legend(["No Progression","Progression"],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# plt.show()
#

#-----------------------MIX 4------------------------------------------------------------

DLBCL_MIX1 = DLBCL_relapse_MIX4.append(DLBCL_norelapse_MIX4, ignore_index=True)
HL_MIX1 = HL_relapse_MIX4.append(HL_norelapse_MIX4, ignore_index=True)
FL_MIX1 = FL_relapse_MIX4.append(FL_norelapse_MIX4, ignore_index=True)

DLBCL_MIX1 = DLBCL_MIX1.append(HL_MIX1, ignore_index=True)
DLBCL_MIX1 = DLBCL_MIX1.append(FL_MIX1, ignore_index=True)



DLBCL_MIX1 = DLBCL_MIX1.drop(['CD8','MIX','CD103','2B4','CTLA-4'],axis=1)
print(DLBCL_MIX1.head(10))

DLBCL_MIX1 = pd.melt(DLBCL_MIX1, id_vars=["Progression 0 = No, 1 = Yes, 2= NA", "ID"], var_name="measurement")
DLBCL_MIX1['ID'] = DLBCL_MIX1['ID'].str.replace('\d+', '')


print(DLBCL_MIX1)


Markers = ['o','x']
g = sns.catplot(x="measurement", y="value", hue="ID", data=DLBCL_MIX1,
                height=12, kind="swarm", palette="bright",legend = True,ci=None, legend_out=True,col='Progression 0 = No, 1 = Yes, 2= NA')
g.despine(left=True)
g.set_ylabels("Values")

plt.legend(['No Progression','Progression'],bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('TEST.png', dpi=400)
plt.show()