import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/new_csv_descriptions/mass_train_with_class.csv')
df_test = pd.read_csv('/Users/christopherlawton/galvanize/module_2/capstone_2/new_csv_descriptions/mass_test_with_class.csv')
df_full = pd.concat([df_train, df_test])

#count of malignant and benign images
df_full.groupby('pathology').count()

#count of patient ids
df_full['patient_id'].nunique() #892

#count of malignant and begning across CC only and mlo only
# view_paths = df_full.groupby(['image view', 'pathology'])['pathology'].count().unstack().reset_index()
#
# views = list(view_paths['image view'].values)
#
# fig1, ax1 = plt.subplots()
# ind = np.arange(2)
# width = 0.35
# p1 = ax1.bar(ind, view_paths['BENIGN'].values, width, color='b')
# p2 = ax1.bar(ind + width, view_paths['MALIGNANT'].values, width, color='g')
# ax1.set_title('Count of pathology type by image view')
# ax1.set_xticks(ind + width / 2)
# ax1.set_xticklabels(views)
# ax1.legend((p1[0], p2[0]), ('BENIGN', 'MALIGNANT'))
# plt.show()


#number of malignant and benign by mass shape top 4
# stack = df_full.groupby(['mass shape', 'pathology'])['pathology'].count().unstack()
# sort = stack.sort_values(['BENIGN', 'MALIGNANT']).reset_index() #note: the most frequent masses are IRREGULAR, ROUND, LOBULATED, OVAL
# top_shapes = sort[sort['mass shape'].isin(['IRREGULAR', 'ROUND', 'LOBULATED', 'OVAL'])]
#
# shapes = list(top_shapes['mass shape'].values)
#
#
# fig, ax = plt.subplots()
# ind = np.arange(4)
# width = 0.35
# p1 = ax.bar(ind, top_shapes['BENIGN'].values, width, color='b')
# p2 = ax.bar(ind + width, top_shapes['MALIGNANT'].values, width, color='g')
# ax.set_title('Count of pathology type for common mass shapes')
# ax.set_xticks(ind + width / 2)
# ax.set_xticklabels(shapes)
# ax.legend((p1[0], p2[0]), ('BENIGN', 'MALIGNANT'))

dense = df_full.groupby(['breast_density', 'pathology'])['pathology'].count().unstack().reset_index()

fig, ax = plt.subplots()
ind = np.arange(4)
width = 0.35
p1 = ax.bar(ind, dense['BENIGN'].values, width, color='b')
p2 = ax.bar(ind + width, dense['MALIGNANT'].values, width, color='g')
ax.set_title('Count of pathology across breast density values')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(dense['breast_density'].values)
ax.set_xlabel('Breast density value')
ax.legend((p1[0], p2[0]), ('BENIGN', 'MALIGNANT'))
