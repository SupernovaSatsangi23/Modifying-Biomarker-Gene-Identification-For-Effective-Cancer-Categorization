import numpy as np
import pandas as pd
import random
from sklearn.feature_selection import f_regression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import scipy as stats
import tkinter as tk






root = tk.Tk()

root.title("CSM#967 Project GUI")

root.minsize(1300, 1000)
root.maxsize(1300, 1000)

root.configure(background='#121212')


reading_data_frame = tk.Frame(root, background="#121212")
reading_data_frame.grid(row=0, column=0)


















# reading csv file
data = pd.read_csv("/home/venom/Desktop/CSM967_Dissertation_Moksh_2006531/archive/METABRIC_RNA_Mutation.csv")

# GUI
reading_data_label_frame = tk.LabelFrame(reading_data_frame, text=' Reading CSV data ', width=200, font=('Ubuntu 10 bold'), background="#121212", foreground= "#39ff14")
reading_data_label_frame.grid(row=0, column=0, padx=10, pady=6, sticky="n")

data_path_label = tk.Label(reading_data_label_frame, text="Data path: /home/venom/Desktop/CSM967_Dissertation_Moksh_2006531/archive/METABRIC_RNA_Mutation.csv", background="#121212", foreground='white', font=('Ubuntu 10'))
data_path_label.grid(row=0, column=0, padx=10, pady=6, sticky="w")
# GUI


# df = {'patient_id':data["patient_id"], 'age_at_diagnosis':data['age_at_diagnosis'], 'type_of_breast_surgery':data['type_of_breast_surgery'], 'cancer_type':data['cancer_type'], 'cancer_type_detailed':data['cancer_type_detailed'], 'cellularity':data['cellularity'], 'chemotherapy':data['chemotherapy'], 'pam50_+_claudin-low_subtype':data['pam50_+_claudin-low_subtype'], 'cohort':data['cohort'], 'er_status_measured_by_ihc':data['er_status_measured_by_ihc'], 'er_status':data['er_status'], 'neoplasm_histologic_grade':data['neoplasm_histologic_grade'], 'her2_status_measured_by_snp6':data['her2_status_measured_by_snp6'], 'her2_status':data['her2_status'], 'tumor_other_histologic_subtype':data['tumor_other_histologic_subtype'], 'hormone_therapy':data['hormone_therapy'], 'inferred_menopausal_state':data['inferred_menopausal_state'], 'integrative_cluster':data['integrative_cluster'], 'primary_tumor_laterality':data['primary_tumor_laterality'], 'lymph_nodes_examined_positive':data['lymph_nodes_examined_positive'], 'mutation_count':data['mutation_count'], 'nottingham_prognostic_index':data['nottingham_prognostic_index'], 'oncotree_code':data['oncotree_code'], 'overall_survival_months':data['overall_survival_months'], 'overall_survival':data['overall_survival'], 'pr_status':data['pr_status'], 'radio_therapy':data['radio_therapy'], 'tumor_size':data['tumor_size'], 'tumor_stage':data['tumor_stage'], 'death_from_cancer':data['death_from_cancer']}
df = {'patient_id':data["patient_id"], 'age_at_diagnosis':data['age_at_diagnosis'], 'cancer_type_detailed':data['cancer_type_detailed'], 'chemotherapy':data['chemotherapy'], 'cohort':data['cohort'], 'neoplasm_histologic_grade':data['neoplasm_histologic_grade'], 'hormone_therapy':data['hormone_therapy'], 'lymph_nodes_examined_positive':data['lymph_nodes_examined_positive'], 'mutation_count':data['mutation_count'], 'nottingham_prognostic_index':data['nottingham_prognostic_index'], 'overall_survival_months':data['overall_survival_months'], 'overall_survival':data['overall_survival'], 'radio_therapy':data['radio_therapy'], 'tumor_size':data['tumor_size'], 'tumor_stage':data['tumor_stage']}
#df = {'patient_id':data["patient_id"], 'age_at_diagnosis':data['age_at_diagnosis']}


clinical_cancer_data = pd.DataFrame(df)
print(clinical_cancer_data.head())

# removing all missing ( NAN ) values row-wise
clinical_cancer_data.dropna(inplace=True)






# type_of_breast_surgery_unique = clinical_cancer_data.type_of_breast_surgery.unique()
# #print(type_of_breast_surgery_unique)
# clinical_cancer_data.loc[clinical_cancer_data["type_of_breast_surgery"] == "BREAST CONSERVING", "type_of_breast_surgery"] = 0
# clinical_cancer_data.loc[clinical_cancer_data["type_of_breast_surgery"] == "MASTECTOMY", "type_of_breast_surgery"] = 1
# #print(clinical_cancer_data["type_of_breast_surgery"])


# for idx, feat in enumerate(clinical_cancer_data.cancer_type.unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["cancer_type"] == f"{feat}", "cancer_type"] = idx
# print(clinical_cancer_data["cancer_type"])


# for idx, feat in enumerate(clinical_cancer_data.cellularity.unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["cellularity"] == f"{feat}", "cellularity"] = idx
# print(clinical_cancer_data["cellularity"])


# for idx, feat in enumerate(clinical_cancer_data["pam50_+_claudin-low_subtype"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["pam50_+_claudin-low_subtype"] == f"{feat}", "pam50_+_claudin-low_subtype"] = idx
# print(clinical_cancer_data["pam50_+_claudin-low_subtype"])


# for idx, feat in enumerate(clinical_cancer_data["er_status_measured_by_ihc"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["er_status_measured_by_ihc"] == f"{feat}", "er_status_measured_by_ihc"] = idx
# print(clinical_cancer_data["er_status_measured_by_ihc"])


# for idx, feat in enumerate(clinical_cancer_data["er_status"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["er_status"] == f"{feat}", "er_status"] = idx
# print(clinical_cancer_data["er_status"])


# for idx, feat in enumerate(clinical_cancer_data["her2_status_measured_by_snp6"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["her2_status_measured_by_snp6"] == f"{feat}", "her2_status_measured_by_snp6"] = idx
# print(clinical_cancer_data["her2_status_measured_by_snp6"])


# for idx, feat in enumerate(clinical_cancer_data["her2_status"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["her2_status"] == f"{feat}", "her2_status"] = idx
# print(clinical_cancer_data["her2_status"])


# for idx, feat in enumerate(clinical_cancer_data["tumor_other_histologic_subtype"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["tumor_other_histologic_subtype"] == f"{feat}", "tumor_other_histologic_subtype"] = idx
# print(clinical_cancer_data["tumor_other_histologic_subtype"])


# for idx, feat in enumerate(clinical_cancer_data["inferred_menopausal_state"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["inferred_menopausal_state"] == f"{feat}", "inferred_menopausal_state"] = idx
# print(clinical_cancer_data["inferred_menopausal_state"])


# for idx, feat in enumerate(clinical_cancer_data["integrative_cluster"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["integrative_cluster"] == f"{feat}", "integrative_cluster"] = idx
# print(clinical_cancer_data["integrative_cluster"])


# for idx, feat in enumerate(clinical_cancer_data["primary_tumor_laterality"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["primary_tumor_laterality"] == f"{feat}", "primary_tumor_laterality"] = idx
# print(clinical_cancer_data["primary_tumor_laterality"])


# for idx, feat in enumerate(clinical_cancer_data["oncotree_code"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["oncotree_code"] == f"{feat}", "oncotree_code"] = idx
# print(clinical_cancer_data["oncotree_code"])


# for idx, feat in enumerate(clinical_cancer_data["pr_status"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["pr_status"] == f"{feat}", "pr_status"] = idx
# print(clinical_cancer_data["pr_status"])


# for idx, feat in enumerate(clinical_cancer_data["death_from_cancer"].unique()):
# 	clinical_cancer_data.loc[clinical_cancer_data["death_from_cancer"] == f"{feat}", "death_from_cancer"] = idx
# print(clinical_cancer_data["death_from_cancer"])








# testing if f_regression works with these datatypes

# 'X' ( FEATURES )
X = clinical_cancer_data.loc[:, clinical_cancer_data.columns != "cancer_type_detailed"]
#print(X.columns)
print(X.head(100), f"Size of dataset (number of samples) = {X.shape}")
print(f"\nNumber of features in the dataset = {len(X.columns) + 1}\nNumber of features fed in the algorithm = {len(X.columns)}")


# editing for 'y' ( CLASSES/LABELS )
clinical_cancer_data.loc[clinical_cancer_data["cancer_type_detailed"] == "Breast", "cancer_type_detailed"] = 0 # Regular
clinical_cancer_data.loc[clinical_cancer_data["cancer_type_detailed"] == "Breast Invasive Ductal Carcinoma", "cancer_type_detailed"] = 1
clinical_cancer_data.loc[clinical_cancer_data["cancer_type_detailed"] == "Breast Mixed Ductal and Lobular Carcinoma", "cancer_type_detailed"] = 2
clinical_cancer_data.loc[clinical_cancer_data["cancer_type_detailed"] == "Breast Invasive Lobular Carcinoma", "cancer_type_detailed"] = 3
clinical_cancer_data.loc[clinical_cancer_data["cancer_type_detailed"] == "Breast Invasive Mixed Mucinous Carcinoma", "cancer_type_detailed"] = 4
clinical_cancer_data.loc[clinical_cancer_data["cancer_type_detailed"] == "Metaplastic Breast Cancer", "cancer_type_detailed"] = 5

class_dict = {0:'Normal', 1:'Breast Invasive Ductal Carcinoma', 2:'Breast Mixed Ductal and Lobular Carcinoma', 3:'Breast Invasive Lobular Carcinoma', 4:'Breast Invasive Mixed Mucinous Carcinoma', 5:'Metaplastic Breast Cancer'}

#print(clinical_cancer_data["cancer_type_detailed"])

y = clinical_cancer_data["cancer_type_detailed"]
print(y.shape)


# GUI
data_cleaning_label_frame = tk.LabelFrame(reading_data_frame, text=' Data cleaning ', font=('Ubuntu 10 bold'), background="#121212", foreground= "#39ff14")
data_cleaning_label_frame.grid(row=0, column=1, padx=10, pady=6, sticky="n")

data_cleaning_label1 = tk.Label(data_cleaning_label_frame, text=f"Original dataset shape: {data.shape}", background="#121212", foreground='white', font=('Ubuntu 10'))
data_cleaning_label1.grid(row=0, column=0, padx=10, pady=6, sticky="w")

data_cleaning_label2 = tk.Label(data_cleaning_label_frame, text="Removing NaN row-wise", background="#121212", foreground='white', font=('Ubuntu 10'))
data_cleaning_label2.grid(row=1, column=0, padx=10, pady=6, sticky="w")

data_cleaning_label3 = tk.Label(data_cleaning_label_frame, text="Removing String values column-wise", background="#121212", foreground='white', font=('Ubuntu 10'))
data_cleaning_label3.grid(row=2, column=0, padx=10, pady=6, sticky="w")

data_cleaning_label4 = tk.Label(data_cleaning_label_frame, text=f"Processed dataset shape: {clinical_cancer_data.shape}", background="#121212", foreground='white', font=('Ubuntu 10'))
data_cleaning_label4.grid(row=3, column=0, padx=10, pady=6, sticky="w")

number_of_features_label = tk.Label(reading_data_label_frame, justify=tk.LEFT, text=f"Number of features ( X ) in the dataset = {len(X.columns) + 1}\n\nNumber of features ( y ) fed in the algorithm = {len(X.columns)}", background="#121212", foreground='white', font=('Ubuntu 10'))
number_of_features_label.grid(row=1, column=0, padx=10, pady=6, sticky="w")
# GUI




# inputs:
#    X: pandas.DataFrame, features
#    y: pandas.Series, target variable
#    K: number of features to select

# compute F-statistics and initialize correlation matrix
# F = pd.Series(f_regression(X, y)[0], index = X.columns)
# corr = pd.DataFrame(.00001, index = X.columns, columns = X.columns)

# # initialize list of selected features and list of excluded features
# selected = []
# not_selected = X.columns.to_list()

# # repeat K times
# for i in range(K):
  
#     # compute (absolute) correlations between the last selected feature and all the (currently) excluded features
#     if i > 0:
#         last_selected = selected[-1]
#         corr.loc[not_selected, last_selected] = X[not_selected].corrwith(X[last_selected]).abs().clip(.00001)
        
#     # compute FCQ score for all the (currently) excluded features (this is Formula 2)
#     score = F.loc[not_selected] / corr.loc[not_selected, selected].mean(axis = 1).fillna(.00001)
    
#     # find best feature, add it to selected and remove it from not_selected
#     best = score.index[score.argmax()]
#     selected.append(best)
#     not_selected.remove(best)

# print(selected)



##########################################
# GUI
mrmr_frame = tk.Frame(root, background="#121212")
mrmr_frame.grid(row=1, column=0)

selected_features_label_frame = tk.LabelFrame(mrmr_frame, text=' Selected features ', font=('Ubuntu 10 bold'), background="#121212", foreground= "#39ff14")
selected_features_label_frame.grid(row=0, column=0, padx=10, pady=20)
# GUI

from mrmr import mrmr_classif
k = 7
selected_features = mrmr_classif(X, y, K=k)
# K = number of features we want to keep
# In real applications, one can choose K based on domain knowledge or other constraints, such as model capacity, machine memory or time available.

print(selected_features)

# GUI
mrmr_label1 = tk.Label(selected_features_label_frame, justify=tk.LEFT, text=f"Commencing MRMR algorithm (K = {k}) . . .\n\n\nSelected features are:", background="#121212", foreground='white', font=('Ubuntu 10'))
mrmr_label1.grid(row=0, column=0, padx=10, pady=6, sticky='w')

mrmr_label2 = tk.Label(selected_features_label_frame, text=", ".join(selected_features), background="#121212", foreground='white', font=('Ubuntu 10'))
mrmr_label2.grid(row=1, column=0, padx=10, pady=6, sticky='w')

mrmr_label3 = tk.Label(selected_features_label_frame, text="Commencing genetic algorithm . . .", background="#121212", foreground='white', font=('Ubuntu 10'))
mrmr_label3.grid(row=2, column=0, padx=10, pady=6, sticky='w')
# GUI
####################################################



# gathering/structuring the preprocessed data properly into one dataframe
selected_data = pd.DataFrame()
for feature in selected_features:
	selected_data[f"{feature}"] = X[feature]


# NOT adding a label column because at the time of genetic operators like crossover and mutation, the labels of NEW GENERATED chromosomes will become unknown, instead using KNN
# selected_data['cancer_type_detailed'] = y # labels column
print(selected_data.tail(), selected_data.shape)


# keeping the last 100 samples from the dataset for testing-phase
# testing_data = selected_data.loc[-100:]
testing_data = selected_data.tail(50)
print(testing_data.head())



# GENETIC ALGORITHM IMPLEMENTATION
# initial solutions are considered as the rows in the original preprocessed dataset
generation = selected_data.loc[:selected_data.shape[0] - 50].copy(deep=True) # copying data so that it doesn't cause changes in the originally preprocessed-data
print(generation.head(), generation.tail(), generation.shape)




def mahalanobis_fitness_score(x, data, dimension=1):
	mu_x = x - np.mean(data)
	cov = np.cov(data.values.T)
	cov_inverse = np.linalg.inv(cov)

	if dimension == 1:
		left = np.dot(mu_x, cov_inverse)
		mahalanobis_distance = np.dot(left, mu_x.T)
		return mahalanobis_distance.diagonal()

	if dimension == 0:
		from scipy.spatial import distance
		return (distance.mahalanobis(x, np.mean(data), cov_inverse)**2) # because their formula starts from sqrt(...)

generation['mahalanobis_fitness'] = mahalanobis_fitness_score(generation, generation[generation.columns])
print(generation.head(), generation.tail())



# sorting solutions based on fitness score in descending order (max score first)
generation.sort_values(by='mahalanobis_fitness', ascending=True, ignore_index=True, inplace=True)
print(generation.head(50), generation.tail())

# GUI
generation_frame = tk.Frame(root, background="#121212")
generation_frame.grid(row=2, column=0)

generation_label_frame1 = tk.LabelFrame(generation_frame, text=' Initial generation ', font=('Ubuntu 10 bold'), background="#121212", foreground= "#39ff14")
generation_label_frame1.grid(row=0, column=0, padx=10, pady=6)

generation_label1 = tk.Label(generation_label_frame1, text=f"{generation.head()}", background="#121212", foreground='white', font=('Ubuntu 10'))
generation_label1.grid(row=0, column=0, padx=10, pady=6, sticky='w')

generation_label2 = tk.Label(generation_label_frame1, text=f"{generation.tail()}", background="#121212", foreground='white', font=('Ubuntu 10'))
generation_label2.grid(row=1, column=0, padx=10, pady=6, sticky='w')
# GUI


# here the worst solutions (that should be replaced with new generation better solution) are the ones with the highest Mahalanobis distance because they are much further than the distribution meaning they are not much of a cancer

def sp_crossover_op(x1, x2):
	print(f"Replacing\n {x1} and {x2}\n")
	# single point crossover operation
	print(f"Performing X'over for columns: cohort, tumor_size")
	# x1.loc[:, ['cohort', 'tumor_size']], x2.loc[:, ['cohort', 'tumor_size']] = x2.loc[:, ['cohort', 'tumor_size']], x1.loc[:, ['cohort', 'tumor_size']]
	temp_x1, temp_x2 = x1.loc[:, ['neoplasm_histologic_grade', 'cohort', 'tumor_size']].copy(), x2.loc[:, ['neoplasm_histologic_grade', 'cohort', 'tumor_size']].copy()
	x1.loc[:, ['neoplasm_histologic_grade', 'cohort', 'tumor_size']], x2.loc[:, ['neoplasm_histologic_grade', 'cohort', 'tumor_size']] = temp_x2.values, temp_x1.values
	print(f"New offsprings {x1}, {x2}\n")
	return x1, x2



def mutation_op(x1):
	print(f"Replacing\n {x1}\n")
	# random resetting mutation operation
	random_idx = random.randint(0, x1.shape[0] - 1) # [0, 5]
	print(f"Performing mutation at index {random_idx}")
	max_val = generation[generation.columns[random_idx]].max() # check, of which column does this random-index belongs to, in the main generation dataframe and then take the max value from that column to use in the random() limit
	x1[x1.columns[random_idx]] = random.randint(0, max_val)
	print(f"New offspring {x1}\n")
	return x1


# example for one dimensional calculation of mahalanobis distance
# aaa = mahalanobis_fitness_score(generation.loc[1075, :'age_at_diagnosis'], generation[generation.columns[:-1]], dimension=0)
# # print(generation.loc[0, :'age_at_diagnosis'].T.values, generation.loc[0, :'age_at_diagnosis'].T.values.reshape(5, 1).shape)
# print(aaa)


gen, threshold = 0, 3.5 # setting a threshold for stopping criterion of genetic iterations | generate as long as the worst mahalanobis distance is 6.0 units
while generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness'] >= threshold:
	print(f"\nGeneration /{gen}")
	operator = random.choice(['crossover', 'mutation'])

	if operator == 'crossover':
		random_idx1 = random.randint(0, 1075)
		random_idx2 = random.randint(0, 1075)
		print(f"X'over accepting parent chromosomes, at index [{random_idx1}] & [{random_idx2}]")
		offspring1, offspring2 = sp_crossover_op(generation.loc[[random_idx1]], generation.loc[[random_idx2]])
		print(f"shape = {offspring1.shape}, {offspring2.shape}")

		# calculating fitness scores
		try:
			# score1 = mahalanobis_fitness_score(offspring1.loc[:, :'age_at_diagnosis'], generation[generation.columns[:-1]], dimension=0)
			score1 = mahalanobis_fitness_score(offspring1.loc[:, offspring1.columns != 'mahalanobis_fitness'], generation[generation.columns[:-1]], dimension=0)
			print(f"score1 = {score1}")
			# score2 = mahalanobis_fitness_score(offspring2.loc[:, :'age_at_diagnosis'], generation[generation.columns[:-1]], dimension=0)
			score2 = mahalanobis_fitness_score(offspring2.loc[:, offspring2.columns != 'mahalanobis_fitness'], generation[generation.columns[:-1]], dimension=0)
			print(f"score2 = {score2}")

			if score1 < generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness']: # if score of new offspring is less that the last offspring of the dataset (which has the worst (highest) score)
				offspring1['mahalanobis_fitness'] = score1
				generation.loc[[generation.shape[0] - 1]] = offspring1.values
				# generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness'] = score1
				print(f"Parent at last index is replaced by parent at index [{random_idx1}]\nRe-sorting Generation")
				generation.sort_values(by='mahalanobis_fitness', ascending=True, ignore_index=True, inplace=True)
				print(generation.head(), generation.tail())

			if score2 < generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness']: # if score of new offspring is less that the last offspring of the dataset (which has the worst (highest) score)
				offspring2['mahalanobis_fitness'] = score2
				generation.loc[[generation.shape[0] - 1]] = offspring2.values
				# generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness'] = score2
				print(f"Parent at last index is replaced by parent at index [{random_idx2}]\nRe-sorting Generation")
				generation.sort_values(by='mahalanobis_fitness', ascending=True, ignore_index=True, inplace=True)
				print(generation.head(), generation.tail())
		except:
			print('Singular matrix error. Commensing new iteration...')
			break


	if operator == 'mutation':
		random_idx = random.randint(0, 1075)
		print(f"Mutation accepting parent chromosome, at index [{random_idx}]")
		offspring = mutation_op(generation.loc[[random_idx]])
		print(f"shape = {offspring.shape}")

		# calculating fitness scores
		# score = mahalanobis_fitness_score(offspring.loc[:, :'age_at_diagnosis'], generation[generation.columns[:-1]], dimension=0)
		try:
			score = mahalanobis_fitness_score(offspring.loc[:, offspring.columns != 'mahalanobis_fitness'], generation[generation.columns[:-1]], dimension=0)
			print(f"score = {score}")

			if score < generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness']: # if score of new offspring is less that the last offspring of the dataset (which has the worst (highest) score)
				offspring['mahalanobis_fitness'] = score
				generation.loc[[generation.shape[0] - 1]] = offspring.values
				# generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness'] = score
				print(f"Parent at last index is replaced by parent at index [{random_idx}]\nRe-sorting Generation")
				generation.sort_values(by='mahalanobis_fitness', ascending=True, ignore_index=True, inplace=True)
				print(generation.head(), generation.tail())
		except:
			print('Singular matrix error. Commensing new iteration...')
			break
	gen += 1



# GUI
generation_label_frame2 = tk.LabelFrame(generation_frame, text=f' Final generation# {gen} ', font=('Ubuntu 10 bold'), background="#121212", foreground= "#39ff14")
generation_label_frame2.grid(row=0, column=1, padx=10, pady=6)

generation_label1 = tk.Label(generation_label_frame2, text=f"{generation.head()}", background="#121212", foreground='white', font=('Ubuntu 10'))
generation_label1.grid(row=0, column=0, padx=10, pady=6, sticky='w')

generation_label2 = tk.Label(generation_label_frame2, text=f"{generation.tail()}", background="#121212", foreground='white', font=('Ubuntu 10'))
generation_label2.grid(row=1, column=0, padx=10, pady=6, sticky='w')
# GUI




# for gen in range(5):
# 	print(f"\nGeneration /{gen}")
# 	operator = random.choice(['crossover', 'mutation'])

# 	if operator == 'crossover':
# 		random_idx1 = random.randint(0, 1075)
# 		random_idx2 = random.randint(0, 1075)
# 		print(f"X'over accepting parent chromosomes, at index [{random_idx1}] & [{random_idx2}]")
# 		offspring1, offspring2 = sp_crossover_op(generation.loc[[random_idx1]], generation.loc[[random_idx2]])
# 		print(f"shape = {offspring1.shape}, {offspring2.shape}")

# 		# calculating fitness scores
# 		score1 = mahalanobis_fitness_score(offspring1.loc[:, offspring1.columns != 'mahalanobis_fitness'], generation[generation.columns[:-1]], dimension=0)
# 		print(f"score1 = {score1}")
# 		score2 = mahalanobis_fitness_score(offspring2.loc[:, offspring2.columns != 'mahalanobis_fitness'], generation[generation.columns[:-1]], dimension=0)
# 		print(f"score2 = {score2}")

# 		if score1 < generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness']: # if score of new offspring is less that the last offspring of the dataset (which has the worst (highest) score)
# 			offspring1['mahalanobis_fitness'] = score1
# 			generation.loc[[generation.shape[0] - 1]] = offspring1.values
# 			# generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness'] = score1
# 			print(f"Parent at last index is replaced by parent at index [{random_idx1}]\nRe-sorting Generation")
# 			generation.sort_values(by='mahalanobis_fitness', ascending=True, ignore_index=True, inplace=True)
# 			print(generation.head(), generation.tail())

# 		if score2 < generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness']: # if score of new offspring is less that the last offspring of the dataset (which has the worst (highest) score)
# 			offspring2['mahalanobis_fitness'] = score2
# 			generation.loc[[generation.shape[0] - 1]] = offspring2.values
# 			# generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness'] = score2
# 			print(f"Parent at last index is replaced by parent at index [{random_idx2}]\nRe-sorting Generation")
# 			generation.sort_values(by='mahalanobis_fitness', ascending=True, ignore_index=True, inplace=True)
# 			print(generation.head(), generation.tail())


# 	if operator == 'mutation':
# 		random_idx = random.randint(0, 1075)
# 		print(f"Mutation accepting parent chromosome, at index [{random_idx}]")
# 		offspring = mutation_op(generation.loc[[random_idx]])
# 		print(f"shape = {offspring.shape}")

# 		# calculating fitness scores
# 		score = mahalanobis_fitness_score(offspring.loc[:, offspring.columns != 'mahalanobis_fitness'], generation[generation.columns[:-1]], dimension=0)
# 		print(f"score = {score}")

# 		if score < generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness']: # if score of new offspring is less that the last offspring of the dataset (which has the worst (highest) score)
# 			offspring['mahalanobis_fitness'] = score
# 			generation.loc[[generation.shape[0] - 1]] = offspring.values
# 			# generation.loc[generation.shape[0] - 1, 'mahalanobis_fitness'] = score
# 			print(f"Parent at last index is replaced by parent at index [{random_idx}]\nRe-sorting Generation")
# 			generation.sort_values(by='mahalanobis_fitness', ascending=True, ignore_index=True, inplace=True)
# 			print(generation.head(), generation.tail())





train_on = generation.loc[:, generation.columns != 'mahalanobis_fitness']
y_train = y.loc[:selected_data.shape[0] - 50].copy(deep=True)
print(f"\nshape train_on = {train_on.shape}, shape y_train = {y_train.shape}\n")

# now train a KNN Classifier on the resulting generation
# kmeans = KMeans(n_clusters=6, random_state=0).fit(train_on)

# print(f"\nlabels = {kmeans.labels_}")

print(f"\nPredicting for\n{testing_data}, {testing_data.shape}")
# pred = kmeans.predict(testing_data)
# print(f"\npredictions:\n{pred}")


# train with MLP
clf = MLPClassifier(random_state=1, max_iter=500, verbose=True).fit(train_on, y_train.astype('int'))
pred = clf.predict(testing_data)
print(f"\npredictions:\n{pred}")


# print(f"\ny test = \n{y.head()}\n{y.shape}\n\n{y.tail(100)}\n\n{clinical_cancer_data.tail(100)}")

acc_score = accuracy_score(y.tail(50).tolist(), pred)
print(f"\nAccuracy = {acc_score}")


# self_acc = clf.score(testing_data, y.tail(50).tolist())
# print(f"\nAccuracy = {self_acc}")


# GUI
mlp_frame = tk.Frame(root, background="#121212")
mlp_frame.grid(row=3, column=0)

mlp_label_frame1 = tk.LabelFrame(mlp_frame, text=' Multi Layer Perceptron Classifier ', font=('Ubuntu 10 bold'), background="#121212", foreground= "#39ff14")
mlp_label_frame1.grid(row=0, column=0, padx=10, pady=6)

mlp_label1 = tk.Label(mlp_label_frame1, text=f"Commencing classifier training . . .", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label1.grid(row=0, column=0, padx=10, pady=6, sticky='w')

mlp_label2 = tk.Label(mlp_label_frame1, text="Training data (head):", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label2.grid(row=1, column=0, padx=10, pady=6, sticky='w')

mlp_label3 = tk.Label(mlp_label_frame1, text=f"{train_on.head()}", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label3.grid(row=2, column=0, padx=10, pady=6, sticky='w')

mlp_label4 = tk.Label(mlp_label_frame1, text="Testing data (head):", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label4.grid(row=1, column=1, padx=10, pady=6, sticky='w')

mlp_label5 = tk.Label(mlp_label_frame1, text=f"{testing_data.head()}", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label5.grid(row=2, column=1, padx=10, pady=6, sticky='w')

mlp_label6 = tk.Label(mlp_label_frame1, text="Model predictions (for testing data):", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label6.grid(row=3, column=0, padx=10, pady=6, sticky='w')

mlp_label7 = tk.Label(mlp_label_frame1, justify=tk.LEFT, text=f"{pred}", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label7.grid(row=4, column=0, padx=10, pady=6, sticky='w')

mlp_label8 = tk.Label(mlp_label_frame1, text=f"Accuracy: {acc_score}", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label8.grid(row=3, column=1, padx=10, pady=6, sticky='w')

mlp_label9 = tk.Label(mlp_label_frame1, justify=tk.LEFT, text=f"Interpretation:\n\n100% of testing samples are classified as: \n\n{class_dict[1]}", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label9.grid(row=5, column=0, padx=10, pady=6, sticky='w')

mlp_label10 = tk.Label(mlp_label_frame1, justify=tk.LEFT, text=f"Classes: \n0: {class_dict[0]}\n1: {class_dict[1]}\n2: {class_dict[2]}\n3: {class_dict[3]}\n4: {class_dict[4]}\n5: {class_dict[5]}", background="#121212", foreground='white', font=('Ubuntu 10'))
mlp_label10.grid(row=5, column=1, padx=10, pady=6, sticky='w')
# GUI




root.mainloop()