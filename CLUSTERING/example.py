
from optimizer import run




# Select optimizers
optimizer=["ABC","DE","WOA","ABC_DE","ABC_WOA","ABC_WOA_DE"]#,"SSA"]

# Select objective function
# "SSE","TWCV","SC","DB","DI"
objectivefunc=["SSE"]#,"SSE"]#,"TWCV"] 

# Select data sets
#"aggregation","aniso","appendicitis","balance","banknote","blobs","Blood","circles","diagnosis_II","ecoli","flame","glass","heart","ionosphere","iris","iris2D","jain","liver","moons","mouse","pathbased","seeds","smiley","sonar","varied","vary-density","vertebral2","vertebral3","wdbc","wine"
dataset_List = ["ionosphere"]#,"Blood","heart"]#,"aggregation"]"iris"
#dataset_List=['aggregation', 'aniso', 'appendicitis', 'balance', 'banknote', 'blobs', 'Blood', 'circles', 'diagnosis_II', 'ecoli', 'flame', 'glass', 'heart', 'ionosphere', 'iris2D', 'iris', 'jain', 'liver', 'moons', 'mouse', 'pathbased', 'seeds', 'smiley', 'sonar', 'varied', 'vary-density', 'vertebral2', 'vertebral3', 'wdbc', 'wine']
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns=1

# Select general parameters for all optimizers (population size, number of iterations) ....
params = {'PopulationSize' : 30, 'Iterations' : 100}

#Choose whether to Export the results in different formats
export_flags = {'Export_avg':True, 'Export_details':True, 'Export_details_labels':True, 'Export_convergence':True, 'Export_boxplot':True}
run(optimizer, objectivefunc, dataset_List, NumOfRuns, params, export_flags)

#example 2
#run(optimizer, objectivefunc, dataset_List, NumOfRuns, params, export_flags, auto_cluster = False, n_clusters = [3,7], labels_exist = True, metric='cityblock')
