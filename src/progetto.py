import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import linear_model, svm
import warnings

#1) CARICAMENTO DATASET

data = pd.read_csv('./Data/loan_approval_dataset.csv')

#***************************************************************

# 2) PRE-PROCESSING

data.drop('loan_id',axis=1,inplace=True)

#Controllo presenza di valori nulli 
print(f"Valori NaN presenti nel dataset :\n{data.isnull().sum()}\n\n")

#Conversione dei valori della colonna target
#data[" loan_status"] = data[" loan_status"].replace({' Approved': 1, ' Rejected': 0})

#Sostituzione dei valori privi di significato nella feature residential_assets_value
data[" residential_assets_value"] = data[" residential_assets_value"].replace({-100000: 0})

#Trasformo le colonne oggetto in colonne categoriche
data[" education"] = data[" education"].astype("category")
data[" self_employed"] = data[" self_employed"].astype("category")
data[" loan_status"] = data[" loan_status"].astype("category")

#Stampa delle informazioni relative al dataset
print(f"{data.head()}\n\n")
print(f"{data.info()}\n\n")

data_col = data[" bank_asset_value"]

#*********************************************************************************************

# 3) EXPLORATORY DATA ANALYSIS (EDA)
"""
#Visualizzo la differenza che c'è tra la media delle entrate annue di un laureato e di un non laureato
data_graduate = data[data[" education"]==" Graduate"]
data_not_graduate = data[data[" education"]==" Not Graduate"]

media_income_graduate = np.mean(data_graduate[" income_annum"])
media_income_not_graduate = np.mean(data_not_graduate[" income_annum"])

plt.bar(['Graduate','Not Graduate'], [media_income_graduate, media_income_not_graduate], color=['blue', 'orange'],width=0.5)
plt.ylabel('income_annum')
plt.ticklabel_format(style='plain', axis='y')

plt.show()

print(f"Media delle entrate annue dei non laureati : {media_income_not_graduate.round(2)}")
print(f"Media delle entrate annue dei laureati : {media_income_graduate.round(2)}\n\n")

"""
warnings.filterwarnings("ignore", message="use_inf_as_na option is deprecated")
#Controllo quanto è importante avere un cibil_score alto per ricevere il presito
random_sample = data.sample(1000)

sns.pointplot(data=random_sample, x=random_sample.index, y=" cibil_score", hue=" loan_status",linestyles="",scale=0.7)
plt.title('Pointplot')
plt.xlabel('indice')
plt.ylabel('Cibil_score')
plt.show()

#Chiedere al prof se va bene plottare grafici che hanno come variabile categorica la feature target?


#Visualizzo le distribuzioni dei vari tipi di asset
plt.figure(figsize=(10, 6))
sns.histplot(data[" residential_assets_value"], bins=50)
plt.title('Distribuzione dei valori della colonna')
plt.xlabel('residential_asset_value')
plt.ylabel('Frequenza')

plt.ticklabel_format(style='plain', axis='x')

plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data[" commercial_assets_value"], bins=50)
plt.title('Distribuzione dei valori della colonna')
plt.xlabel('commercial_asset_value')
plt.ylabel('Frequenza')

plt.ticklabel_format(style='plain', axis='x')

plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(data[" luxury_assets_value"], bins=50)
plt.title('Distribuzione dei valori della colonna')
plt.xlabel('luxury_asset_value')
plt.ylabel('Frequenza')

plt.ticklabel_format(style='plain', axis='x')

plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(data[" bank_asset_value"], bins=50)
plt.title('Distribuzione dei valori della colonna')
plt.xlabel('bank_asset_value')
plt.ylabel('Frequenza')

plt.ticklabel_format(style='plain', axis='x')

plt.show()

#Analizzo la relazione che c'è tra l'ammontare del prestito e la durata di quest'ultimo, evidenziando se è stato o meno accettato
plt.figure(figsize=(10, 10))

sns.scatterplot(data=data,
                x=" income_annum",
                y=" loan_amount",
                hue=" self_employed") 
plt.show()


sns.scatterplot(data,
                x=" income_annum",
                y=" residential_assets_value") 
plt.show()


sns.scatterplot(data,
                x=" income_annum",
                y=" commercial_assets_value") 
plt.show()


sns.scatterplot(data,
                x=" income_annum",
                y=" luxury_assets_value",
                hue=" self_employed") 
plt.show()



sns.scatterplot(data,
                x=" income_annum",
                y=" bank_asset_value") 
plt.show()



sns.scatterplot(data=random_sample,
                y=" loan_amount",
                x=" loan_term",
                hue=" loan_status") 

plt.ticklabel_format(style='plain', axis='y')

plt.show()



#Estrazione parte numerica del dataset
num_data =  data[[" no_of_dependents"," income_annum",
                   " loan_amount"," loan_term",
                   " cibil_score"," residential_assets_value",
                   " commercial_assets_value"," luxury_assets_value",
                   " bank_asset_value"]]


print(f"Stampa del dataset composto da attributi numerici : \n{num_data.info()}\n")


#Creazione della matrice di correlazione con relativa stampa
C = num_data.corr()

plt.figure(figsize=(10, 8))
plt.matshow(num_data.corr(), vmin=-1, vmax=1)
plt.xticks(np.arange(0, num_data.shape[1]), num_data.columns, rotation=90)
plt.yticks(np.arange(0, num_data.shape[1]), num_data.columns)
plt.title("Visualizzazione matrice di correlazione")
plt.colorbar()
plt.show()


#*****************************************************************************************************

#4) SPLITTING 

#Split (train - test)
from sklearn import model_selection

#np.random.seed(42)


data_final = data[[" no_of_dependents"," income_annum",
                   " loan_amount"," loan_term",
                   " cibil_score"," residential_assets_value",
                   " commercial_assets_value"," luxury_assets_value",
                   " bank_asset_value"," loan_status"]]


data_train, data_test = model_selection.train_test_split(data_final, train_size=0.8)

data_train2, data_val = model_selection.train_test_split(data_train, train_size=0.65)

#******************************************************************************************************

#5) REGRESSIONE 


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

#Traccio retta di regressione tra la feature di input "income_annum" e quella di output "loan_amount"
#In questo caso applichiamo una regressione univariata dato che la dimensione dell'input è 1
X_train = data_train[" income_annum"].values.reshape(-1,1)
y_train = data_train[" loan_amount"].values.reshape(-1,1)

modello = LinearRegression()
modello.fit(X_train,y_train)

y_pred_train = modello.predict(X_train)


plt.scatter(X_train, y_train, color='skyblue')
plt.plot(X_train, y_pred_train, color='red')
plt.title('Regressione tra entrate annue e ammontare prestito')
plt.xlabel('income_annum')
plt.ylabel('loan_amount')
plt.show()

X_test = data_test[" income_annum"].values.reshape(-1,1)
y_test = data_test[" loan_amount"].values.reshape(-1,1)

y_pred_test = modello.predict(X_test)

print('PRIMA REGRESSIONE LINEARE --------------------')
#Calcolo del mean squared error
mse = mean_squared_error(y_test,y_pred_test)

# Calculate R square vale
rsq = r2_score(y_test,y_pred_test)

print('mean squared error per entrate annue e importo in presito :',mse,'\n')
print('r square per entrate annue ed importo in presito:',rsq,'\n')

#Analisi dei residui
residuals = y_test - y_pred_test

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='skyblue')
plt.title('Plot dei Residui')
plt.show()

from scipy.stats import shapiro
 
print(f"Test di Shapiro : {shapiro(residuals)}\n\n")


#Traccio retta di regressione tra la feature di input "income_annum" e quella di output "loan_amount"
X_train = data_train[" loan_amount"].values.reshape(-1,1)
y_train = data_train[" luxury_assets_value"].values.reshape(-1,1)

modello = LinearRegression()
modello.fit(X_train,y_train)

y_pred_train = modello.predict(X_train)

plt.scatter(X_train, y_train, color='lightgreen')
plt.plot(X_train, y_pred_train, color='red')
plt.title('Regressione tra ammontare prestito e beni di lusso')
plt.xlabel('loan_amount')
plt.ylabel('luxury_assets_value')
plt.show()

X_test = data_test[" loan_amount"].values.reshape(-1,1)
y_test = data_test[" luxury_assets_value"].values.reshape(-1,1)

y_pred_test = modello.predict(X_test)


mse = mean_squared_error(y_test,y_pred_test)

# Calculate R square vale
rsq = r2_score(y_test,y_pred_test)

print('SECONDA REGRESSIONE LINEARE --------------')
print('mean squared error per beni di lusso e importo in prestito :',mse,'\n')
print('r square per beni di lusso ed importo in presito:',rsq,'\n')


#Analisi dei residui
residuals = y_test - y_pred_test

plt.figure(figsize=(8, 6))
sns.histplot(residuals, kde=True, color='lightgreen')
plt.title('Plot dei Residui')
plt.show()

 
print(f"Test di Shapiro : {shapiro(residuals)}\n")

#********************************************************************************

# 6) ADDESTRAMENTO DEL MODELLO

X = data_train2[[" no_of_dependents"," income_annum",
                   " loan_amount"," loan_term",
                   " cibil_score"," residential_assets_value",
                   " commercial_assets_value"," luxury_assets_value",
                   " bank_asset_value"]]

y = data_train2[[" loan_status"]]
y = np.ravel(y)



#Addestramento Regressione logistica sui dati del train set
modelLogistic = linear_model.LogisticRegression()
modelLogistic.fit(X, y)

#Non sono riuscito ad addestrare un modello svm.SVC con un kernel "linear"
#model = svm.SVC(kernel="linear", C=1)
#model.fit(X,y)

#estrarre X_val e y_val da data_val
X_val = data_val[[" no_of_dependents"," income_annum",
                   " loan_amount"," loan_term",
                   " cibil_score"," residential_assets_value",
                   " commercial_assets_value"," luxury_assets_value",
                   " bank_asset_value"]]

y_val = data_val[[" loan_status"]]


print("Accuratezza predizione SVM kernel poly sul validation set : \n")
#7) HYPERPARAMTER TUNING kernel 'poly'
accuratezze = np.empty(7)

for d in range(1, 8):
    
    model = svm.SVC(kernel="poly", degree=d)
    model.fit(X,y)

    y_pred = model.predict(X_val)


    ME = np.sum(y_pred != y_val[" loan_status"])

    MR = ME / len(y_pred)
    print(f"MR : {MR}.")

    Acc = 1 - MR
    print(f"Acc grado = {d}: {Acc}.")
    
    accuratezze[d-1] = Acc
    
#Fine hyperparameter tuning per il kernel poly

plt.figure(figsize=(10, 6))
plt.plot(range(1, 8), accuratezze, marker='o', linestyle='-', color='b')
plt.xlabel('Grado del polinomio')
plt.ylabel('Accuratezza')
plt.title('Accuratezza in funzione del grado del polinomio (kernel poly)')
plt.grid(True)
plt.show()

#Estraggo il grado che mi restituisce l'accuratezza massima
maxGradoPoly = np.argmax(accuratezze)
#Estraggo la massima accuratezza
maxElem = accuratezze[maxGradoPoly]
print(f"Accuratezza massima Poly : {maxElem} \nGrado massimo : {maxGradoPoly+1}")

#Salvarsi all'interno del ciclo l'accuratezza e visualizzare con plt.plot il grafico dell'errore
print('\n\n')

print("Accuratezza predizione SVM kernel RBF sul validation set : \n")
#7) HYPERPARAMTER TUNING kernel 'rbf'
accuratezza = []
C_values = [0.1, 1, 10, 100, 1000]

#Ciclo for in cui cambio i valori di C per un modello svm.SVC rbf
for c in C_values:
    
    modelRBF = svm.SVC(kernel='rbf', C=c, gamma = 'scale')
    modelRBF.fit(X,y)

    y_pred = modelRBF.predict(X_val)


    ME = np.sum(y_pred != y_val[" loan_status"])

    MR = ME / len(y_pred)
    print(f"MR : {MR}.")

    Acc = 1 - MR
    print(f"Acc C = {c}: {Acc}.")
    
    accuratezza.append(Acc)


accuratezzeRBF = np.array(accuratezza)

plt.figure(figsize=(10, 6))
plt.plot(C_values, accuratezzeRBF, marker='o', linestyle='-', color='b')
plt.xlabel('Costo del kernel rbf')
plt.ylabel('Accuratezza')
plt.title('Accuratezza in funzione del costo (kernel rbf)')
plt.grid(True)
plt.show()
 

#Trovo l'indice del valore massimo, ovvero il rispettivo Costo
maxC_RBF = C_values[np.argmax(accuratezzeRBF)]

#Accuratezza massima
maxElem = np.max(accuratezzeRBF)
print(f"Accuratezza massima RBF : {maxElem} \nC migliore : {maxC_RBF}")

#**************************************************************************************************

#8)VALUTAZIONE DELLA PERFORMANCE
from sklearn.metrics import  confusion_matrix

#Estraggo gli input e la variabile target dal test set
X_test = data_test[[" no_of_dependents"," income_annum",
                   " loan_amount"," loan_term",
                   " cibil_score"," residential_assets_value",
                   " commercial_assets_value"," luxury_assets_value",
                   " bank_asset_value"]]

y_test = data_test[[" loan_status"]]

#Predico la variabile target con il modello di regressione logistica
y_pred_test = modelLogistic.predict(X_test)

print("\nAccuratezza sul test set della logistic regression:\n")

#Misurare l'errore di misclassificazione
ME = np.sum(y_pred_test != y_test[" loan_status"])#quante volte la predizione è diversa dal reale
print(f"ME : {ME}.")

MR = ME / len(y_pred_test)
print(f"MR : {MR}.")

Acc = 1 - MR
print(f"Acc: {Acc}.")
print('\n\n')

conf_matrix = confusion_matrix(y_test, y_pred_test)
print(f'Matrice di confusione logistic regression :\n{conf_matrix}')


#Addestro il modello con il grado migliore e predico la variabile target
modelPoly = svm.SVC(kernel= "poly", degree=maxGradoPoly)
modelPoly.fit(X,y)

y_pred_test = modelPoly.predict(X_test)

print("\nAccuratezza sul test set del svm kernel poly:\n")

#Misurare l'errore di misclassificazione
ME = np.sum(y_pred_test != y_test[" loan_status"])#quante volte la predizione è diversa dal reale
print(f"ME : {ME}.")

MR = ME / len(y_pred_test)
print(f"MR : {MR}.")

Acc = 1 - MR
print(f"Acc: {Acc}.")
print('\n\n')

conf_matrix = confusion_matrix(y_test, y_pred_test)
print(f'Matrice di confusione kernel poly :\n{conf_matrix}')

#Addestro il modello con kernel = 'rbf' e il migliore costo
modelRBF = svm.SVC(kernel = "rbf", C = maxC_RBF, gamma="scale")
modelRBF.fit(X,y)

#Predizione del modello
y_pred_test = modelRBF.predict(X_test)

print("\nAccuratezza sul test set del svm kernel RBF:\n")

#Misurare l'errore di misclassificazione
ME = np.sum(y_pred_test != y_test[" loan_status"])#quante volte la predizione è diversa dal reale
print(f"ME : {ME}.")

MR = ME / len(y_pred_test)
print(f"MR : {MR}.")

Acc = 1 - MR
print(f"Acc: {Acc}.")
print('\n\n')

conf_matrix = confusion_matrix(y_test, y_pred_test)
print(f'Matrice di confusione kernel rbf :\n{conf_matrix}\n')

#9) STUDIO STATISTICO SUI RISULTATI DELLA VALUTAZIONE

accuratezze = []
MR_array = []
ME_array = []


#Ripeto le fasi di addestramento e testing un numero k di volte per la logistic regression 
for i in range(1,20) :
    random_state = np.random.randint(0, 10000)
    
    data_train, data_test = model_selection.train_test_split(data_final, train_size=0.7,random_state= random_state)
   
    X_train = data_train[[" no_of_dependents"," income_annum",
                       " loan_amount"," loan_term",
                       " cibil_score"," residential_assets_value",
                       " commercial_assets_value"," luxury_assets_value",
                       " bank_asset_value"]]

    y_train = data_train[[" loan_status"]]
    y_train = np.ravel(y_train)
    
   
    X_test = data_test[[" no_of_dependents"," income_annum",
                       " loan_amount"," loan_term",
                       " cibil_score"," residential_assets_value",
                       " commercial_assets_value"," luxury_assets_value",
                       " bank_asset_value"]]

    y_test = data_test[[" loan_status"]]
   
    modelLogisticTest = linear_model.LogisticRegression()
    modelLogisticTest.fit(X_train, y_train)
    
    y_pred_test = modelLogistic.predict(X_test)

    
    ME = np.sum(y_pred_test != y_test[" loan_status"])#quante volte la predizione è diversa dal reale
    ME_array.append(ME)

    MR = ME / len(y_pred_test)
    MR_array.append(MR)

    Acc = 1 - MR
    accuratezze.append(Acc)


plt.figure(figsize=(8, 6))
plt.boxplot(accuratezze)
plt.xlabel('Accuratezza (Acc)')
plt.title('Boxplot dell accuratezza media (Acc) su 15 ripetizioni')
plt.grid(True)
plt.show()


#Calcolo l'intervallo di confidenza
from scipy import stats

mean = np.mean(accuratezze)

# Calcolare la deviazione standard del campione
std_dev = np.std(accuratezze)

# Calcolare l'errore standard della media
n = len(accuratezze)
SE = std_dev / np.sqrt(n)

df = n-1 #gradi di liberta

# Calcolare l'intervallo di confidenza al 95%
alpha = 0.05
livello_di_confidenza = 1-alpha

# Valore critico dalla distribuzione t di Student
t_critico = stats.t.ppf(1 - alpha / 2, df)

# Margine di errore
margine_di_errore = t_critico * (std_dev / np.sqrt(n))

# Calcolo dell'intervallo di confidenza
intervallo_di_confidenza = (mean - margine_di_errore, mean + margine_di_errore)

print(f"Intervallo di confidenza al {livello_di_confidenza * 100}%: {intervallo_di_confidenza}")

# Stampare i risultati
print(f"Media del campione: {mean}")
print(f"Deviazione standard del campione : {std_dev}")
