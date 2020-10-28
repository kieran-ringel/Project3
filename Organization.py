import pandas as pd
import random
import math

class Org():
    def __init__(self, file_name, header, class_loc, discrete):
        self.file_name = file_name
        self.header = header
        self.class_loc = class_loc
        self.discrete = discrete

    """Kieran Ringel
    Takes all files and standardized them so they are formatted the same.
    This removes the header included on any file, and
    moves the class to the last column.
    Machines removes the ERP(estimated relative performance from the original article) column.
    Glass removes the index in the first column."""
    def open(self):
        file = open(self.file_name, 'r')        #opens file
        df = pd.DataFrame([line.strip('\n').split(',') for line in file.readlines()])   #splits file by lines and commas

        if self.header != [-1]:             #if user input that the data includes a header
            df = df.drop(self.header, axis=0)   #drop the header
            df = df.reset_index(drop=True)      #reset the axis
        if self.class_loc != -1:            #if the class is not in the last row
            end = df.shape[1] - 1           #moves class to last column
            col = df.pop(0)
            df.insert(end, end + 1, col)
            df.columns = df.columns - 1

        if self.file_name == "Data/glass.data" or self.file_name =="Data/breast-cancer-wisconsin.data": #if file is glass or breast cancer data
            df = df.drop(0, axis=1)             #remove column of index
            df = df.reset_index(drop=True)      #reset axis

        elif self.file_name == "Data/machine.data": #if file is machine data
            df = df.drop(0, axis=1)                 #remove vendor name
            df = df.drop(1, axis=1)                 #remove model name CHECK
            df = df.drop(9, axis=1)                 #remove column with ERP
            df = df.reset_index(drop=True)          #reset acis

        df.columns = range(df.shape[1])
        df.columns = [*df.columns[:-1], "class"]    #give column containing class label 'class'
        newdf = self.missingData(df)
        
        df = self.normalize(newdf)
        return(df)      #returns edited file

    def missingData(self, df):
        print(df)
        classoptions = df["class"].unique()
        
        if (len(classoptions)<20): #this if statement prevents a regression data set from running this function
                                    #only classification data sets have missing values in this case
                                    #classtables holds the dataframes for each class
            
            classtables = [] #creates table for each class, filling it with rows associated with that class
            
            for i in classoptions:
                classtables.append(pd.DataFrame(columns = df.columns))
            df2 = classtables[1]
            for row in df.iterrows():
                for i in range(len(classoptions)):
                    if (row[1][df.shape[1]-1] == classoptions[i]):
                        classtables[i] = classtables[i].append(pd.Series(row[1]),ignore_index=True)
            
            prob = pd.DataFrame(columns = ['class', 'feature', 'chosen', 'highest prob']) #this data frame will be used to store a probabiltity table 
            
            for table in range(len(classtables)):
               
                mini = pd.DataFrame(columns = ['class', 'feature', 'chosen', 'highest prob'])  #mini is the dataframe for each class
                df2 = classtables[table]
                
                whichclass = df2["class"][1]#stores what class the data belong to
                for i in range(df2.shape[1]-2):
                    
                    options = pd.value_counts(df2[i]) #computes for each feature the number of occurrences of each value. Like 146 1s, 45 2s, etc. 
                    totaloptionsum = 0
                    indexes = options.index#indexes store features
                    for o in range(len(options)):#for each feature, adds up all occurrences. Should be the same for each feature in that class
                        totaloptionsum += options[o]
                    chosen = 0 
                    highestprob = 0
                    for p in range(len(options)): #for each feature computes all probabilities, choosing the highest probability and its correlated value
                        probability = options[p]/totaloptionsum
                        if probability>highestprob :
                            highestprob = probability
                            chosen = indexes[p]
                    mini.loc[i] = [whichclass,i, chosen, highestprob] #Adds the chosen class, the feeature, the chosen value, and the highest prob to the dataframe for the class
                prob = prob.append(mini)#adds the dataframe for the class to the probability table
                
            
            for column in range(df.shape[1] - 1):#checks the dataframe for ? values
                for row in range(df.shape[0]): #finds corresponding class and feature in the probability table, gives the missing value the value of the highest probability                                 
                    if df[column][row] == '?':
                        for r in prob.iterrows():
                            if ((column == r[1][1]) and (df["class"][row] == r[1][0])) :
                              
                                df[column][row] = r[1][2]

                  
        return(df)

    def normalize(self, file):
        """Kieran Ringel
        Normalizes all real valued data points using z score normalization"""
        for column in file.iloc[:,:-1]:
            mean= 0
            sd = 0
            if column not in self.discrete:
                for index,row in file.iterrows():
                    mean += float(file[column][index])
                mean /= file.shape[0]                   #calcualates the mean value for each attribute
                for index,row in file.iterrows():
                    sd += (float(file[column][index]) - mean) ** 2
                sd /= file.shape[0]
                sd = math.sqrt(sd)                      #calculated the standard deviation for each attribute
                for index, row in file.iterrows():
                    if sd == 0:
                        file[column][index] = mean      #gets rid of issue of sd = 0
                    else:
                        file[column][index] = (float(file[column][index]) - mean) / sd  #changed value in file to standardized value
        return(file)
