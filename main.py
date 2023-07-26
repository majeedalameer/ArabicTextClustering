
import csv
import numpy
import time
import minisom
import selector as slctr

# Select optimizers
GWO = True




optimizer=[GWO] 


datasets=["MSA"] # First We need to specify the data here In this case

NumOfRuns=1 # Select number of repetitions for each experiment in order to select the best results.

PopulationSize = 5

Iterations= 1

epochs= 7000

# True For Exporting the results, And false to prevent exporting results 
Export=True


#ExportToFile="YourResultsAreHere.csv"

#Automaticly generated file name by date and time


ExportToFile="Output.csv"




#In order to check the algorithm if it works at least once
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader=[]
Algo = "minisom"
for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))

trainDataset="MSATrain_small.csv"
testDataset="MSATest_small.csv"


for j in range (0, len(datasets)):        # specfiy the number of the datasets
    for i in range (0, len(optimizer)):
    
        if((optimizer[i]==True)): # start experiment if an optimizer and an objective function is selected
            for k in range (0,NumOfRuns):
                
                func_details=["costNN",0,1]
                trainDataset=datasets[j]+"Train.csv"
                testDataset=datasets[j]+"Test.csv"
                x=slctr.selector(i,func_details,PopulationSize,Iterations,trainDataset,testDataset, epochs)
                  
               # if(Export==True):
                #    with open(ExportToFile, 'a',newline='\n') as out:
                 #       writer = csv.writer(out,delimiter=',')
                  #      if (Flag==False): # just one time to write the header of the CSV file
                   #         header= numpy.concatenate([["Optimizer","Dataset","objfname","Experiment","startTime","EndTime","ExecutionTime","trainAcc", "trainTP","trainFN","trainFP","trainTN", "testAcc", "testTP","testFN","testFP","testTN"],CnvgHeader])
                    #        writer.writerow(header)
                     #   a=numpy.concatenate([[x.optimizer,datasets[j],Algo,k+1,x.startTime,x.endTime,x.executionTime,x.f1_score, x.precision_score,x.recall_score,x.accuracy_score],x.convergence])
                      #  writer.writerow(a)
                   # out.close()
                Flag=True # at least one experiment
                
if (Flag==False): # Faild to run at least one experiment
    print("The cost function or the optimizer is not selected, Please verify") 
        
        
