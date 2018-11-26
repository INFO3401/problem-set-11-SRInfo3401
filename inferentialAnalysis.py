import numpy as np
import pandas as pd
import scipy

import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

#Monday (11.12):
#1. What statistical test would you use for the following scenarios?

#(a) Does a student's current year (e.g., freshman, sophomore, etc.) effect their GPA?

#(Answer a) - Ind var=current year - categorical, Dep var=gpa - continious. Use Single Logistic Regression.

#(b) Has the amount of snowfall in the mountains changed over time?

#(Answer b) - Ind var=Time - continious, Dep var=Snowfall - continious. Use generalized regression.

#(c) Over the last 10 years, have there been more hikers on average in Estes Park in the spring or summer? 

#(Answer c) - Ind var= Spring/Summer - categorical, Dep var=Ave number of hikers in park - continious. Use 

#(d) Does a student's home state predict their highest degree level?

#(Answer d) - Ind var=Home state - categorical, Dep var=Degree level - categorical. Use


# Extract the data. Return both the raw data and dataframe
def generateDataset(filename):
    data = pd.read_csv(filename)
    df = data[0:]
    df = df.dropna()
    return data, df

#Run a t-test
def runTTest(ivA, ivB, dv):
    ttest = scipy.stats.ttest_ind(ivA[dv], ivB[dv])
    print(ttest)
    
#Run ANOVA
def runAnova(data, formula):
    model = ols(formula, data).fit()
    aov_table = sm.stats.anova_lm(model, typ=2)
    print(aov_table)
    
#Run the analysis
rawData, df = generateDataset('simpsons_paradox.csv')

#iv = gender dv = admissions
print("Does gender correlate with admissions?")
men = df[(df['Gender']=='Male')]
women = df[(df['Gender']=='Female')]
runTTest(men, women, 'Admitted')

#iv = department dv = admissions
print("Does department correlate with admissions?")
simpleFormula = 'Admitted ~ C(Department)'
runAnova(rawData, simpleFormula)

#iv = department,  dv = admitted 
print("Do gender and department correlate with admissions")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData, moreComplex)

#iv = department,  dv = % admitted 
print("Do gender and department correlate with admissions. Is admissions bias?")
evenmoreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData, evenmoreComplex)

#iv = department,  dv = % admitted, % rejected 
print("Do gender and department correlate with admissions. Is admissions bias?")
evenmoreComplex = 'Department ~ C(Admitted)  + C(Rejected)'
runAnova(rawData, evenmoreComplex)


#2. You've been given some starter code in class that shows you how to set up ANOVAs and Student's T-Tests in addition to the regression code from the last few weeks. Now, use this code to more deeply explore the simpsons_paradox.csv dataset. Compute new dependent variables that shows the percentage of students admitted and rejected for each row in the CSV. Use those rows to try to understand what significant correlations exist in this data. What factors appear to contribute most heavily to admissions? Do you think the admissions process is biased based on the available data? Why or why not?
