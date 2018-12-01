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

#(Answer c) - Ind var= Spring/Summer - categorical, Dep var=Ave number of hikers in park - continious. Use a T-test

#(d) Does a student's home state predict their highest degree level?

#(Answer d) - Ind var=Home state - categorical, Dep var=Degree level - categorical. Use T-test


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

#2. You've been given some starter code in class that shows you how to set up ANOVAs and Student's T-Tests in addition to the regression code from the last few weeks. Now, use this code to more deeply explore the simpsons_paradox.csv dataset. Compute new dependent variables that shows the percentage of students admitted and rejected for each row in the CSV. Use those rows to try to understand what significant correlations exist in this data. What factors appear to contribute most heavily to admissions? Do you think the admissions process is biased based on the available data? Why or why not?

#(2:Compute new dependent variables that shows the percentage of students admitted and rejected for each row in the CSV.)

#iv = department,  dv = % admitted, % rejected 
print("Do gender and department correlate with admissions. Is admissions bias?")
evenmoreComplex = 'Department ~ C(Admitted)  + C(Rejected)'
runAnova(rawData, evenmoreComplex)

#(2:Use those rows to try to understand what significant correlations exist in this data. What factors appear to contribute most heavily to admissions? Do you think the admissions process is biased based on the available data? Why or why not?
#(Answer 2) 


#Monday (11.26)
#3. There's a data quality issue hiding in the admissions dataset from Monday. Correct this issue and compare your new results. How are they the same? How do they differ?

#(3 Answer) The data quality issues was that the data tests were only taking into acount the subset of admitted and the subset of rejected seperatly. You would need to combine the total number of both admitted and rejected applicants then use that sum to inspect the acceptance rate. To do so you would add the both sides to get total number of applicants then divide the admitted by total applicant to find the proper acceptance rate. The old results and clean results are the same in..... They differ because in the....

df_inspect= pd.read_csv('simpsons_paradox.csv')
print(df_inspect)

df_inspect['Total Applicants'] = df_inspect['Admitted']+df_inspect['Rejected']
df_inspect['Acceptance Rate'] = df_inspect['Admitted']/df_inspect['Total Applicants']
print(df_inspect)

df_inspect.to_csv('Simpsons_Paradox_Inspected_Cleaned.csv')

#4. The data also represents an example of Simpson's Paradox. Use whatever visualization tools you'd like to illustrate the two possible perspectives. Make sure to include a screenshot of each and explain the perspective shown in each.
