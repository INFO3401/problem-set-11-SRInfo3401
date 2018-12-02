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
    
#Run new percentage clean
def cleanRate(df):
    df['Total_Applicants'] = df['Admitted'] + df['Rejected']
    df['Acceptance_Rate'] = df['Admitted']/df['Total_Applicants']
    df['Rejection_Rate'] = df['Rejected']/df['Total_Applicants']
    return df
    
#Run the analysis
rawData, df = generateDataset('simpsons_paradox.csv')
sp_clean = cleanRate(df)

#iv = gender dv = admissions(reuse this with new call for #2)
print("Does gender correlate with admissions?")
men = df[(df['Gender']=='Male')]
women = df[(df['Gender']=='Female')]
runTTest(men, women, 'Admitted')

#(2 Tests Code)
#(Edited score results)
#iv = department,  dv = % admitted, % rejected
print("Does gender correlate with admissions? Is admissions bias?")
men = sp_clean[(sp_clean['Gender']=='Male')]
women = sp_clean[(sp_clean['Gender']=='Female')]
runTTest(men, women, 'Total_Applicants')

#Does gender correlate with admissions?
#Ttest_indResult(statistic=5.332277756733584, pvalue=0.001774285663548817)

#Does gender correlate with admissions? Is admissions bias?
#Ttest_indResult(statistic=2.323714716166561, pvalue=0.05914738704979229)

#(Edited score results)
#iv = department dv = admissions
print("Does department correlate with admissions?")
simpleFormula = 'Admitted ~ C(Department)'
runAnova(rawData, simpleFormula)

#Does department correlate with admissions?
#                  sum_sq   df         F    PR(>F)
#C(Department)   92266.75  5.0  0.737438  0.622205
#Residual       150141.50  6.0       NaN       NaN

#(Edited score results)
#iv = department,  dv = admitted 
print("Do gender and department correlate with admissions")
moreComplex = 'Admitted ~ C(Department) + C(Gender)'
runAnova(rawData, moreComplex)

#Do gender and department correlate with admissions
#                      sum_sq   df          F    PR(>F)
#C(Department)   41153.333333  5.0   7.515304  0.036670
#C(Gender)      145760.750000  2.0  66.546025  0.000851
#Residual         4380.750000  4.0        NaN       NaN


#2. You've been given some starter code in class that shows you how to set up ANOVAs and Student's T-Tests in addition to the regression code from the last few weeks. Now, use this code to more deeply explore the simpsons_paradox.csv dataset. Compute new dependent variables that shows the percentage of students admitted and rejected for each row in the CSV. Use those rows to try to understand what significant correlations exist in this data. What factors appear to contribute most heavily to admissions? Do you think the admissions process is biased based on the available data? Why or why not?

#(2:Compute new dependent variables that shows the percentage of students admitted and rejected for each row in the CSV.)
#(Answer 2)
#def cleanRate(df):
    #df['Total_Applicants'] = df['Admitted'] + df['Rejected']
    #df['Acceptance_Rate'] = df['Admitted']/df['Total_Applicants']
    #df['Rejection_Rate'] = df['Rejected']/df['Total_Applicants']
    #return df

#(2:Use those rows to try to understand what significant correlations exist in this data. What factors appear to contribute most heavily to admissions? Do you think the admissions process is biased based on the available data? Why or why not?

#(Answer 2) Based on the results using the origional dataset values to compute the ttest and anova scores you can support that there is a indication that bias is not present to admissions. When comparing the old and new test results in the question does gender correlate with admissions it appears that the gender aspect contributes most heavily to admissions processing. This is supported in the data but we also must check with the new p vlaue to verify, which is .059. A p-value over .05 is cutoff from being considered significant. Having this .059 p-value tells me we can accept the null hypothesis for these variables and there is a lack of bias in admission processing based on the tested independent variables. Further evidence is shown in the processing of new percentage data in the begining of question 2. In the old data and newly edited percentage data it was clear by the new percentage accepted and rejected columns that ther is lack of bias. I can support this because there was no pattern between male or female applicants and the departments acceptance or rejection percentage rates, they ranged all over. Overall there was no clear indicatoins that any or either specific department is bias in acceptance rates based on a particular sex.

#Monday (11.26)
#3. There's a data quality issue hiding in the admissions dataset from Monday. Correct this issue and compare your new results. How are they the same? How do they differ?

#(3 Answer)The data quality issue was that the data tests were only taking into acount the subset of admitted and the subset of rejected seperatly. You would need to combine the total number of both admitted and rejected applicants then use that sum to inspect the acceptance rate. To do so you would add both the sides to get a total number of applicants then divide the admitted by total applicants to find the proper acceptance/rejection rate. The old results and clean results differ greatly. The old results identified the independent variables as having a high significance to admission processing, this is shown in the original scores. They also differ greatly after the data was edited because now in the new test results the independent variables lost their significince to the variation in admisison processing which is shown by the new pr scores.

#(3 Code translated to .py use(Added to #2/def at top))
#(Python Code Test/CSV Check)
#df_inspect= pd.read_csv('simpsons_paradox.csv')
#df_inspect['Total_Applicants'] = df_inspect['Admitted'] + df_inspect['Rejected']
#df_inspect['Acceptance_Rate'] = df_inspect['Admitted']/df_inspect['Total_Applicants']
#df_inspect['Rejection_Rate'] = df_inspect['Rejected']/df_inspect['Total_Applicants']
#print(df_inspect)
#df_inspect.to_csv('Simpsons_Paradox_Inspected_Cleaned.csv')

#4.(Done in jupyter notebook/screenshots) The data also represents an example of Simpson's Paradox. Use whatever visualization tools you'd like to illustrate the two possible perspectives. Make sure to include a screenshot of each and explain the perspective shown in each.

#(4 Answer PARAGRAPHS ARE IN NOTEBOOK UNDER VISUALIZATIONS/PLOTS) 

