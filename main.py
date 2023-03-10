# Importing necassary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#This line I use inside of jupyter notebook

# %matplotlib inline

# Reading and exploring data a bot
df = pd.read_csv('911.csv')
df.head()

# top 5 zip-codes with most calls from
top5_zip = df['zip'].value_counts().head()
top5_zip

# Top 5 townships
top5_twp = df['twp'].value_counts().head()
top5_twp

# There are 110 unique titles for accidents
unique_titles = df['title'].nunique()
unique_titles

# So here I'm getting reason, whether it was EMS, trafic or fire
df['reasons'] = df['title'].apply(lambda x: x.split(':')[0])

# And how many times each one happend
df['reasons'].value_counts()

# Plotting distribution of accidents by reason
reasons = df['reasons']
countplot = sns.countplot(x=reasons)
plt.show()

#  getting percentage distribution
countplot = sns.countplot(x=reasons)
totals = []
for container in countplot.containers:
    total = sum([h.get_height() for h in container])
    totals.append(total)
for i, container in enumerate(countplot.containers):
    for h, p in zip(container, [h/totals[i]*100 for h in [h.get_height() for h in container]]):
        countplot.annotate(f"{p:.1f}%", xy=(h.get_x()+h.get_width()/2, h.get_height()), ha='center', va='bottom')

# Show the plot
plt.show()

#Creating some additional tables accroding to the time
# Firstly I have to convert timeStamp column from string to datetime format
df['timeStamp'] = pd.to_datetime(df['timeStamp'])
time = df['timeStamp']
df['hour'] = time.apply(lambda x: x.hour)
df['year'] = time.apply(lambda x: x.year)
df['month'] = time.apply(lambda x: x.month)
df['day_of_week'] = time.apply(lambda x: x.dayofweek)

# Mapping numbers and names of days 
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['day_of_week'] = df['day_of_week'].map(dmap)

# Do some plotting
countplot2 = sns.countplot(x=df['day_of_week']) # According to days it looks like at weekend there are less accidnets than through work days.
# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()



month_plot = sns.countplot(x = df['month'],hue=reasons) # It seems like in January there is the hihest nummber of accidents.
# To relocate the legend
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
# Don't know why but we are missing some months

# So i created groupby object by 'month' and plot him

byMonth = df.groupby('month').count()
byMonth['lat'].plot()
plt.show()
byMonth = byMonth.reset_index()

# Here we clearly see distribution by month
lm_plot = sns.lmplot(x='month',y='lat',data=byMonth)
plt.ylabel('Number of accidents')
plt.show()

#Creating additional column so I can get overall distribution for each day
df['Date'] = df['timeStamp'].apply(lambda x: x.date())

# Here I got number of accidents per day
df.groupby('Date').count()['lat']

# And here I plot all data and then reasons one by one
overall_plot = df.groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.show()

EMS_plot = df[df['reasons']== 'EMS'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.show()

FIRE_plot = df[df['reasons']== 'Fire'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.show()

TRAFFIC_plot = df[df['reasons']== 'Traffic'].groupby('Date').count()['lat'].plot()
plt.tight_layout()
plt.show()