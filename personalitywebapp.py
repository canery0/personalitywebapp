import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import altair as alt
from scipy import linalg
from PIL import Image

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import accuracy_score

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.feature_selection import SelectKBest, chi2

q=["index","You regularly make new friends",
  "You spend a lot of your free time exploring various random topics that pique your interest",
"Seeing other people cry can easily make you feel like you want to cry too",
"You often make a backup plan for a backup plan",
"You usually stay calm, even under a lot of pressure",
"At social events, you rarely try to introduce yourself to new people and mostly talk to the ones you already know",
"You prefer to completely finish one project before starting another",
"You are very sentimental",
"You like to use organizing tools like schedules and lists",
"Even a small mistake can cause you to doubt your overall abilities and knowledge",
"You feel comfortable just walking up to someone you find interesting and striking up a conversation",
"You are not too interested in discussing various interpretations and analyses of creative works",
"You are more inclined to follow your head than your heart",
"You usually prefer just doing what you feel like at any given moment instead of planning a particular daily routine",
"You rarely worry about whether you make a good impression on people you meet",
"You enjoy participating in group activities",
"You like books and movies that make you come up with your own interpretation of the ending",
"Your happiness comes more from helping others accomplish things than your own accomplishments",
"You are interested in so many things that you find it difficult to choose what to try next",
"You are prone to worrying that things will take a turn for the worse",
"You avoid leadership roles in group settings",
"You are definitely not an artistic type of person",
"You think the world would be a better place if people relied more on rationality and less on their feelings",
"You prefer to do your chores before allowing yourself to relax",
"You enjoy watching people argue",
"You tend to avoid drawing attention to yourself",
"Your mood can change very quickly",
"You lose patience with people who are not as efficient as you",
"You often end up doing things at the last possible moment",
"You have always been fascinated by the question of what, if anything, happens after death.",
"You usually prefer to be around others rather than on your own",
"You become bored or lose interest when the discussion gets highly theoretical",
"You find it easy to empathize with a person whose experiences are very different from yours",
"You usually postpone finalizing decisions for as long as possible",
"You rarely second-guess the choices that you have made",
"After a long and exhausting week, a lively social event is just what you need.",
"You enjoy going to art museums",
"You often have a hard time understanding other people’s feelings.",
"You like to have a to-do list for each day.",
"You rarely feel insecure.",
"You avoid making phone calls.",
"You often spend a lot of time trying to understand views that are very different from your own.",
"In your social circle, you are often the one who contacts your friends and initiates activities.",
"If your plans are interrupted, your top priority is to get back on track as soon as possible.",
"You are still bothered by mistakes that you made a long time ago.",
"You rarely contemplate the reasons for human existence or the meaning of life.",
"Your emotions control you more than you control them.",
"You take great care not to make people look bad, even when it is completely their fault.",
"Your personal work style is closer to spontaneous bursts of energy than organized and consistent efforts.",
"When someone thinks highly of you, you wonder how long it will take them to feel disappointed in you.",
"You would love a job that requires you to work alone most of the time.",
"You believe that pondering abstract philosophical questions is a waste of time.",
"You feel more drawn to places with busy, bustling atmospheres than quiet, intimate places.",
"You know at first glance how someone is feeling.",
"You often feel overwhelmed.",
"You complete things methodically without skipping over any steps.",
"You are very intrigued by things labeled as controversial.",
"You would pass along a good opportunity if you thought someone else needed it more.",
"You struggle with deadlines.",
"You feel confident that things will work out for you.","Personality"]

data=pd.read_csv('16Pers.csv')
names = ['Q'+str(i) for i in range(61)]
names.append('Type')
data.to_csv("16Pers.csv", header=names, index=False)
data=pd.read_csv('16Pers.csv')
datdum = data.drop("Q0",axis=1)
data = data.drop("Q0",axis=1)
quesdict=dict(zip(names,q))
#Target  Groups
targetnames=data["Type"].unique()
tarsplit=[]
for tar in targetnames:
    tarsplit.append(list(tar))

#  Extrating Character Traits From target groups
tarsplit=np.array(tarsplit)
traits=[]
for i in range(tarsplit.shape[1]):
     traits.append(list(np.unique(tarsplit[:,i])))
        
data["NumType"]=data["Type"].replace(dict(zip(targetnames,range(len(targetnames)))))
# data['Energy'] = data['Type'].apply(lambda x: x[0])
data['Energy'] = data['Type'].apply(lambda x: 0 if (x[0] == traits[0][0]) else 1)
data['Process Information'] = data['Type'].apply(lambda x: 0 if(x[1]== traits[1][0]) else 1)
data['Decision'] = data['Type'].apply(lambda x: 0 if( x[2]==traits[2][0]) else 1)
data['Approch'] = data['Type'].apply(lambda x: 0 if( x[3]==traits[3][0]) else 1)

x= data.iloc[:,:60]
y=data.loc[:,"NumType"]

r1=st.sidebar.radio("Contents", ("Introduction","Classification Models","Try it yourself"))

if r1 == "Introduction":
    im = Image.open('persimage.png')
    st.image(im)
    '''
    ### The Data:
    Myers and Briggs created their personality typology to help people discover their own strengths and gain a better understanding of how people are different.
    The Data used in the project contains a questainaire , whose answers are values are integers from -2 to 1.
    The data is used to predict the personality type of the test-taker.


    '''
    r2=st.radio("Select",("View data", "Personality type"))
    if r2== "View data":
        st.dataframe(datdum.head(8))
    if r2== "Personality type":
        im = Image.open('type.png')
        st.image(im)
        
    
if r1  == "Classification Models":
    '''
    #### Select Training Size

    '''
    
    frac=st.slider("",0.0,1.0,0.33)
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=frac)

    clfnames = ["KNN",
        "Random Forest",
        "Decision Tree",
        "MLP Classifier",
        "Extra Trees"

    ]
    classif=st.selectbox("Classification Model", clfnames)
    if classif == "KNN":
        classifier=KNeighborsClassifier(4)
    if classif == "Random Forest":
        classifier=RandomForestClassifier(max_depth=5, n_estimators=64)
    if classif == "Decision Tree":
        classifier=DecisionTreeClassifier(max_depth=16)
    if classif == "MLP Classifier":
        classifier=MLPClassifier(alpha=1, max_iter=1000)
    if classif == "Extra Trees":
        classifier=ExtraTreesClassifier(n_estimators=100, max_depth=None,min_samples_split=10, random_state=0)


    
    classifier.fit(x_train, y_train)
    predicted=classifier.predict(x_test)

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    f=disp.figure_.suptitle("Confusion Matrix")
    st.write((f.figure))
    
    '''
    ##### The Accuracy and the Cohen Kappa Score
    '''
    r3=st.radio(" ",("Accuracy","Cohen Kappa Score"))
    if r3=="Cohen Kappa Score":
        st.write(cohen_kappa_score(y_test, predicted))
    if r3=="Accuracy":
        st.write(accuracy_score(y_test, predicted))

if r1  == "Try it yourself":
    frac=0.33
    '''
    ### QUIZ
    ##### Select The number of questions, you'd like to answer. 
    (The more you answer the better the accuracy)
    '''
    qnum=st.slider(" ",10,60,10)
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=frac)
    x_train=x_train.apply(lambda x: x+3)
    
    selector=SelectKBest(chi2, k=qnum)
    x_reduced=selector.fit_transform(x_train, y_train)
    col= selector.get_support(indices=True)
    f_df = x.iloc[:,col]
    y_new= y
    uservec=[]
    count=0
    '''
    ##### Select a value from 0 to 5 that best represents your answer!
    (0 being least likely and 1 being the most.)
    '''
    for column in f_df:
        st.write(quesdict[column])
        val=st.select_slider("Select",options=[0,1,2,3,4,5],key=count)
        uservec.append(val)
        count+=1
    x_t1,x_tes,y_t1,y_tes= train_test_split(f_df,y,test_size=frac)
    classifier=KNeighborsClassifier(4)
    classifier.fit(x_t1, y_t1)
    predicted=classifier.predict(x_tes) 
    '''
    ######  Accuracy for this quiz is given below, to increase the accuracy, attempt more questions.
    '''
    st.write(accuracy_score(y_tes, predicted)) 
    uservec=np.array(uservec)
    uservec=uservec.reshape(1,qnum)
    p2=classifier.predict(uservec) 
    tardict=dict(zip(range(len(targetnames)),targetnames))
    if st.button('Predict'):
        st.write(tardict[int(p2)])
        
    



    
    










