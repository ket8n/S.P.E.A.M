from django.shortcuts import render,HttpResponse,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
import os

# Libraries
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
import seaborn as sns
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from SPEAM.settings import BASE_DIR

# Create your views here.
def index(request):
    return render(request,'index.html')



def login(request):
    if request.method =='POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = auth.authenticate(username = username, password = password)
        if user is not None:
            auth.login(request,user)
            User.username = username
            messages.success(request,'success')
            return redirect('/')
        else:
            messages.error(request,'Invalid')

    return render(request,'login.html')

def logout(request):
    auth.logout(request)
    messages.success(request,'logout')
    return redirect('/')

    
# main model
def model_url_k(request):
    return render(request,'analysis.html',km_context)


km_context= ""

def model_k(request):
    
    data = pd.read_csv(os.path.join(BASE_DIR, "static/csv/instagram_reach.csv"), encoding = 'latin1')
    
    # data.info()

     
    stopwords = set(STOPWORDS) 
    stopwords.add('will')

    sns.set()
    plt.style.use('seaborn-whitegrid')

    def WordCloudPlotter(dfColumn):
        colData = data[dfColumn]
        textCloud = ''
        
        #text processing
        # converting colums to a 
        #single line of text
        for mem in colData:
            textCloud = textCloud + str(mem)
        
        # plotting word cloud
        wordcloud = WordCloud(width = 800, height = 800,background_color ='white', 
                            stopwords = stopwords,  min_font_size = 10).generate(textCloud)
        plt.figure(figsize = (8, 8), facecolor = None)
        plt.style.use('seaborn-whitegrid')
        plt.imshow(wordcloud) 
        plt.rcParams.update({'font.size': 25})
        plt.axis("off") 
        plt.title('Word Cloud: ' + str(dfColumn))
        plt.tight_layout(pad = 0) 
    
        plt.savefig("static/images/model_k/"+dfColumn+".png")
        
    
    WordCloudPlotter('Caption')

    WordCloudPlotter('Hashtags')

    data['Time since posted'] = data['Time since posted'].map(lambda a: int(re.sub('hours', '', a)))

    def PlotData(features):
        plt.figure(figsize= (20, 10))    
    #     pltNum = 1
    #     for mem in features:
    #         plt.subplot(1, 2)
        plt.style.use('seaborn-whitegrid')
        plt.grid(True)
        plt.title('Regplot Plot for '+ str(features))
        sns.regplot(data = data, x = features, y = 'Likes' , color = 'green')
    #         pltNum += 1
        
        
    #     for f in features:
    #         plt.savefig(f+".png")
        
        plt.savefig("static/images/model_k/"+features+".png")
       
        
    PlotData('Followers')

    PlotData('Time since posted')

    
    features = np.array(data[['Followers', 'Time since posted']], dtype = 'float32')
    targets = np.array(data['Likes'], dtype = 'float32')
    maxValLikes = max(targets)
    # print('Max value of target is {}'.format(maxValLikes))


    targets = targets/maxValLikes


    xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size = 0.1, random_state = 42)

    stdSc = StandardScaler()
    xTrain = stdSc.fit_transform(xTrain)
    xTest = stdSc.transform(xTest)


    gbr = GradientBoostingRegressor()
    gbr.fit(xTrain, yTrain)

    predictions = gbr.predict(xTest)
    plt.scatter(yTest, predictions)
    plt.style.use('seaborn-whitegrid')
    plt.xlabel('true values')
    plt.ylabel('predicted values')
    plt.title('GradientRegressor')
    plt.plot(np.arange(0,0.4, 0.01), np.arange(0, 0.4, 0.01), color = 'green')
    plt.grid(True)
    plt.savefig("static/images/model_k/GradientRegressor.png")

    def PredictionsWithConstantFollowers(model, followerCount, scaller, maxVal):
        followers = followerCount * np.ones(24)
        hours = np.arange(1, 25)
        
        # defining vector 
        featureVector = np.zeros((24, 2))
        featureVector[:, 0] = followers
        featureVector [:, 1] = hours
        
        # doing scalling
        featureVector = scaller.transform(featureVector)
        predictions = model.predict(featureVector)
        predictions = (maxValLikes * predictions).astype('int')
        
        plt.figure(figsize= (10, 10))
        plt.plot(hours, predictions)
        plt.style.use('seaborn-whitegrid')
        plt.scatter(hours, predictions, color = 'g')
        plt.grid(True)
        plt.xlabel('hours since posted')
        plt.ylabel('Likes')
        plt.title('Likes progression with ' + str(followerCount) +' followers')
        plt.savefig("static/images/model_k/pred_with_follower_count_"+str(followerCount)+".png")
        
    
    PredictionsWithConstantFollowers(gbr, 100, stdSc, maxValLikes)

    # likes progression for 200 followers
    PredictionsWithConstantFollowers(gbr, 200, stdSc, maxValLikes)

    # Like progression for 1000 followers
    PredictionsWithConstantFollowers(gbr, 1000, stdSc, maxValLikes)

    global km_context
    km_context = {
        "maxValLikes" : maxValLikes
    }


    return redirect('/model_url_k/')
    