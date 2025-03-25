from django.shortcuts import render,redirect
from django.contrib.auth.models import User,auth
from django.contrib import messages
from .models import MentalStress

# Create your views here.
def index(request):
    return render(request,"index.html")

def register(request):
    if request.method=="POST":
        fname=request.POST['fname']
        lname=request.POST['lname']
        uname=request.POST['uname']
        email=request.POST['email']
        psw=request.POST['psw']
        psw1=request.POST['psw1']
        if psw==psw1:
            if User.objects.filter(username=uname).exists():
                messages.info(request,"Username Exists")
                return render(request,"register.html")
            elif User.objects.filter(email=email).exists():
                messages.info(request,"Email Exists")
                return render(request,"register.html")
            else:
                user=User.objects.create_user(first_name=fname,
                last_name=lname,username=uname,email=email,
                password=psw)
                return redirect('login')
        else:
            messages.info(request,"Password not matching")
            return render(request,"register.html")
    return render(request,"register.html")

def login(request):
    if request.method=="POST":
        uname=request.POST['uname']
        psw=request.POST['psw']
        user=auth.authenticate(username=uname,password=psw)
        if user is not None:
            auth.login(request,user)
            return redirect('data')
        else:
            messages.info(request,"Invalid Credentials")
            return render(request,"login.html")
    return render(request,"login.html")


def data(request):
    if request.method=="POST":
        anxiety=int(request.POST['anxiety_level'])
        self_esteem=int(request.POST['self'])
        mental_health=int(request.POST['mental_health'])
        depression=int(request.POST['depression'])
        headache=int(request.POST['headache'])
        bp=int(request.POST['bp'])
        sleep=int(request.POST['sleep'])
        breathe=int(request.POST['breathe'])
        noise=int(request.POST['noise'])
        living=int(request.POST['living'])
        safety=int(request.POST['safety'])
        basic=int(request.POST['basic'])
        academic=int(request.POST['academic'])
        study=int(request.POST['study'])
        teacher=int(request.POST['teacher'])
        future=int(request.POST['future'])
        social=int(request.POST['social'])
        peer=int(request.POST['peer'])
        extra=int(request.POST['extra'])
        bullying=int(request.POST['bullying'])
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, GridSearchCV
        from sklearn.metrics import classification_report, confusion_matrix
        myd = pd.read_csv(r"static/datasets/StressLevelDataset.csv")

        # Display the first 5 rows of the data
        print(myd.head())
        print(myd.isnull().sum())
        plt.figure(figsize=(12, 20)) 

        for i, column in enumerate(myd.columns, 1):
            plt.subplot(7, 3, i)
            sns.countplot(x=column, data=myd, palette='Blues_d')
            plt.xticks(rotation=45)
            plt.title(column)

        plt.tight_layout()
        plt.show()
        plt.figure(figsize= (12,8))
        sns.boxplot(data=myd, orient='h', color='silver')
        plt.title('Boxplot of all columns')
        plt.show()
        # Percentage of students with mental health history
        mental_health_history = myd['mental_health_history'].value_counts(normalize=True) * 100
        print('Percentage of students with mental health history:', mental_health_history[1])
        # Correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(myd.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()
        # Correlation heat map only stress level 
        correlation = myd.corr()
        correlation_stress = correlation['stress_level'].sort_values(ascending=False)
        correlation_stress = correlation_stress.drop('stress_level')
        plt.figure(figsize=(8, 6))
        sns.heatmap(correlation_stress.to_frame(), annot=True, cmap='coolwarm')
        plt.title('Correlation of Features with Stress Level')
        plt.show()

        # Average stress level
        average_stress_level = myd['stress_level'].mean()
        print('Average Stress Level:', average_stress_level)

        # Percentage breakdown of stress levels pie chart
        myd['stress_level'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
        plt.title('Percentage Breakdown of Stress Levels')
        plt.ylabel('')
        plt.show() 

        # Split the data into features and target
        X = myd.drop('stress_level', axis=1)
        y = myd['stress_level']

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA

        # Scree plot
        pca = PCA()
        pca.fit(X_scaled)
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, marker='o')
        plt.title('Scree Plot')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance')
        plt.show()

        # 2 components explain most of the variance

        # Fit PCA with 2 components
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Plot PCA representation
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='crest') #cividis
        plt.title('PCA Representation')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()


        # Decision Tree Classifier
        #X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(X, y)
        import numpy as np
        pred=np.array([[anxiety,self_esteem,mental_health,depression,headache,
        bp,sleep,breathe,noise,living,safety,academic,basic,study,teacher,future,social,peer,
        extra,bullying]],dtype=object)
        y_pred = dt.predict(pred)
        print("Prediction: ",y_pred)
        
        mental=MentalStress.objects.create(anxiety_level=anxiety,self_esteem=self_esteem,mental_health_history=mental_health,
        depression=depression,headache=headache,blood_pressure=bp,sleep_quality=sleep,
        breathing_problem=breathe,noise_level=noise,living_conditions=living,
        safety=safety,academic_performance=academic,basic_needs=basic,
        teacher_student_relationship=teacher,future_career_concerns=future,
        social_support=social,peer_pressure=peer,extracurricular_activities=extra,
        bullying=bullying,stress_level=y_pred,study_load=study)
        mental.save()

        return render(request,"predict.html",{"anxiety_level":anxiety,
        "self_esteem":self_esteem,"mental_health":mental_health,"depression":depression,"headache":headache,
        "bp":bp,"sleep":sleep,"breathe":breathe,"noise":noise,"living":living,"safety":safety,
        "academic":academic,"basic":basic,"study":study,"teacher":teacher,"future":future,
        "extra":extra,"social":social,"peer":peer,"extra":extra,"bullying":bullying,"prediction":y_pred})
        
    return render(request,"data.html")

def predict(request):
    return render(request,"predict.html")

def logout(request):
    auth.logout(request)
    return redirect('/')