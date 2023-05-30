import streamlit as st
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from streamlit_option_menu import option_menu
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
plt.rcParams.update({'font.size': 5})

st.set_page_config(layout="wide")
test=pd.read_csv("test.csv")
train=pd.read_csv("train.csv")
button_style = """
        <style>
        .stButton > button {
            color: white;
            background: purple;
            border-radius: 15px;
        }
        </style>
        """
st.markdown(button_style, unsafe_allow_html=True)




selected = option_menu(
        menu_title="Human Activity Recognition Using Device Data",
        options=["Home","Data Analysis","Prediction"],
        default_index=0,
        orientation="horizontal",
        styles={
            "menu-title":{"font-size" : "35px","font-weight" : "bold","color": "green"},
            "container": {"padding": "0!important", "background-color": "white","height": "50px"},
            "icon": {"color": "orange", "font-size": "15px"}, 
            "nav-link": {"font-size": "20px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "purple"},
        }
    )
    
# st.title("Human Activity Recognition Using Smartphone Data")
if selected=="Home":
    st.markdown("  <br>",unsafe_allow_html=True)
    st.write("### **PROJECT   DESCRIPTION**")
    st.markdown("  <br>",unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### About the Project :")
        st.markdown('<div style="text-align: justify;">In this project we classify activities such as WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS,  SITTING, STANDING & LAYING using data collected through various sensors such as accelerometer and gyroscope. Data has been analysed through various plots and dimensionality reduction techniques like PCA and TSNE and is cross validated to provide better results. Algorithms such as Logistic regression, SVM, Decision tree have been used to classify the activities . </div>', unsafe_allow_html=True)
        st.markdown("  <br>",unsafe_allow_html=True)
        st.write("#### Dataset Used :")
        st.markdown('<div style="text-align: justify;">The experiments have been carried out with a group of 30 volunteers within an age bracket of 19-48 years. Each person performed six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING) wearing a smartphone (Samsung Galaxy S II) on the waist. Using its embedded accelerometer and gyroscope, we captured 3-axial linear acceleration and 3-axial angular velocity at a constant rate of 50Hz. </div>', unsafe_allow_html=True)
        st.write("[More about the dataset](https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones)")
    with col2:
        st.markdown("  <br>",unsafe_allow_html=True)
        st.image("phoneuse.jpg")
    st.markdown("  <br>",unsafe_allow_html=True)
    if st.button(" **Show Train Data**"):
        st.dataframe(train.head())
        st.write(" Rows & Columns : ",train.shape)
    if st.button(" **Show Test Data**"):
        st.dataframe(test.head())
        st.write(" Rows & Columns : ",test.shape)
    st.markdown("<hr> ",unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.write("#### Libraries Used  ")
        st.markdown("""
    - NumPy
    - Scikit learn
    - Pandas
    - Seaborn
    - Matplotlib 

""")
    
    with col2:
        st.write("#### Models Implemented   ",unsafe_allow_html=True)
        st.markdown(     """
    - Logistic Regression
    - Support Vector Classifier
    - Decision Tree
    - Random Forest

"""  )
    st.write(" <hr>  ",unsafe_allow_html=True)


    
    

if selected=="Data Analysis":
    # progress=st.progress(0)
    # for i in range(100):
    #     time.sleep(0.01)
    #     progress.progress(i+1)
    width=4
    height=3
    if st.button("### **Check For Missing Values & Duplicates In Datasets**"):
         st.write("Number of missing values in test data: ",test.isna().values.sum())
         st.write("Number of missing values in train data: ",train.isna().values.sum())
         st.write("Number of duplicates in train data: ",sum(train.duplicated()))
         st.write("Number of duplicates in test data: ",sum(test.duplicated()))
    st.write("<hr>",unsafe_allow_html=True)
    st.write("#### Class Imbalances In Datasets")
    st.markdown("  <br>",unsafe_allow_html=True)
    if st.button(" **Edit Plot Size**"):
        width = st.slider("plot width", 1, 25, 3)
        height = st.slider("plot height", 1, 25, 1)
    fig = plt.figure(figsize=(width,height))
    plt.xticks(rotation=45)
    sns.countplot(x = "Activity", data = train)
    st.pyplot(fig)
    st.write("##### No class imbalance exists as each activity has similar number of observations. <hr>",unsafe_allow_html=True)
    st.write("## Exploratory Data Analysis")
    st.write("### 1.How Acceleration Relates To Body Activity ?")
    res=st.selectbox("Choose Plot : ",["DistPlot","BoxPlot"],index=0)
    if res=="DistPlot":
        facetgrid=sns.FacetGrid(train,hue='Activity',height=5,aspect=3)
        st.pyplot(facetgrid.map(sns.distplot,'tBodyAccMag-mean()',hist=False).add_legend())
    if res=="BoxPlot":
        fig = plt.figure(figsize=(5,3))
        sns.boxplot(x="Activity",y="tBodyAccMag-mean()",data=train,showfliers=False)
        plt.xticks(rotation=30)
        st.pyplot(fig)
    st.write("### 2.How Angle (gravityMean with X-axis) Relates To Body Activity ?")
    fig = plt.figure(figsize=(5,3))
    sns.boxplot(x="Activity",y='angle(X,gravityMean)',data=train,showfliers=False)
    plt.xticks(rotation=30)
    st.pyplot(fig)
    st.write("### 3.How Angle (gravityMean with Y-axis) Relates To Body Activity ?")
    fig = plt.figure(figsize=(5,3))
    sns.boxplot(x="Activity",y='angle(Y,gravityMean)',data=train,showfliers=False)
    plt.xticks(rotation=30)
    st.pyplot(fig)
    st.write("### 4. Analysing Data using PCA and tsne techniques:")
    st.write("### Using PCA")
    xforpca=train.drop(['Activity','subject'],axis=1)
    pca=PCA(n_components=2,random_state=0).fit_transform(xforpca)
    fig = plt.figure(figsize=(5,3))
    sns.scatterplot(x=pca[:,0],y=pca[:,1],hue=train['Activity'])
    st.pyplot(fig)
    st.write("### Using TSNE")
    with st.spinner('Running the TSNE technique (takes about 30 seconds)...'):
        xfortsne=train.drop(['Activity','subject'],axis=1)
        tsne=TSNE(n_components=2,random_state=0,n_iter=1000).fit_transform(xfortsne)
        st.success('Done!')
    fig = plt.figure(figsize=(5,3))
    sns.scatterplot(x=tsne[:,0],y=tsne[:,1],hue=train['Activity'])
    st.pyplot(fig)

if selected=="Prediction":
    xtrain=train.drop(['Activity','subject'],axis=1)
    ytrain=train.Activity
    xtest=test.drop(['Activity','subject'],axis=1)
    ytest=test.Activity
    st.write("##### Shape of train data :",xtrain.shape)
    st.write("##### Shape of test data :",xtest.shape)
    res=st.selectbox("#### Choose The Classification Model : ",["Logistic Regression","SVM","Decision Tree & Random Forest"],index=1)
    if res=="Logistic Regression":
        with st.spinner('Running the Logistic Regression model (takes about a minute)...'):
             parameters={'max_iter':[500,100,1500]}
             lr_classifier=LogisticRegression()
             lr_classifier_rs=RandomizedSearchCV(lr_classifier,param_distributions=parameters,random_state=42)
             lr_classifier_rs.fit(xtrain,ytrain)
             y_pred_lr=lr_classifier_rs.predict(xtest)
             lr_acc=accuracy_score(y_true=ytest,y_pred=y_pred_lr)
             cm=confusion_matrix(ytest,y_pred_lr)
             st.success('Done!')
             st.write("## Prediction Results")
             st.write("##### Accuracy is :",lr_acc*100,"%")
             st.write("### Confusion Matrix ")
             fig=plt.figure(figsize=(8,6))
             cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_pred_lr)).plot()
             plt.xticks(rotation=45)
             st.set_option('deprecation.showPyplotGlobalUse', False)
             st.pyplot()
    if res=="SVM":
        with st.spinner('Running the SVM model (takes about a minute)...'):
             parameters={'kernel':['rbf','linear','poly','sigmoid'],
            'C':[100,50]}
             svm_rs=RandomizedSearchCV(SVC(),param_distributions=parameters,cv=3,random_state=42)
             svm_rs.fit(xtrain,ytrain)
             y_pred_lr=svm_rs.predict(xtest)
             svm_acc=accuracy_score(y_true=ytest,y_pred=y_pred_lr)
             cm=confusion_matrix(ytest,y_pred_lr)
             st.success('Done!')
             st.write("## Prediction Results")
             st.write("##### Accuracy is :",svm_acc*100,"%")
             st.write("### Confusion Matrix ")
             fig=plt.figure(figsize=(8,6))
             cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_pred_lr)).plot()
             plt.xticks(rotation=45)
             st.set_option('deprecation.showPyplotGlobalUse', False)
             st.pyplot()
    if res=="Decision Tree & Random Forest":
        st.write("## Decision Tree")
        with st.spinner('Running the Decision Tree model (takes about a minute)...'):
             parameters={'max_depth' :np.arange(2,10,2)}
             dt_classifier_rs=RandomizedSearchCV(DecisionTreeClassifier(),param_distributions=parameters,cv=5,random_state=42)
             dt_classifier_rs.fit(xtrain,ytrain)
             y_pred_lr=dt_classifier_rs.predict(xtest)
             dt_classifier_acc=accuracy_score(y_true=ytest,y_pred=y_pred_lr)
             cm=confusion_matrix(ytest,y_pred_lr)
             st.success('Done!')
             st.write("## Prediction Results")
             st.write("##### Accuracy is :",dt_classifier_acc*100,"%")
             st.write("### Confusion Matrix ")
             fig=plt.figure(figsize=(8,6))
             cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_pred_lr)).plot()
             plt.xticks(rotation=45)
             st.set_option('deprecation.showPyplotGlobalUse', False)
             st.pyplot()
        st.write("## Random Forest")
        with st.spinner('Running the Random Forest model (takes about a minute)...'):
             parameters={'n_estimators' :np.arange(2,10,2),
             'max_depth' :np.arange(2,10,2)}
             rf_classifier_rs=RandomizedSearchCV(RandomForestClassifier(),param_distributions=parameters,cv=5,random_state=42)
             rf_classifier_rs.fit(xtrain,ytrain)
             y_pred_lr=rf_classifier_rs.predict(xtest)
             rf_classifier_acc=accuracy_score(y_true=ytest,y_pred=y_pred_lr)
             cm=confusion_matrix(ytest,y_pred_lr)
             st.success('Done!')
             st.write("## Prediction Results")
             st.write("##### Accuracy is :",rf_classifier_acc*100,"%")
             st.write("### Confusion Matrix ")
             fig=plt.figure(figsize=(8,6))
             cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=np.unique(y_pred_lr)).plot()
             plt.xticks(rotation=45)
             st.set_option('deprecation.showPyplotGlobalUse', False)
             st.pyplot()




        
        
          

   
    