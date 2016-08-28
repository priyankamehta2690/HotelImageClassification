import numpy                                                  
import glob
import csv
from PIL import Image
from sklearn import svm
from PIL import ImageFile
from sklearn import cross_validation
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from os import listdir
import re

path = '/Users/PriyankaMehta/Desktop/HotelImageClassification/'  
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
def main():
    
    accuracy=supportVectorMachine()
    print accuracy

#Returns the feature vector and class labels for the training data    
def getTrainingData():
    dataLabel=[]
    trainingLabel=[]
    featureVectorTrain=[]
    damagedImagesTrain=0
    #Reading the class labels from train.csv and writing to a list
    with open(path + 'trainSample.csv', 'rb') as f:
        reader = csv.reader(f)
        f.next()
        for row in reader:
            dataLabel.append(row)
    #Reading the training images in increasing order of their IDs      
    imageFolderPath = path + 'train/'
    imagePath = sorted(glob.glob(imageFolderPath+'/*.jpg'),key=numericalSort) 
    for i in range(5100):
        try:
            #Resizing the image and converting it to grayscale and stored in numpy array
            imageArray = numpy.array(Image.open(imagePath[i]).resize((255,255)).convert('L'), 'f')
            #Computing the histogram of oriented gradients for each training image
            fd, hog_image = hog(imageArray, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
            if len(featureVectorTrain)==0:
                featureVectorTrain = fd
            else:
                featureVectorTrain = numpy.vstack((featureVectorTrain,fd))
            #Computing the id of image under consideration
            id=imagePath[i].split('/')[-1][:-4]
            #Computing the class label for each image where class=1 if col1 contains 1,
            #class=2 if col2 contains 1 and so on in train.csv file
            if(id==dataLabel[i][0]):
                for j in range(1,9):
                    if(dataLabel[i][j]=='1'):
                        trainingLabel.append(j)                       
        except IOError:
            #Counting the number of corrupted images in training data
            damagedImagesTrain=damagedImagesTrain+1
    return featureVectorTrain, trainingLabel

#Returns the feature vector and IDs for test images alongwith the number of corrupted images in test data 
#and their IDs              
def getTestData():
    featureVectorTest=[]
    damagedTestImagesId=[]
    damagedImagesTest=0
    #Path to the test data directory
    imageFolderPathTest = path + 'test/'
    imagePathTest = glob.glob(imageFolderPathTest+'/*.jpg')
    #Reading the IDs of test data 
    testImageId=listdir(imageFolderPathTest)
    for i in range(2000):
        testImageId[i] = testImageId[i][:-4]
    for i in range(2000):
        try:
            #Resizing the image and converting it to grayscale and stored in numpy array
            imageArrayTest = numpy.array(Image.open(imagePathTest[i]).resize((255,255)).convert('L'), 'f')
            #Computing the histogram of oriented gradients for each test image            
            fd, hog_image = hog(imageArrayTest, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualise=True)
            if len(featureVectorTest)==0:
                featureVectorTest = fd
            else:
                featureVectorTest = numpy.vstack((featureVectorTest,fd))
        except IOError:
            #Counting the number of corrupted images in test data and storing their IDs
            damagedTestImagesId.append(testImageId[i])
            damagedImagesTest=damagedImagesTest+1
    #Separating out the IDs of corrupted images
    for item in damagedTestImagesId:
        testImageId.remove(item)
    return featureVectorTest, testImageId, damagedImagesTest, damagedTestImagesId
    
#Performs PCA on training and test data
def pca():
    #Get the feature vector and class labels for the training data  
    featureVectorTrain, trainingLabel=getTrainingData()
    #Get the feature vector and IDs for test images alongwith the number of corrupted images in test data 
    #and their IDs
    featureVectorTest, testImageId,damagedImagesTest, damagedTestImagesId, =getTestData()          
    #Train PCA using the training feature vector    
    pca = PCA(n_components=100).fit(featureVectorTrain)
    arrayTrain = pca.transform(featureVectorTrain)
    print arrayTrain.shape
    arrayTest = pca.transform(featureVectorTest)
    return arrayTrain, trainingLabel, arrayTest, testImageId, damagedTestImagesId, damagedImagesTest

#Calculates the probability of each class for each test image using SVM classifier and writing it to test.csv 
def supportVectorMachine():
    #PCA
    arrayTrain, trainingLabel, arrayTest, testImageId, damagedTestImagesId, damagedImagesTest=pca()
    #Cross validation with 30% validation data    
    imagesArray_train, imagesArray_validate, y_train, y_validate = cross_validation.train_test_split(arrayTrain, trainingLabel, test_size=0.3, random_state=0)
    #Scaling the training, validation and test array    
    stdSlrTrain=StandardScaler().fit(imagesArray_train)
    imagesArray_train=stdSlrTrain.transform(imagesArray_train)
    stdSlrValidate=StandardScaler().fit(imagesArray_train)
    imagesArray_validate=stdSlrValidate.transform(imagesArray_validate)
    stdSlrTest=StandardScaler().fit(imagesArray_train)
    arrayTest=stdSlrTest.transform(arrayTest)
    #Training the SVM classifier
    clf = svm.SVC(probability=True, C=1.0, gamma=0.1, kernel='poly')
    clf.fit(arrayTrain,trainingLabel)
    accuracy= clf.score(imagesArray_validate, y_validate)
    print clf.score(imagesArray_validate, y_validate)
    #Writing probabilities to csv file
#    with open('test.csv', 'wb') as csvfile:
#         fieldnames = ['id', 'col1','col2','col3','col4','col5','col6','col7','col8']
#         writer = csv.DictWriter(csvfile, fieldnames=fieldnames,delimiter=',',quoting=csv.QUOTE_ALL)
#         writer.writeheader()
#         #Writing probability of each class for each test image
#         for i in xrange(len(arrayTest)):
#             probability = str(clf.predict_proba(arrayTest[i])).split()
#             if "]]" in probability[8]: 
#                 probability[8]=probability[8][:-2]
#             writer.writerow({'id' : testImageId[i],'col1':probability[1],'col2':probability[2],'col3':probability[3],'col4':probability[4],'col5':probability[5],'col6':probability[6],'col7':probability[7],'col8':probability[8] } )
#             csvfile.flush()
#         #Assigning equal probability for each class of corrupted test images 
#         for j in xrange(len(damagedTestImagesId)):
#             writer.writerow({'id' : damagedTestImagesId[j],'col1':'0.125','col2':'0.125','col3':'0.125','col4':'0.125','col5':'0.125','col6':'0.125','col7':'0.125','col8':'0.125'} )
#             csvfile.flush()
    print accuracy        
    return accuracy

numbers = re.compile(r'(\d+)')
#Sorts the set of values in increasing order
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts   
    
if __name__ == '__main__':
    main()