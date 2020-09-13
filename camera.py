import cv2
import boto3
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIAXUSHUTWWXATS6UHP",
                        aws_secret_access_key="C5RS8Ep8kLOPHm8ns2uyRGPm13U1yAPEjL73MbmB",
                        aws_session_token="FwoGZXIvYXdzEJD//////////wEaDDFmk3LYY7T9Qs/t2CLUAQFflixhKtJPN9p2axLVNBIk0pAX6alApoLIfgxM8tQeuHvm7p1BYs5Gq/en6U2MoM93HhQ5re5m6WgR7KyPBFJ6taTakV3HEqq47DnlE6UWzHUhYCKsnPjrvlhG9DocNpIQXirecuJbsokUlLqnusJ4cKbo8EG3xgumbzmh1ARorZnIVM3QbsdF/EYTwN0goeqBDZ3P2ojXx0HFAs6rY4hP+YJ0/0TGnmmb37+W2Emv12l8fqiI1F6SbPwcruXVTi6Wt1v9WpZgvrF6WpAq7LJ/TI/fKPXM8foFMi3smdOHhyxAdOxucQ7YI8Bm7Q6Rj6DmP1QjchMWzzDkepLZnYRD1y5snAiCc34=",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:525210000813:project/mask-detect/version/mask-detect.2020-09-12T01.37.40/1599854861158',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://4x3o3qourh.execute-api.us-east-1.amazonaws.com/maskcount?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
