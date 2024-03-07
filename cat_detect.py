import cv2 as cv

img = cv.imread("Photos/cat_04.jpg")

cv.imshow("cat", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

haar_cascade = cv.CascadeClassifier("haar_cat.xml") #reads the xml code, and stores in the variable

cat_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3) #returns the rectangles from the cat to variable
#scale factor= image reduction at each scale

print(f"Number of cats found = {len(cat_rect)}")

for (x, y,w,h) in cat_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2 )

cv.imshow("Detected Faces", img)

cv.waitKey(0)