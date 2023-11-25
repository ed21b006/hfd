def extract_eye(landmark):
    (x1, y1) = landmark[1]
    x2 = landmark[15][0]
    y2 =landmark[24][1] * 2 - landmark[44][1]
    
    left = ((x1, y1), (int((x2+x1)/2), y2))
    right = ((int((x2+x1)/2), y1), (x2, y2))

    return (left, right)




if __name__=='__main__':
    import cv2 as cv
    import os, sys
    sys.path.append(os.getcwd())
    from FaceRecognitionModule.dlib_model import FaceModule
    obj=FaceModule()

    video_capture = cv.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        try:
            landmark = obj.find_landmarks(gray)[0]
            faces = obj.detections[0]
        except:
            cv.imshow('Video', frame)
            if(cv.waitKey(1) & 0xFF == ord('q')):
                break
            continue

        (x,y,x2,y2) = (faces.left(), faces.top(), faces.right(), faces.bottom())
        cv.rectangle(frame,(x,y),(x2,y2),(255,0,0),2)
        roi_gray = gray[y:y2, x:x2]
        roi_color = frame[y:y2, x:x2]

        eyes = extract_eye(landmark)
        cv.circle(frame, landmark[1], 1, (0,0,255), 2)
        cv.circle(frame, landmark[15], 1, (0,0,255), 2)
        cv.circle(frame, landmark[24], 1, (0,0,255), 2)
        cv.circle(frame, landmark[44], 1, (0,0,255), 2)
        for ((ex,ey),(ex2,ey2)) in eyes:
            cv.rectangle(frame,(ex,ey2),(ex2,ey),(0,255,0),2)
            ...

        cv.imshow('Video', frame)

        if(cv.waitKey(1) & 0xFF == ord('q')):
            break

    #Finally when video capture is over, release the video capture and destroyAllWindows
    video_capture.release()
    cv.destroyAllWindows()