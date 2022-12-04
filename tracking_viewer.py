import cv2
import numpy as np
import os

from cv_viewer.utils import *
import pyzed.sl as sl

#----------------------------------------------------------------------
#       2D VIEW
#----------------------------------------------------------------------
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 

def cvt(pt, scale):
    '''
    Function that scales point coordinates
    '''
    out = [pt[0]*scale[0], pt[1]*scale[1]]
    return out

def render_2D(left_display, img_scale, objects, is_tracking_on, body_format):
    '''
    Parameters
        left_display (np.array): numpy array containing image data
        img_scale (list[float])
        objects (list[sl.ObjectData]) 
    '''
    overlay = left_display.copy()

    # Render skeleton joints and bones
    for obj in objects:
        if render_object(obj, is_tracking_on):
            if len(obj.keypoint_2d) > 0:
                color = generate_color_id_u(obj.id)
                # POSE_18
                if body_format == sl.BODY_FORMAT.POSE_18:
                    # Draw skeleton bones
                    for part in SKELETON_BONES:
                        kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
                        kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)
                        # Check that the keypoints are inside the image
                        if(kp_a[0] < left_display.shape[1] and kp_a[1] < left_display.shape[0] 
                        and kp_b[0] < left_display.shape[1] and kp_b[1] < left_display.shape[0]
                        and kp_a[0] > 0 and kp_a[1] > 0 and kp_b[0] > 0 and kp_b[1] > 0 ):
                            cv2.line(left_display, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])), color, 1, cv2.LINE_AA)

                    # Get spine base coordinates to create backbone
                    left_hip = obj.keypoint_2d[sl.BODY_PARTS.LEFT_HIP.value]
                    right_hip = obj.keypoint_2d[sl.BODY_PARTS.RIGHT_HIP.value]
                    spine = (left_hip + right_hip) / 2
                    kp_spine = cvt(spine, img_scale)
                    kp_neck = cvt(obj.keypoint_2d[sl.BODY_PARTS.NECK.value], img_scale)
                    # Check that the keypoints are inside the image
                    if(kp_spine[0] < left_display.shape[1] and kp_spine[1] < left_display.shape[0] 
                    and kp_neck[0] < left_display.shape[1] and kp_neck[1] < left_display.shape[0]
                    and kp_spine[0] > 0 and kp_spine[1] > 0 and kp_neck[0] > 0 and kp_neck[1] > 0
                    and left_hip[0] > 0 and left_hip[1] > 0 and right_hip[0] > 0 and right_hip[1] > 0 ):
                        cv2.line(left_display, (int(kp_spine[0]), int(kp_spine[1])), (int(kp_neck[0]), int(kp_neck[1])), color, 1, cv2.LINE_AA)

                    # Skeleton joints for spine
                    if(kp_spine[0] < left_display.shape[1] and kp_spine[1] < left_display.shape[0]
                    and left_hip[0] > 0 and left_hip[1] > 0 and right_hip[0] > 0 and right_hip[1] > 0 ):
                        cv2.circle(left_display, (int(kp_spine[0]), int(kp_spine[1])), 3, color, -1)
                        
                    # Extract landmarks
                    # Counter variables
                    counter = 0
                    stage = None
                    try:
                        # Get coordinates
                        left_shoulder = obj.keypoint_2d[sl.BODY_PARTS.LEFT_SHOULDER.value]
                        left_elbow = obj.keypoint_2d[sl.BODY_PARTS.LEFT_ELBOW.value]
                        left_wrist = obj.keypoint_2d[sl.BODY_PARTS.LEFT_WRIST.value]
                            
                        right_shoulder = obj.keypoint_2d[sl.BODY_PARTS.RIGHT_SHOULDER.value]
                        right_elbow = obj.keypoint_2d[sl.BODY_PARTS.RIGHT_ELBOW.value]
                        right_wrist = obj.keypoint_2d[sl.BODY_PARTS.RIGHT_WRIST.value]
                        
                        left_shoulder_y = obj.keypoint_2d[sl.BODY_PARTS.LEFT_SHOULDER.value][1]
                        right_shoulder_y = obj.keypoint_2d[sl.BODY_PARTS.RIGHT_SHOULDER.value][1]
            
                        # Calculate angle
                        angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                        angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
                        # Curl counter logic
                        #if angle1 and angle2 > 100:
                        #    stage = "down"
                        #if (angle1 and angle2 < 70) and stage =='down':
                        #    stage = "up"
                        #    counter +=1
                        #    os.system("mpg123"+"1.mp3")
                            
                        if angle1 and angle2 > 70:
                            stage = None
                        else:
                            stage = "correct"
                            counter = 1
                        
                        if left_shoulder_y > right_shoulder_y + 50:
                            stage = "right"
                            counter = 1
                        if right_shoulder_y > left_shoulder_y + 50:
                            stage = "left"
                            counter = 1
                            
                        if stage == "right" or "left":
                            pass
                        elif (counter == 1 and (angle1 and angle2 < 70)):
                            stage = "correct"
                            counter = 2

                    except:
                        pass
        
                    # Render counter and setup status box
                    cv2.rectangle(left_display, (0,0), (225,73), (245,117,16), -1)
        
                    # Rep data
                    cv2.putText(left_display, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(left_display, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
                    # Stage data
                    cv2.putText(left_display, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
                    cv2.putText(left_display, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    
                elif body_format == sl.BODY_FORMAT.POSE_34:
                    # Draw skeleton bones
                    for part in sl.BODY_BONES_POSE_34:
                        kp_a = cvt(obj.keypoint_2d[part[0].value], img_scale)
                        kp_b = cvt(obj.keypoint_2d[part[1].value], img_scale)
                        # Check that the keypoints are inside the image
                        if(kp_a[0] < left_display.shape[1] and kp_a[1] < left_display.shape[0] 
                        and kp_b[0] < left_display.shape[1] and kp_b[1] < left_display.shape[0]
                        and kp_a[0] > 0 and kp_a[1] > 0 and kp_b[0] > 0 and kp_b[1] > 0 ):
                            cv2.line(left_display, (int(kp_a[0]), int(kp_a[1])), (int(kp_b[0]), int(kp_b[1])), color, 1, cv2.LINE_AA)
            
                # Skeleton joints
                for kp in obj.keypoint_2d:
                    cv_kp = cvt(kp, img_scale)
                    if(cv_kp[0] < left_display.shape[1] and cv_kp[1] < left_display.shape[0]):
                        cv2.circle(left_display, (int(cv_kp[0]), int(cv_kp[1])), 3, color, -1)
                    

    cv2.addWeighted(left_display, 0.9, overlay, 0.1, 0.0, left_display)
