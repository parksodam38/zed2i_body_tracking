########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
   This sample shows how to detect a human bodies and draw their 
   modelised skeleton in an OpenGL window
"""
import cv2
import sys
import pyzed.sl as sl
import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer
import numpy as np

# Counter variables
counter = 0
stage = None

# Calculate angles
def calculate_angle(a, b, c):

    a = np.array(a); b = np.array(b); c = np.array(c)
    ab = np.array(3); bc = np.array(3)

    for i in range(3):
        ab[i] = b[i] - a[i]
        bc[i] = c[i] - b[i]

    # Nomarlize ab and bc
    ab = ab / np.linalg.norm(ab)
    bc = bc / np.linalg.norm(bc)

    # Extract the angle from the dot products
    try:
        cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
    except:
        print("invalid cosine")
        exit()

    return np.degrees(angle)

if __name__ == "__main__":
    print("Running Body Tracking sample ... Press 'q' to quit")

    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
    init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    
    # If applicable, use the SVO given as parameter
    # Otherwise use ZED live stream
    if len(sys.argv) == 2:
        filepath = sys.argv[1]
        print("Using SVO file: {0}".format(filepath))
        init_params.svo_real_time_mode = True
        init_params.set_from_svo_file(filepath)

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Enable Positional tracking (mandatory for object detection)
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True            # Smooth skeleton move
    obj_param.enable_tracking = True                # Track people across images flow
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST 
    obj_param.body_format = sl.BODY_FORMAT.POSE_18  # Choose the BODY_FORMAT you wish to use

    # Enable Object Detection module
    zed.enable_object_detection(obj_param)

    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()

    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280), min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
                 , display_resolution.height / camera_info.camera_resolution.height]

    # Create OpenGL viewer
    viewer = gl.GLViewer()
    viewer.init(camera_info.calibration_parameters.left_cam, obj_param.enable_tracking,obj_param.body_format)

    # Create ZED objects filled in the main loop
    bodies = sl.Objects()
    image = sl.Mat()

    while viewer.is_available():
        # Grab an image
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # Retrieve objects
            zed.retrieve_objects(bodies, obj_runtime_param)

            # Update GL view
            viewer.update_view(image, bodies) 
            # Update OCV view
            image_left_ocv = image.get_data()
            cv_viewer.render_2D(image_left_ocv,image_scale,bodies.object_list, obj_param.enable_tracking, obj_param.body_format)
            cv2.imshow("ZED | 2D View", image_left_ocv)
            cv2.waitKey(10)

            # Extract landmarks
            try:
                # Get coordinates
                left_shoulder = sl.get_idx_34(LEFT_SHOULDER)
                left_elbow = sl.get_idx_34(LEFT_ELBOW)
                left_wrist = sl.get_idx_34(LEFT_WRIST)

                right_shoulder = sl.get_idx_34(RIGHT_SHOULDER)
                right_elbow = sl.get_idx_34(RIGHT_ELBOW)
                right_wrist = sl.get_idx_34(RIGHT_WRIST)

                # Calculate angle
                angle1 = calculate_angle(left_shoulder, left_elbow, left_wrist)
                angle2 = calculate_angle(right_shoulder, right_elbow, right_wrist)

                # Counter logic
                if(angle1 and angle2 > 165):
                    stage = "down"
                if((angle1 and angle2 < 30) and stage == 'down'):
                    stage = "up"
                    counter +=1
                    print(counter)
            except:
                pass

            # Render counter
            # Setup status box
            cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)

            # Rep data
            cv2.putText(image, 'REPS', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

            # Stage data
            cv2.putText(image, 'STAGE', (65,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (60,60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

    viewer.exit()

    image.free(sl.MEM.CPU)
    # Disable modules and close camera
    zed.disable_object_detection()
    zed.disable_positional_tracking()
    zed.close()