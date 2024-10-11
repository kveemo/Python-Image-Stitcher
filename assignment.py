# -*- coding: utf-8 -*-
import cv2
import numpy as np
import PySimpleGUI as sg

img_left = ""
img_right = "" #Empty variables for left/right images
img_left_gray = ""
img_right_gray = ""
flag = False #flag to see if user selected an image, mainly here for error handling

#-----THE FUNCTIONS FOR WORKING WITH THE IMAGES ARE HERE-----
def get_harris_corner(img):
    harris_img = img.copy() #create copy of image so original can be reused for sift
    gray_img = np.float32(cv2.cvtColor(harris_img, cv2.COLOR_BGR2GRAY))
    corners = cv2.cornerHarris(gray_img, 2, 7, 0.04)
    harris_img[corners>0.01*corners.max()]=[0,0,255]
    
    return harris_img

#Step 3

def get_sift_features(img):
    sift_img = img.copy() #create copy so original image isnt modified
    gray_img = cv2.cvtColor(sift_img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img,None)
    sift_img = cv2.drawKeypoints(sift_img, keypoints, sift_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("keypoints detected: " + str(len(keypoints)))
    print("descriptor dimensions are " + str(descriptors.shape))
    return sift_img, keypoints, descriptors

def get_orb_features(img):
    keypoint_number = 2000 #Max number of keypoints returned
    orb_img = img.copy() #create copy so original image isnt modified
    gray_img = cv2.cvtColor(orb_img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(keypoint_number)
    keypoints, descriptors = orb.detectAndCompute(img,None)
    orb_img = cv2.drawKeypoints(orb_img, keypoints, orb_img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print("keypoints detected: " + str(len(keypoints)))
    print("descriptor dimensions are " + str(descriptors.shape))
    return orb_img, keypoints, descriptors
    
    

def get_ssd(descriptor_left, descriptors_right):
    #gets the ssd between a descriptor in the left image, and all the ones in the right
    distance_list = np.sum((descriptors_right-descriptor_left)**2,axis=1)
    return distance_list

def match_features(descriptors_left, descriptors_right):
    i = 0
    found_matches = [] #empty list to store matches that pass ratio test
    for descriptor_left in descriptors_left:
        distance_list = get_ssd(descriptor_left, descriptors_right)
        sorted_distance = np.argsort(distance_list) #sort the list so 2 closest are at front
        #this is used for ratio testing
        
        minimum_dist_index = sorted_distance[0]
        second_dist_index = sorted_distance[1]
        minimum_distance = distance_list[minimum_dist_index]
        second_distance = distance_list[second_dist_index]
        
        #---RATIO TEST IMPLEMENTED HERE----
        threshold = 0.5 #can change as needed
        if minimum_distance/second_distance <= threshold:
            found = cv2.DMatch(i, minimum_dist_index, minimum_distance) #convert to dmatch so it can be used in drawMatches
            found_matches.append(found)
        i+=1

    return found_matches

def match_orb_features(descriptors_left, descriptors_right):
    i = 0
    found_matches = [] #empty list to store matches that pass ratio test
    for descriptor_left in descriptors_left:
        distance_list = get_ssd(descriptor_left, descriptors_right)
        sorted_distance = np.argsort(distance_list) #sort the list so 2 closest are at front
        #this is used for ratio testing
        
        minimum_dist_index = sorted_distance[0]
        second_dist_index = sorted_distance[1]
        minimum_distance = float(distance_list[minimum_dist_index])
        second_distance = float(distance_list[second_dist_index])
        
        #---RATIO TEST IMPLEMENTED HERE----
        threshold = 0.5 #can change as needed
        if minimum_distance/second_distance <= threshold:
            found = cv2.DMatch(i, minimum_dist_index, minimum_distance) #convert to dmatch so it can be used in drawMatches
            found_matches.append(found)
        i+=1

    return found_matches

def get_homography(left_points, right_points, matches):
    global img_left
    global img_right
    
    #format left_points and right_points so they can be used in findHomography
    left_points_formatted = np.float32([left_points[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    right_points_formatted = np.float32([right_points[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    #gets homography matrix
    matrix, _ = cv2.findHomography(right_points_formatted, left_points_formatted, cv2.RANSAC)
    
    #make empty array with size img1 width + img2width (also same height)
    canvas = (img_left.shape[1] + img_right.shape[1], img_left.shape[0]) 
    stitched = cv2.warpPerspective(img_right, matrix, canvas)
    stitched[0:img_left.shape[0], 0:img_left.shape[1]] = img_left  #overlay left on warped right
    
    new_width = stitched.shape[1] // 2 #resize so it fits on my screen
    new_height = stitched.shape[0] // 2
    scaled = cv2.resize(stitched, (new_width, new_height))
    cv2.imshow("Stitched", scaled)
    return stitched


#----FUNCTIONS TO SETUP THE GUI ARE HERE----#
def create_start_gui(startup_text):
    #Sets up the initial GUI where the user can pick which image set they want 
    #Its just easier than changing the paths each time 
    global flag
    global img_left
    global img_right
    layout = [[sg.Text(text=startup_text,
                 font=('Arial Bold', 20),
                 size=20,
                 expand_x=True,
                 justification='center')],
             [sg.Text('Select an Image set:'), 
                  sg.Combo(['Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5', 'Set 6'], 
                  default_value='Set 1')],
             [sg.Button('Pick an image set', key = '-PICK_IMAGE-')]]
             

    window = sg.Window('AssignmentGUI', layout, size=(1000,250))
    sg.theme('Dark Purple 2') #looks cool
    #sg.theme('Dark Blue 8') #this one is cool too uncomment if you want to switch
    while True:
       event, values = window.read()
       print(event, values)
       if event in (None, 'Exit'):
          break
       if event == "-PICK_IMAGE-" and values[0] == 'Set 1':
          flag = True #shows user did not exit without selecting an image set
          img_left = cv2.imread(r"images\left.png", cv2.IMREAD_COLOR)
          img_right = cv2.imread(r"images\right.png", cv2.IMREAD_COLOR)
          window.close()
          

       elif event == "-PICK_IMAGE-" and values[0] == 'Set 2':
          flag = True #shows user did not exit without selecting an image set
          img_left = cv2.imread(r"images\left1.png", cv2.IMREAD_COLOR)
          img_right = cv2.imread(r"images\right1.png", cv2.IMREAD_COLOR)
          window.close()

        
       elif event == "-PICK_IMAGE-" and values[0] == 'Set 3':
          flag = True #shows user did not exit without selecting an image set
          img_left = cv2.imread(r"images\left2.png", cv2.IMREAD_COLOR)
          img_right = cv2.imread(r"images\right2.png", cv2.IMREAD_COLOR)
          window.close()

          
       elif event == "-PICK_IMAGE-" and values[0] == 'Set 4':
          flag = True #shows user did not exit without selecting an image set
          img_left = cv2.imread(r"images\left3.png", cv2.IMREAD_COLOR)
          img_right = cv2.imread(r"images\right3.png", cv2.IMREAD_COLOR)
          window.close()
    
          
       elif event == "-PICK_IMAGE-" and values[0] == 'Set 5':
          flag = True #shows user did not exit without selecting an image set
          img_left = cv2.imread(r"images\left4.png", cv2.IMREAD_COLOR)
          img_right = cv2.imread(r"images\right4.png", cv2.IMREAD_COLOR)
          window.close()

          
       if event == "-PICK_IMAGE-" and values[0] == 'Set 6':
          flag = True #shows user did not exit without selecting an image set
          img_left = cv2.imread(r"images\left5.png", cv2.IMREAD_COLOR)
          img_right = cv2.imread(r"images\right5.png", cv2.IMREAD_COLOR)
          window.close()
         
          
    window.close()
    
def create_process_gui():
    #Sets up the GUI where the user can see the effects of edge detection and stitching
    global img_left
    global img_right
    left_points  = None
    right_points = None
    
    layout = [[sg.Text(text="Feature detection and image stitching",
                 font=('Arial Bold', 20),
                 size=20,
                 expand_x=True,
                 justification='center')],
             [sg.Button('Harris Corner', key = '-HARRIS-')],
             [sg.Button('SIFT', key = '-SIFT-')],
             [sg.Button('ORB', key = '-ORB-')],
                 [sg.Button('Feature Match SIFT', key = '-MATCH-', disabled=True)],
                 [sg.Button('Feature Match ORB', key = '-MATCHORB-', disabled=True)],
                 [sg.Button('Stitch with SIFT!', key = '-STITCH-', disabled=True)],
                 [sg.Button('Stitch with ORB!', key = '-STITCHORB-', disabled=True)]]
             

    window = sg.Window('FeatureGUI', layout, size=(1000,550))
    sg.theme('Dark Purple 2') 
    while True:
       event, values = window.read()
       print(event, values)
       if event in (None, 'Exit'):
          break
       if event == "-HARRIS-":
          harris_left = get_harris_corner(img_left)
          harris_right = get_harris_corner(img_right)
          cv2.imshow("harris_left",harris_left)
          cv2.imshow("harris_right",harris_right)
          
       elif event == "-SIFT-":
           sift_left, left_points, left_desc = get_sift_features(img_left)
           sift_right, right_points, right_desc = get_sift_features(img_right)
           cv2.imshow("sift_left",sift_left)
           cv2.imshow("sift_right",sift_right)
           
           window['-MATCH-'].update(disabled=False) #Allow user to feature match
       
       elif event == "-ORB-":
           orb_left, left_points_orb, left_desc_orb = get_orb_features(img_left)
           orb_right, right_points_orb, right_desc_orb = get_orb_features(img_right)
           cv2.imshow("orb_left",orb_left)
           cv2.imshow("orb_right",orb_right)
      
           window['-MATCHORB-'].update(disabled=False) #Allow user to feature match with ORB
           
       elif event == "-MATCH-":
           found_matches = match_features(left_desc, right_desc)
           combined_img = np.hstack((img_left, img_right)) 
           drawn = cv2.drawMatches(img_left, left_points, img_right, right_points, 
                                   found_matches, combined_img, 
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
           new_width = drawn.shape[1] // 2 #resize so it fits on my screen
           new_height = drawn.shape[0] // 2
           scaled = cv2.resize(drawn, (new_width, new_height))
           cv2.imshow("matches", scaled)
           cv2.waitKey(0)  

           window['-STITCH-'].update(disabled=False) #Allow user to stitch images 
           
       elif event == "-MATCHORB-":
           found_matches_orb = match_orb_features(left_desc_orb, right_desc_orb)
           combined_img_orb = np.hstack((img_left, img_right)) 
           drawn = cv2.drawMatches(img_left, left_points_orb, img_right, right_points_orb, 
                                   found_matches_orb, combined_img_orb, 
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
           new_width = drawn.shape[1] // 2 #resize so it fits on my screen
           new_height = drawn.shape[0] // 2
           scaled = cv2.resize(drawn, (new_width, new_height))
           cv2.imshow("matches", scaled)
           cv2.waitKey(0)  

           window['-STITCHORB-'].update(disabled=False) #Allow user to stitch images 
       
       elif event == '-STITCH-':
           get_homography(left_points, right_points, found_matches)
           cv2.waitKey(0) 
       
       elif event == '-STITCHORB-':
           get_homography(left_points_orb, right_points_orb, found_matches_orb)
           cv2.waitKey(0) 
       
          
    cv2.destroyAllWindows()
    window.close()

create_start_gui("Image Set Selector")

if flag == False:
    while flag == False:
        #Error handling to make sure user cant process an empty image
        create_start_gui("Error, please select an image set before continuing")

create_process_gui()





