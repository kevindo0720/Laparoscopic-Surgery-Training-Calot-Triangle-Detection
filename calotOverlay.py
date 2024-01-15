import cv2
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def euclidean_distance(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def adapt_triangle_to_new_bbox(original_bbox, new_bbox, triangle_vertices):
    # Flatten the triangle vertices array to get a list of (x, y) coordinates
    flat_triangle = triangle_vertices.reshape(-1, 2)

    # Calculate the scaling factors for width and height
    scale_x = new_bbox[2] / original_bbox[2]
    scale_y = 1.5 * (new_bbox[3] / original_bbox[3])  # Adjust the scaling factor for vertical expansion

    # Scale the triangle vertices to match the new bounding box dimensions
    scaled_triangle = (flat_triangle - np.array([original_bbox[0], original_bbox[1]])) * np.array([scale_x, scale_y]) + np.array([new_bbox[0], new_bbox[1]])

    # Calculate the translation to ensure the top and bottom points touch the bounding box
    translation_y_top = -np.min(scaled_triangle[:, 1]) + new_bbox[1]
    translation_y_bottom = -np.max(scaled_triangle[:, 1]) + new_bbox[1] + new_bbox[3]

    # Translate the scaled triangle along the y-axis
    translated_triangle = scaled_triangle + np.array([0, translation_y_top])

    # Reshape the translated triangle to the original format
    adapted_triangle = translated_triangle.reshape(triangle_vertices.shape)

    return adapted_triangle

def quadrilateral_area(vertices):
    n = len(vertices)
    area = 0.5 * abs(sum(vertices[i][0]*vertices[(i+1)%n][1] - vertices[(i+1)%n][0]*vertices[i][1] for i in range(n)))
    return area

def triangle_area(vertices):
    area = 0.5 * abs(vertices[0][0] * (vertices[1][1] - vertices[2][1]) + vertices[1][0] * (vertices[2][1] - vertices[0][1]) + vertices[2][0] * (vertices[0][1] - vertices[1][1]))
    return area

def findBbox(coordinates):
    
    # Flatten the array to get a list of (x, y) coordinates
    flat_coordinates = coordinates.reshape(-1, 2)

    # Calculate the bounding box coordinates
    min_x, min_y = np.min(flat_coordinates, axis=0)
    max_x, max_y = np.max(flat_coordinates, axis=0)

    # Calculate width and height of the bounding box
    width = math.ceil((max_x - min_x)*1.25)
    height = math.ceil((max_y - min_y)*1.5)
    
    return [min_x,min_y,width,height]

def find_median(points):
    # Extract x and y coordinates
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]

    # Find median for x and y coordinates
    x_median = np.median(x_values)
    y_median = np.median(y_values)

    return x_median, y_median

def detectTriangleShapes(frame):

    original_image = frame
    
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Convert the image to HSV for better handling of color ranges
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    
    # Define lower and upper bounds for the color yellow in HSV
    lower_yellow = np.array([0, 100, 100])
    upper_yellow = np.array([10, 255, 255])

    # Create a mask for yellow regions
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    
    # Combine the grayscale image and the yellow mask
    combined_mask = cv2.bitwise_or(gray_image, yellow_mask)
    
    # Apply Gaussian Blur to the combined mask
    blurred = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # FIND 15th percentile of values based on the median grayscale value
    flat_image = combined_mask.flatten()
    flat_image = sorted(flat_image)
    percent15thcolor = np.percentile(flat_image, 15)

    # Threshold the grayscale image to identify dark regions
    _, binary_image = cv2.threshold(combined_mask, percent15thcolor, 255, cv2.THRESH_BINARY_INV)


    # Find contours in the binary image and overlay it
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)    
    height, width, _ = original_image.shape
    black_image = np.zeros((height, width, 3), dtype=np.uint8)
    black_image2 = black_image.copy()
    
    
    #filtered out small contours (insiginificant ones), do this based on size, choose the one that is above 90th percentile
    filtered_contours_size = []
    len_contours = []
    for i in contours:
        len_contours.append(len(i))
    percent90thlen = np.percentile(len_contours, 80)
    for i in contours:
        if len(i) > percent90thlen:
            filtered_contours_size.append(i)
    
    filtered_contours_triangle = []
    #filter out based on triangular shape
    for cnt in filtered_contours_size:
        # Approximate the contour to reduce the number of vertices
        epsilon = 0.06 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)      
        if abs(len(approx)-3) <= 1 and len(approx) != 2:
            filtered_contours_triangle.append(approx)  
            
    
    filtered_contours_area_shapes = []
    areas = []
    for shape in filtered_contours_triangle:
        coords = []
        for vertex in shape:
            vertexExtracted = vertex[0]
            coord = [vertexExtracted[0],vertexExtracted[1]]
            coords.append(coord)
            
        if len(shape) == 3:
            areas.append(triangle_area(coords))
        elif len(shape) == 4:
            areas.append(quadrilateral_area(coords))
        
    seventypercentilearea = np.percentile(areas,50)
    
    coordsSelectedShapes = []
    for shape in filtered_contours_triangle:
        coords = []
        for vertex in shape:
            vertexExtracted = vertex[0]
            coord = [vertexExtracted[0],vertexExtracted[1]]
            coords.append(coord)
            
        
        if len(shape) == 3:
            areaShape = triangle_area(coords)
        elif len(shape) == 4:
            areaShape = quadrilateral_area(coords)
        
        if areaShape >= seventypercentilearea and areaShape > 1000:
            medianx,mediany = find_median(coords)
            coordsSelectedShapes.append([medianx,mediany])
            filtered_contours_area_shapes.append(shape)
    
    return coordsSelectedShapes,filtered_contours_area_shapes

def Overlay(frame):
    frameread = frame
    selectedShapesCoords,contours = detectTriangleShapes(frameread)
    trainImage = cv2.imread("./sampleImage20.png",cv2.IMREAD_GRAYSCALE)
    trainImage2 = cv2.imread("./sampleImage18.png",cv2.IMREAD_GRAYSCALE)
    trainImage3 = cv2.imread("./sampleImage19.png",cv2.IMREAD_GRAYSCALE)
    trainImage4 = cv2.imread("./sampleImage21.png",cv2.IMREAD_GRAYSCALE)


    trainImage = cv2.GaussianBlur(trainImage, (5, 5), 0)
    trainImage2 = cv2.GaussianBlur(trainImage2, (5, 5), 0)
    trainImage3 = cv2.GaussianBlur(trainImage3, (5, 5), 0)
    trainImage4 = cv2.GaussianBlur(trainImage4, (5, 5), 0)

    testImage = frameread
    testImage = cv2.GaussianBlur(testImage, (5, 5), 0)

    trains = [trainImage, trainImage2, trainImage3, trainImage4]
    test = [testImage]
    setMedianTrains = []

    for imageTrain in trains:
        #create ORB object using sift 
        orb = cv2.xfeatures2d.SIFT_create()
        kp1, des1, = orb.detectAndCompute(imageTrain, None)
        kp2, des2 = orb.detectAndCompute(testImage, None)

        #feature matching
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = bf.match(des1,des2)

        good_matches = []
        for i in range(len(matches)-1):
            if matches[i].distance < 0.75 * matches[i+1].distance:
                good_matches.append(matches[i])

        locations = []
        good_matches = sorted(matches, key = lambda x:x.distance)
        for match in good_matches[:150]:
            trainindex = match.trainIdx
            # Retrieve the keypoint location in the test image
            point = kp2[trainindex].pt
            locations.append(point)


        # img3 = cv2.drawMatches(trainImage,kp1, testImage, kp2, good_matches[:150], testImage, flags=2)
        medx,medy = find_median(locations)
        setMedianTrains.append([medx,medy])
        
        
    bestshapeIndex = 0
    distanceMax = 39428398434239843984938

    index = 0
    for coords in selectedShapesCoords:
        distance1 = euclidean_distance(coords[0],coords[1],setMedianTrains[0][0],setMedianTrains[0][1])
        distance2 = euclidean_distance(coords[0],coords[1],setMedianTrains[1][0],setMedianTrains[1][1])
        distance3 = euclidean_distance(coords[0],coords[1],setMedianTrains[2][0],setMedianTrains[2][1])
        distance4 = euclidean_distance(coords[0],coords[1],setMedianTrains[3][0],setMedianTrains[3][1])

        totalDistance = distance1+distance2 + distance3 + distance4
        if totalDistance < distanceMax:
            bestshapeIndex = index
            distanceMax = totalDistance
        index+=1
        
    contourFinal = contours[bestshapeIndex]

    # cv2.drawContours(frameread, [contourFinal], -1, (0, 255, 0), cv2.FILLED)
    mask = np.zeros_like(frameread)
    # Draw the filled contour on the mask
    cv2.fillPoly(mask, [contourFinal], color=(0, 255, 0, 240))  # The fourth value (128) is the alpha value for transparency
    # Blend the mask with the original image
    result = cv2.addWeighted(frameread, 1, mask, 0.5, 0)
    
    return result, contourFinal
    
    
def tracking(coordsBBox,frame):
    bbox = (coordsBBox[0],coordsBBox[1],coordsBBox[2],coordsBBox[3])
    tracker = cv2.TrackerKCF_create()
    ok = tracker.init(frame, bbox)
    
    # Update tracker
    ok, bbox = tracker.update(frame)
 
    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
        
    return frame 
    
 
def main():
    cap = cv2.VideoCapture("IMG_5605.mov")
    
    # Set the output video file properties
    output_file = "overlayed5605.mp4"
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count < 1:
            processed, contours = Overlay(frame)
            bboxContours = findBbox(contours)
            cv2.imwrite("filled3.jpg", processed)
            cv2.imshow('Original Frame', frame)
            bbox = bboxContours
            tracker = cv2.TrackerKCF_create()
            ok = tracker.init(frame, bbox)
            originalbbox = bbox
        else:
            timer = cv2.getTickCount()
            ok, bbox = tracker.update(frame)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            adaptedtriangles = adapt_triangle_to_new_bbox(originalbbox, bbox, contours)

            mask = np.zeros_like(frame)
            cv2.fillPoly(mask, [adaptedtriangles.astype(int)], color=(0, 255, 0, 240))
            result = cv2.addWeighted(frame, 1, mask, 0.5, 0)

            if ok:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            else:
                cv2.putText(result, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            cv2.imshow("Tracking", result)
            out.write(result)  # Write the processed frame to the output video

        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Release the VideoWriter object
    cv2.destroyAllWindows()

main()
