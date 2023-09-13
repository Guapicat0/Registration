import cv2

def Knn_match(des1, des2,normType,distance_threshold ):
    bf = cv2.BFMatcher(normType, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < distance_threshold  * n.distance:
            good.append(m)

    return good

def BF_match(des1,des2,normType, crossCheck):
    bf = cv2.BFMatcher(normType, crossCheck)
    matches = bf.match(des1, des2)
    '''
    good = []
    match_distance=[]

    #matches = sorted(matches, key=lambda x: x.distance)

    for match in matches:
        match_distance.append(match.distance)
        if match.distance < distance_threshold:
            good.append(match)

    # 筛选
    #matches = matches[0:20]
    '''
    return matches