def match_bounding_boxes(frame_content_1, frame_content_2):
    """
    SIMPLIFIED VERSION FOR NOW!

    Match the bounding boxes between two frames so that you know which tracker ID from frame 1 corresponds to which tracker ID from frame 2.
    This is a simplified version of the final code because both the frames are exactly the same, but next we will implement camera calibration, 
    so that we know how the one camera is oriented relative to the other - and with that info we would be able to match the bounding box of one 
    to that of the otter. 

    imagine you and a few others are in the gym, with two camera facing towards you all. If we know where one camera is placed relative to the other,
    we can match the bounding box of one person in one camera to that of the same person in the other camera. (so for example. If I (Shaun for example), 
    have a tracker id of 1 in camera one, and a tracker id of 5 in the other, we have to write a algorithm that would be able to figure that out very 
    efficiently).

    Requirements:
    This algorithm has to have a linear time complexity (Think why this is critical to the optimal functioning of the software!) 


    Wrong example:
    '''
    for i in range(20):
        for j in range(10):
            print(i*j)
    '''
    The above has a time complexity of O(n^2) - which is not ganna work for us. We need a time complexity of O(n).

    Return:
        Return a dictionary that maps tracker IDs from frame 1 to tracker IDs from frame 2. 
        for example: {1:[5, 1], 2:[7, 2], 3:[8, 3]}
        This means that tracker ID 1 in frame 1 corresponds to tracker ID 5 in frame 2, 
        tracker ID 7 in frame 1 corresponds to tracker ID 2 in frame 2,
        tracker ID 8 in frame 1 corresponds to tracker ID 3 in frame 2.
    """
    return {1:[5, 1], 2:[7, 2], 3:[8, 3]} ##EXAMPLE RETURN
