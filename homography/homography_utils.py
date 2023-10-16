# Croke Park. Stadium should be included in initial config and we can pick custom dst_points as needed.
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def get_dst_points():
    dst = {
        '0A': [72.5, 39],
        '0B': [72.5, 49],
        '1A': [0, 0],
        '1B': [0, 34.5],
        '1C': [0, 37],
        '1D': [0, 51],
        '1E': [0, 53.5],
        '1F': [0, 88],
        '1G': [4.5, 37],
        '1H': [4.5, 51],
        '1I': [13, 0],
        '1J': [13, 34.5],
        '1K': [13, 53.5],
        '1L': [13, 88],
        '1M': [21, 0],
        '1N': [21, 34.5],
        '1O': [21, 53.5],
        '1P': [21, 88],
        '1Q': [45, 0],
        '1R': [45, 88],
        '1S': [65, 0],
        '1T': [65, 88],
        '2A': [145, 0],
        '2B': [145, 34.5],
        '2C': [145, 37],
        '2D': [145, 51],
        '2E': [145, 53.5],
        '2F': [145, 88],
        '2G': [140.5, 37],
        '2H': [140.5, 51],
        '2I': [132, 0,],
        '2J': [132, 34.5],
        '2K': [132, 53.5],
        '2L': [132, 88],
        '2M': [124, 0],
        '2N': [124, 34.5],
        '2O': [124, 53.5],
        '2P': [124, 88],
        '2Q': [100, 0],
        '2R': [100, 88],
        '2S': [80, 0],
        '2T': [80, 88],
        '1GPA': [0, 40.75],
        '1GPB': [0, 47.25],
        '2GPA': [145, 40.75],
        '2GPB': [145, 47.25],
    }
    return dst

dst = get_dst_points()
df = pd.DataFrame.from_dict(dst, orient='index', columns=['X', 'Y']).reset_index()
df.columns = ['Key', 'X', 'Y']

data = [
    [53.175, 66.832],
    [59.266, 65.076],
    [55.636, 68.176],
    [98.25, 47.736],
    [61.115, 64.914],
    [54.664, 66.673],
    [78.774, 56.425],
    [77.538, 56.888],
    [57.224, 67.958],
    [69.738, 82.662],
    [72.528, 65.298],
    [57.784, 60.483],
    [84.81, 90.013],
    [71.909, 18.696]
]
df1 = pd.DataFrame(data, columns=['X', 'Y'])

# Assign row names
player_names = [f'player{i+1}' for i in range(len(data))]
df1['Key'] = player_names

# Reorder columns
df1 = df1[['Key', 'X', 'Y']]


# Plotting
plt.figure(figsize=(8, 6))
plt.xlim(0, 145)
plt.ylim(0, 88)
plt.scatter(df['X'], df['Y'], color='blue', label='keypoints')
plt.scatter(df1['X'], df1['Y'], color='red', label='players')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Player and Key points with Key Annotations')
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

def get_perspective_transform(src, dst):
    """Get the homography matrix between src and dst

    Arguments:
        src: np.array of shape (B,X,2) or (X,2), the X>3 original points per image
        dst: np.array of shape (B,X,2) or (X,2), the X>3 corresponding points per image
    Returns:
        M: np.array of shape (B,3,3) or (3,3), each homography per image
    Raises:

    """
    if len(src.shape) == 2:
        print('first')
        M, _ = cv2.findHomography(src, dst, method=0,)
    else:
        M = []
        for src_, dst_ in zip(src, dst):
            M.append(cv2.findHomography(src_, dst_, cv2.RANSAC, 5)[0])
        M = np.array(M)
    return M