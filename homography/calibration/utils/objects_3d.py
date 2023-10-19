import numpy as np
import torch


class pitch3D:
    def __init__(self):
        """
        Initialize 2D coordinates of the pitch
        """
        self.zero0 = np.array([0, 0, 0], dtype="float")
        self.zeroA = np.array([0, -5, 0], dtype="float")
        self.zeroB = np.array([0, 5, 0], dtype="float")
        self.oneA = np.array([-72.5, -44, 0], dtype="float")
        self.oneB = np.array([-72.5, -9.5, 0], dtype="float")
        self.oneC = np.array([-72.5, -7, 0], dtype="float")
        self.oneD = np.array([-72.5, 7, 0], dtype="float")
        self.oneE = np.array([-72.5, 9.5, 0], dtype="float")
        self.oneF = np.array([-72.5, 44, 0], dtype="float")
        self.oneG = np.array([-68, -7, 0], dtype="float")
        self.oneH = np.array([-68, 7, 0], dtype="float")
        self.oneI = np.array([-59.5, -44, 0], dtype="float")
        self.oneJ = np.array([-59.5, -9.5, 0], dtype="float")
        self.oneK = np.array([-59.5, 9.5, 0], dtype="float")
        self.oneL = np.array([-59.5, 44, 0], dtype="float")
        self.oneM = np.array([-51.5, -44, 0], dtype="float")
        self.oneN = np.array([-51.5, -9.5, 0], dtype="float")
        self.oneO = np.array([-51.5, 9.5, 0], dtype="float")
        self.oneP = np.array([-51.5, 44, 0], dtype="float")
        self.oneQ = np.array([-27.5, -44, 0], dtype="float")
        self.oneR = np.array([-27.5, 44, 0], dtype="float")
        self.oneS = np.array([-7.5, -44, 0], dtype="float")
        self.oneT = np.array([-7.5, 44, 0], dtype="float")
        self.twoA = np.array([72.5, -44, 0], dtype="float")
        self.twoB = np.array([72.5, -9.5, 0], dtype="float")
        self.twoC = np.array([72.5, -7, 0], dtype="float")
        self.twoD = np.array([72.5, 7, 0], dtype="float")
        self.twoE = np.array([72.5, 9.5, 0], dtype="float")
        self.twoF = np.array([72.5, 44, 0], dtype="float")
        self.twoG = np.array([68, -7, 0], dtype="float")
        self.twoH = np.array([68, 7, 0], dtype="float")
        self.twoI = np.array([59.5, -44, 0], dtype="float")
        self.twoJ = np.array([59.5, -9.5, 0], dtype="float")
        self.twoK = np.array([59.5, 9.5, 0], dtype="float")
        self.twoL = np.array([59.5, 44, 0], dtype="float")
        self.twoM = np.array([51.5, -44, 0], dtype="float")
        self.twoN = np.array([51.5, -9.5, 0], dtype="float")
        self.twoO = np.array([51.5, 9.5, 0], dtype="float")
        self.twoP = np.array([51.5, 44, 0], dtype="float")
        self.twoQ = np.array([27.5, -44, 0], dtype="float")
        self.twoR = np.array([27.5, 44, 0], dtype="float")
        self.twoS = np.array([7.5, -44, 0], dtype="float")
        self.twoT = np.array([7.5, 55, 0], dtype="float")
        self.oneGPA = np.array([-72.5, -3.25, 0], dtype="float")
        self.oneGPB = np.array([-72.5, 3.25, 0], dtype="float")
        self.twoGPA = np.array([72.5, -3.25, 0], dtype="float")
        self.twoGPB = np.array([72.5, 3.25, 0], dtype="float")

        self.keypoints = [self.zero0, self.zeroB, self.zero0,
                          self.oneA, self.oneB, self.oneC, self.oneD, self.oneE, self.oneF, self.oneG, self.oneH,
                          self.oneI, self.oneJ, self.oneK, self.oneL, self.oneM, self.oneN, self.oneO, self.oneP,
                          self.oneQ, self.oneR, self.oneS, self.oneT,
                          self.twoA, self.twoB, self.twoC, self.twoD, self.twoE, self.twoF, self.twoG, self.twoH,
                          self.twoI, self.twoJ, self.twoK, self.twoL, self.twoM, self.twoN, self.twoO, self.twoP,
                          self.twoQ, self.twoR, self.twoS, self.twoT,
                          self.oneGPA, self.oneGPB, self.twoGPA, self.twoGPB
        ]
        self.keypoint_names = [attr[4:] for attr in dir(self) if isinstance(getattr(self, attr), np.ndarray) and attr.startswith("self")]
        print(f"Key Point Names: {self.keypoint_names}")

class Meshgrid():
    def __init__(self, pitch3D):
        self.keypoints = pitch3D.keypoints
        self.points = torch.tensor(self.keypoints)


