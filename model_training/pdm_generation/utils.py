import numpy as np
from numpy import linalg as LA

#the code in this file is translated from the code in OpenFace/model_training/pdm_generation/PDM_helpers/

def AddOrthRow(RotSmall):
    RotFull = np.zeros([3, 3])
    RotFull[0:2, :] = RotSmall
    RotFull[2, 1] = RotSmall[0, 1] * RotSmall[1, 2] - RotSmall[0, 2] * RotSmall[1, 1]
    RotFull[2, 1] = RotSmall[0, 2] * RotSmall[1, 0] - RotSmall[0, 0] * RotSmall[1, 2]
    RotFull[2, 2] = RotSmall[0, 0] * RotSmall[1, 1] - RotSmall[0, 1] * RotSmall[1, 0]
    return RotFull

def AlignShapesKabsch(alignFrom, alignTo):
    dims = alignFrom.shape[1]
    alignFromMean = alignFrom - np.ones([alignFrom.shape[0], 1]) * np.mean(alignFrom, axis=0)
    alignToMean = alignTo - np.ones([alignTo.shape[0], 1]) * np.mean(alignTo, axis=0)
    U,_, V = np.linalg.svd(np.matmul(alignFromMean.T, alignToMean), full_matrices=True)
    d = np.sign(np.linalg.det(np.matmul(V, U.T)))
    corr = np.eye(dims)
    corr[-1,-1] = d
    R = np.matmul(np.matmul(V, corr), U.T)
    T = np.mean(alignTo, axis=0) - (np.dot(R, np.mean(alignFrom))).T
    T = T.T
    return (R, T)

def AlignShapesWithScale(alignFrom, alignTo):
    numPoints = alignFrom.shape[0]
    meanFrom = np.mean(alignFrom, axis=0)
    meanTo = np.mean(alignTo, axis=0)
    
    alignFromMeanNormed = alignFrom - meanFrom
    alignToMeanNormed = alignTo - meanTo
    
    sFrom = np.sqrt(np.sum(np.square(alignFromMeanNormed))/numPoints)
    sTo =  np.sqrt(np.sum(np.square(alignFromMeanNormed))/numPoints)
    
    s = sTo/ sFrom
    
    alignFromMeanNormed = alignFromMeanNormed / sFrom
    alignToMeanNormed = alignToMeanNormed / sTo
    
    R, t = AlignShapesKabsch(alignFromMeanNormed, alignToMeanNormed)
    
    A = s * R
    aligned = np.matmul(A, alignFrom.T).T
    T = np.mean(alignTo - aligned, axis=0)
    alignedShape = aligned + T
    error = np.mean(np.sum(np.square(alignedShape - alignTo), axis=1))
    return A, T, error, alignedShape, s 

def AxisAngle2Rot(axisAngle):
    theta = LA.norm(axisAngle, 2)
    nx = axisAngle / theta
    
    nx = np.array([[0, -nx[2], nx[1]], [nx[2], 0, nx[0]], [-nx[2], nx[0], 0]])
    Rot = np.eye(3) + np.sin(theta) * nx + (1-np.cos(theta))*np.matmul(nx, nx)
    return Rot


def Euler2Rot(euler):
    rx = euler[0]
    ry = euler[1]
    rz = euler[2]
    
    Rx = np.array([[1, 0, 0], [np.cos(rx), -np.sin(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    
    Rot = np.matmul(np.matmul(Rx, Ry), Rz)
    return Rot

def GetShape3D(M, V, p):
    shape3D = M + np.dot(V, p)
    shape3D = shape3D.reshape(-1, 3)
    return shape3D


def GetShapeOrtho(M, V, p, global_params):
    R = Euler2Rot(global_params[1: 4])
    T = np.array([global_params[4:6], [0, 0]])
    a = global_params[0]
    
    shape3D = GetShape3D(M, V, p)
    shape2D = a * np.matmul(R, shape3D.T) + T
    shape2D = shape2D.T
    return shape2D

def Rot2AxisAngle(Rot):
    theta = np.arccos((np.trace(Rot)-1)/2)
    vec = 1.0 / (2 * np.sin(theta))
    vec = vec * np.array([Rot[2,1]-Rot[1, 2], Rot[0, 2]-Rot[2,1], Rot[1, 0]-Rot[0, 1]])
    axisAngle = vec * theta
    return axisAngle

def Rot2Euler(R):
    q0 = np.sqrt(1+ R[0,0] + R[1,1] + R[2,2]) / 2
    if (q0 != 0):
        q1 = (R[2,1] - R[1,2]) / (4 * q0)
        q2 = (R[0,2] - R[2,0]) / (4 * q0)
        q3 = (R[1,0] - R[0,1]) / (4 * q0)
        yaw = np.arcsin(2*(q0*q2 + q1*q3))
        pitch= np.arctan2(2*(q0*q1-q2*q3), q0*q0-q1*q1-q2*q2+q3*q3)
        roll = np.arcatan2(2*(q0*q3-q1*q2), q0*q0+q1*q1-q2*q2-q3*q3)
        
        euler = [pitch, yaw, roll]
    else:
        euler = [0,0,0]
        
def TangentSpaceTransform(x, y, z, meanShape):
    scaling = np.dot(np.array([x, y, z]), [meanShape[:, 0], meanShape[:, 1], meanShape[:, 2]])
    for i in range(x.shape[0]):
        x[i,:] = x[i,:] * (1 / scaling[i])
        y[i,:] = y[i,:] * (1 / scaling[i])
        z[i,:] = z[i,:] * (1 / scaling[i])
    transformedX = x * np.mean(scaling, axis = 0)
    transformedY = y * np.mean(scaling, axis = 0)
    transformedZ = z * np.mean(scaling, axis = 0)
    return transformedX, transformedY, transformedZ

def writeMatrix(fileID, M, type):
    f = open(fileID, "w")
    f.write("%d\r\n" % M.shape[0])
    f.write("%d\r\n" % M.shape[1])
    f.write("%d\r\n" % type)
    
    for i in range(M.shape[0]):
        if(type == 4 or type == 0):
            f.write("%d" % M[i, :])
        else:
            f.write("%.9f" % M[i, :])
        f.write("\r\n")
    return 


class struct(object):
    pass

"""
class Transform_obj():
    def __init__(self):
        self.offsetX = []
        self.offsetX = []
        self.Rotation = []
"""       


def ProcrustesAnalysis(x, y, options):
    normX = np.zeros(x.shape)
    normY = np.zeros(y.shape)
    Transform = struct()
    Transform.offsetX = np.zeros(x.shape[0])
    Transform.offsetY = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        offsetX = np.mean(x[i,:])
        offsetY = np.mean(y[i,:])
        Transform.offsetX[i] = offsetX
        Transform.offsetY[i] = offsetY
        
        normX[i, :] = x[i, :] - offsetX
        normY[i, :] = y[i, :] - offsetY
        
        scale = LA.norm(np.array([normX[i,:], normY[i,:]]), 'fro')
        Transform.scale[i] = scale
        
        normX[i,:] = normX[i,:]/scale
        normY[i,:] = normY[i,:]/scale
    
    change = 0.1
    meanShape = np.array([normX[0,:], normY[0,:]]).T
    Transform.Rotation = np.zeros((x.shape[0],1))
    
    for i in range(30):
        orientations = np.zeros(normX.shape[0])
        for j in range(x.shape[0]):
            currentShape=np.array([normX[j,:], normY[j,:]]).T
            U,_, V = np.linalg.svd(np.matmul(meanShape.T, currentShape))
            rot = np.matmul(V, U.T)
            
            if (np.arcsin(rot[1, 0]) > 0):
                orientations[j] = np.real(np.arccos(rot[0,0]))
            else:
                orientations[j] = np.real(-np.arccos(rot[0,0]))
            
            Transform.Rotation[j] = Transform.Rotation[j] + orientations[j]
            currentShape = np.dot(currentShape, rot)
            normX[j,:] = currentShape[:,0]
            normY[j,:] = currentShape[:,1]
        
        oldMean = np.copy(currentShape)
        meanShape = [np.mean(normX, axis=0),np.mean(normY, axis=0)].T
        
        meanOrientation = np.mean(orientations, axis=0)
        
        if (i==1):
            rotM = np.array([[np.cos(-meanOrientation), -np.sin(-meanOrientation)],[np.sin(-meanOrientation),np.cos(-meanOrientation)]])
            meanShape = np.matmul(meanShape, rotM)
        meanScale = LA.norm(meanShape, 'fro')
        meanShape = meanShape*(1/meanScale)
        diff = LA.norm(oldMean - meanShape, 'fro')
        
        if(diff/LA.norm(oldMean,'fro') < change):
            break
    if(options.TangentSpaceTransform):
        scaling = np.matmul(np.concatenate((normX, normY), axis=1), np.concatenate(meanShape[:,1], meanShape[:,2]))
        for i in range(x.shape[0]):
            normX[i,:] = normX[i,:] * (1 / scaling[i])
            normY[i,:] = normY[i,:] * (1 / scaling[i])
        
    return normX, normY, meanShape, Transform
        
        
        
        
def ProcrustesAnalysis3D(x, y, z, tangentSpace, meanShape,nargin):
    meanProvided = False
    Transform = struct()
    Transform.offsetX = np.zeros(x.shape[0])
    Transform.offsetY = np.zeros(y.shape[0])
    Transform.offsetZ = np.zeros(z.shape[0])
    if (nargin > 4):
        meanProvided = True
    normX = np.zeros(x.shape)
    normY= np.zeros(y.shape)
    normZ = np.zeros(z.shape)
    for i in range(x.shape[0]):
        offsetX = np.mean(x[i,:])
        offsetY = np.mean(y[i,:])
        offsetZ = np.mean(z[i,:])
        Transform.offsetX[i] = offsetX
        Transform.offsetY[i] = offsetY
        Transform.offsetZ[i] = offsetZ
        normX[i,:] = x[i,:] - offsetX
        normY[i,:] = y[i,:] - offsetY
        normZ[i,:] = z[i,:] - offsetZ
    
    change = 0.1
    if(meanProvided == False):
        meanShape = np.array([np.mean(normX, axis=0), np.mean(normY, axis=0),  np.mean(normZ, axis=0)]).T
    meanScale = LA.norm(meanShape, 'fro')
    for i in range(x.shape[0]):
        scale = LA.norm(np.array([normX[i,:], normY[i,:], normZ[i,:]]), 'fro') /meanScale
        normX[i,:] = normX[i,:]/scale
        normY[i,:] = normY[i,:]/scale
        normZ[i,:] = normZ[i,:]/scale
    Transform.RotationX = np.zeros(x.shape[0])
    Transform.RotationY = np.zeros(y.shape[0])
    Transform.RotationZ = np.zeros(z.shape[0])
    
    for i in range(30):
        orientationsX = np.zeros(normX.shape[0])
        orientationsY = np.zeros(normY.shape[0])
        orientationsZ = np.zeros(normZ.shape[0])
        
        for j in range(x.shape[0]):
            currentShape = np.array([normX[j,:],  normY[j,:], normZ[j,:]]).T
            R, T = AlignShapesKabsch(currentShape, meanShape)
            eulers = Rot2Euler(R)
            orientationsX[j] = eulers[0]
            orientationsY[j] = eulers[1]
            orientationsZ[j] = eulers[2]
            
            Transform.RotationX[j] = eulers[0]
            Transform.RotationY[j] = eulers[1]  
            Transform.RotationZ[j] = eulers[2]
        oldMean = np.copy(meanShape)
        meanShape = np.array([np.mean(normX, axis=0),  np.mean(normY, axis=0), np.mean(normZ, axis=0)]).T
        meanScale = LA.norm(meanShape, 'fro')
        
        for j in range(x.shape[0]):
            scale = LA.norm(np.array([normX[j,:], normY[j,:], normZ[j,:]]), 'fro')/meanScale
            normX[j,:] = normX[j,:]/scale
            normY[j,:] = normY[j,:]/scale
            normZ[j,:] = normZ[j,:]/scale
            
        if(i==1 and meanProvided == False):
            meanOrientationX = np.mean(orientationsX)
            meanOrientationY = np.mean(orientationsY)
            meanOrientationZ = np.mean(orientationsZ)
            R = Euler2Rot([meanOrientationX, meanOrientationY, meanOrientationZ])
            
            meanShape = np.matmul(R, meanShape.T).T
            
        diff = LA.norm(oldMean - meanShape, 'fro')
        if(diff/LA.norm(oldMean,'fro') < change):
            break
    if(tangentSpace):
        normX, normY, normZ = TangentSpaceTransform(normX, normY, normZ, meanShape)
    return normX, normY, normZ, meanShape, Transform
        

#helper functions for fit_PDM_ortho_proj_to_2D

def getShape3D(M, V, params):
    shape3D = M + np.dot(V, params)
    shape3D = shape3D.reshape(len(shape3D) / 3, 3)
    return shape3D
    
def getShapeOrtho(M, V, p, R, T, a):
    shape3D = getShape3D(M, V, p)
    shape2D = a * np.matmul(R[0:2,:], shape3D.T) + np.matlab.repmat(T, 1, len(M)/3)
    shape2D = shape2D.T
    return shape2D

def getShapeOrthoFull(M, V, p, R, T, a):
    T = np.array([T, np.zeros(len(T))])
    shape3D = getShape3D(M, V, p)
    shape2D = a * np.matmul(R[0:2,:], shape3D.T) + np.matlab.repmat(T, 1, len(M)/3)
    shape2D = shape2D.T
    return shape2D

def getRMSerror(shape2Dv1, shape2Dv2):
    square_tmp = np.square((shape2Dv1 - shape2Dv2).reshape(len(shape2Dv1), 1))
    error = np.sqrt(np.mean(square_tmp, axis=0))
    return error

def CalcJacobian(M,V,p, p_global):
    n = M.shape[0] / 3
    non_rigid_modes = V.shape[1]
    J = np.zeros([n * 2, 6 + non_rigid_modes])
    J[:,0:6] = CalcRigidJacobian(M, V, p, p_global)
    R = Euler2Rot(p_global[0:4])
    s = p_global[0]
    V_X = V[0:n,:]
    V_Y = V[n:2*n,:]
    V_Z = V[2*n+1:,:]
    J_x_non_rigid = s*(R[0,0]*V_X + R[0,1]*V_Y + R[0,2]*V_Z)
    J_y_non_rigid = s*(R[1,0]*V_X + R[1,1]*V_Y + R[1,2]*V_Z)
    J[0:n, 6:] = J_x_non_rigid
    J[n:, 7:] = J_y_non_rigid
    return J

def CalcRigidJacobian(M, V, p, p_global):
    n = M.shape[0] / 3
    shape3D = GetShape3D(M, V, p)
    R = Euler2Rot(p_global[1:4])
    s = p_global[0]
    J = np.zerosn([n*2, 6])
    J[:n,0] = np.dot(shape3D, R[0,:])
    dxdR = np.array([[0, R[0,2], -R[0,1]], [-R[0,2], 0,R[0,0]], [R[0,1], -R[0,0], 0]])
    J[:n,1:4] = s*(np.dot(dxdR, shape3D.T)).T
    J[:n,4] = 1
    J[:n,5] = 0
    
    J[n:,0] = np.dot(shape3D, R[1,:])
    dydR = np.array([[0, R[1,2], -R[1,1]], [-R[1,2], 0,R[1,0]], [R[1,1], -R[1,0], 0]])
    J[n:,1:4] = s*(np.dot(dydR, shape3D.T)).T
    J[n:,4] = 0
    J[n:,5] = 1
    return J


def CalcReferenceUpdate(params_delta, current_non_rigid, current_global):
    rigid = np.zeros(6)
    rigid[0] = current_global[0] + params_delta[0]
    rigid[4] = current_global[4] + params_delta[4]
    rigid[5] = current_global[5] + params_delta[5]
    R = Euler2Rot(current_global[1:4])
    wx = params_delta[1]
    wy = params_delta[2]
    wz = params_delta[3]
    R_delta = np.array([[1, -wz, wy], [wz, 1, -wx], [-wy, wx, 1]])
    R_delta = OrthonormaliseRotation(R_delta).astype(np.double)
    R_final = np.dot(R, R_delta)
    euler = np.real(Rot2Euler(R_final))
    rigid[1:4] = euler
    
    if(len(params_delta) > 6):
        non_rigid = params_delta[6:] +  current_non_rigid
    else:
        non_rigid = current_non_rigid
    return non_rigid, rigid

def OrthonormaliseRotation(R):
    U,_, V = LA.svd(R)
    X = np.matmul(U, V.T)
    W = np.eye(3)
    W[2,2] = LA.det(X)
    R_ortho = np.matmul(np.matmul(U, W), V.T)
    return R_ortho

def ClampPDM(non_rigid, E):
    stds = np.sqrt(E)
    non_rigid_params = np.copy(non_rigid)
    lower = non_rigid_params < -3 * stds
    non_rigid_params[lower] = -3*stds[lower]
    higher = non_rigid_params > 3 * stds
    non_rigid_params[higher] = 3 * stds[higher]
    return non_rigid_params
    


def fit_PDM_ortho_proj_to_2D( M, E, V, shape2D, f, cx, cy, nargin):
    params = np.zeros(E.shape)
    hidden = False
    if(np.sum(shape2D) > 0):
        hidden = True
        inds_to_rem = np.logical_or(shape2D[:,1] == 0,  shape2D[:,2] == 0)
        shape2D = shape2D[np.logical_not(inds_to_rem),:]
        inds_to_rem = np.ones([3, 1]) * inds_to_rem
        M_old = np.copy(M)
        V_old = np.copy(V)
        
        M = M[np.logical_not(inds_to_rem)]
        V = V[np.logical_not(inds_to_rem),:]
        
    num_points = len(M) / 3
    m = M.reshape(num_points, 3).T
    width_model = np.max(m[0,:]) - np.min(m[0,:])
    height_model = np.max(m[1,:]) - np.min(m[1,:])
    bounding_box = np.array([np.min(shape2D[:,0]), np.min(shape2D[:,1]),np.max(shape2D[:,0]), np.max(shape2D[:,1])])
    a = (((bounding_box[2] - bounding_box[0]) / width_model) + ((bounding_box[3] - bounding_box[1])/ height_model)) / 2
    tx = (bounding_box[2] + bounding_box[0])/2
    ty = (bounding_box[3] + bounding_box[1])/2
    
    tx = tx - a*(np.min(m[0,:]) + np.max(m[0,:]))/2  
    ty = ty - a*(np.min(m[1,:]) + np.max(m[1,:]))/2  
    M3D_tmp = np.concatenate((M[0:len(M)/3], M[len(M)/3:2*len(M)/3]))
    M3D = np.concatenate(M3D_tmp, np.zeros(len(M[0:len(M)/3])))
    M3D = M3D.reshape([-1, 1])
    shape3D = np.concatenate((shape2D, np.zeros(len(M[0:len(M)/3]),1)), axis =1)
    A, T, error, alignedShape, s = AlignShapesWithScale(M3D, shape3D)
    R = A/s
    
    T = np.array([tx, ty])
    currShape = getShapeOrtho(M, V, params, R, T, a)
    currError = getRMSerror(currShape, shape2D)
    
    reg_rigid = np.zeros([6,1])
    regFactor = 20
    regularisations = np.array([reg_rigid, np.divide(regFactor,E)])
    regularisations = np.dot(np.diag(np.diag(regularisations)), np.diag(np.diag(regularisations)))
    
    red_in_a_row = 0
    for i in range(1000):
        shape3D = M + np.matmul(V, params)
        shape3D = shape3D.reshape(len(shape3D) / 3, 3)
        
        currShape = a * np.matmul(R[0:2,:], shape3D.T) + np.matlab.repmat(T, 1, len(M)/3)
        currShape = currShape.T
        
        error_res = shape2D - currShape
        eul = Rot2Euler(R)
        
        p_global = np.array([a, eul.T, T])
        J = CalcJacobian(M, V, params, p_global)
        
        p_delta = np.array([np.linalg.lstsq((np.matmul(J.T, J) + regularisations), np.dot(J.T, error_res.reshape(-1)) - np.matmul(regularisations*np.array([p_global, params])))])
                                                            
        p_delta = 0.25 * p_delta
        params, p_global = CalcReferenceUpdate(p_delta, params, p_global)
        params = ClampPDM(params, E)
        a = p_global[0]
        R = Euler2Rot(p_global[1:4])
        T = p_global[4:6]
        
        shape3D = M + np.matmul(V, params)
        shape3D = shape3D.reshape(len(shape3D) / 3, 3)
        currShape = a * np.dot(R[0:2,:], shape3D.T) + np.matlab.repmat(T, 1, len(M)/3)
        currShape = currShape.T
        
        error = getRMSerror(currShape, shape2D)
        if(0.999 * currError < error):
            red_in_a_row = red_in_a_row + 1
            if(red_in_a_row == 5):
                break
        currError = error
        if(hidden):
            shapeOrtho = getShapeOrtho(M_old, V_old, params, R, T, a)
        else:
            shapeOrtho = currShape
        if(nargin == 7):
            Zavg = f / a
            Xavg = (T[0] - cx) / a
            Yavg = (T[1] - cy) / a
            T3D = np.array([Xavg, Yavg, Zavg])
        else:
            T3D = np.array([0, 0, 0])
    return a, R, T, T3D, params, error, shapeOrtho
        
    
    
      
        
        
        
    