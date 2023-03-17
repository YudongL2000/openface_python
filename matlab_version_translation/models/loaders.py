import numpy as np
import os
from scipy.io import loadmat
import glob

class struct(object):
    pass

def Load_CECLM_Patch_Experts(col_patch_dir, col_patch_file):
    colourPatchFiles = []
    for file in glob.glob(os.path.join(col_patch_dir + col_patch_file), recursive=True):
        col_patch_file.append(file)
    patches = []
    for i in range(len(colourPatchFiles)):
        file_path = os.path.join(col_patch_dir, colourPatchFiles[i])
        file_dict = loadmat(file_path)
        patch = struct()
        patch.centers = file_dict["centers"]
        patch.trainingScale = file_dict["trainingScale"]
        patch.visibilities = file_dict["visiIndex"] 
        patch.patch_experts = file_dict["patch_experts"].patch_experts
        patch.correlations = file_dict["patch_experts"].correlations
        patch.rms_errors = file_dict["patch_experts"].rms_errors
        patch.modalities = file_dict["patch_experts"].types
        patch.multi_modal_types = file_dict["patch_experts"].types
        patch.type = 'CEN'
        patch.normalisationOptionsCol = file_dict["normalisationOptions"]        
        patches.append(patch)
    if (len(patches) == 0):
        print("Could not find CEN patch experts, for instructions of how to download them, please visit - https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download\n")
        raise Exception('cenExperts:modelError: Could not find CEN model at location %s, see - https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download' % os.path.join(col_patch_dir, col_patch_file))
    
    return patches


def Load_CECLM_general():
    patches = Load_CECLM_Patch_Experts( '../models/cen/', 'cen_patches_*_general.mat')
    pdmLoc = '../models/pdm/pdm_68_aligned_wild.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm = struct()
    pdm.M = np.array(pdmLoc_dict["M"]).astype(np.double)
    pdm.E = np.array(pdmLoc_dict["E"]).astype(np.double)
    pdm.V = np.array(pdmLoc_dict["V"]).astype(np.double)
    
    clmParams = struct()
    clmParams.window_size = np.array([[25,25], [23,23], [21,21], [21,21]])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    clmParams.regFactor = 0.9*np.array([35, 27, 20, 20])
    clmParams.sigmaMeanShift = 1.5*np.array([1.25, 1.375, 1.5, 1.5])
    clmParams.tikhonov_factor = np.array([2.5, 5, 7.5, 7.5])
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    clmParams.multi_modal_types  = patches[0].multi_modal_types
    clmParams.numPatchIters = 4
    
    cen_general_data = loadmat('../models/cen/cen_general_mapping.mat')
    early_term_params = cen_general_data["early_term_params"]
    return patches, pdm, clmParams, early_term_params


def Load_CECLM_menpo():
    patches = Load_CECLM_Patch_Experts( '../models/cen/', 'cen_patches_*_general.mat')
    pdmLoc = '../models/pdm/pdm_68_aligned_menpo.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm = struct()
    pdm.M = np.array(pdmLoc_dict["M"]).astype(np.double)
    pdm.E = np.array(pdmLoc_dict["E"]).astype(np.double)
    pdm.V = np.array(pdmLoc_dict["V"]).astype(np.double)
    
    clmParams = struct()
    clmParams.window_size = np.array([[25,25], [23,23], [21,21], [21,21]])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    clmParams.regFactor = 0.9*np.array([35, 27, 20, 20])
    clmParams.sigmaMeanShift = 1.5*np.array([1.25, 1.375, 1.5, 1.5])
    clmParams.tikhonov_factor = np.array([2.5, 5, 7.5, 7.5])
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    clmParams.multi_modal_types  = patches[0].multi_modal_types
    clmParams.numPatchIters = 4
    
    cen_general_data = loadmat('../models/cen/cen_general_mapping.mat')
    early_term_params = cen_general_data["early_term_params"]
    return patches, pdm, clmParams, early_term_params
    

def Load_CLM_general():    
    clmParams = struct()
    clmParams.window_size = np.array([[25,25], [23,23], [21,21], [21,21]])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    patches = Load_Patch_Experts('../models/wild/', 'svr_patches_*_wild.mat', [], [], clmParams)
    
    
    pdmLoc = '../models/pdm/pdm_68_aligned_wild.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm = struct()
    pdm.M = np.array(pdmLoc_dict["M"]).astype(np.double)
    pdm.E = np.array(pdmLoc_dict["E"]).astype(np.double)
    pdm.V = np.array(pdmLoc_dict["V"]).astype(np.double)
    
    
    clmParams.regFactor = 0.9*np.array([35, 27, 20, 20])
    clmParams.sigmaMeanShift = 1.5*np.array([1.25, 1.375, 1.5, 1.5])
    clmParams.tikhonov_factor = np.array([2.5, 5, 7.5, 7.5])
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    clmParams.multi_modal_types  = patches[0].multi_modal_types
    return patches, pdm, clmParams

def Load_CLM_params_eye():
    clmParams = struct()
    clmParams.window_size = np.array(
        [[15, 15], [13, 13]]
    )
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    pdmLoc = '../models/hierarch_pdm/pdm_6_r_eye.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm_right = struct()
    pdm_right.M = pdmLoc_dict["M"].astype(np.double)
    pdm_right.E = pdmLoc_dict["E"].astype(np.double)
    pdm_right.V = pdmLoc_dict["V"].astype(np.double)
    
    pdmLoc = '../models/hierarch_pdm/pdm_6_l_eye.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm_left = struct()
    pdm_left.M = pdmLoc_dict["M"].astype(np.double)
    pdm_left.E = pdmLoc_dict["E"].astype(np.double)
    pdm_left.V = pdmLoc_dict["V"].astype(np.double)
    clmParams.regFactor = 0.1
    clmParams.sigmaMeanShift = 2
    clmParams.tikhonov_factor = 0
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 5
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    clmParams.tikhonov_factor = 0
    
    return clmParams, pdm_right, pdm_left


def Load_CLM_params_eye_28():
    clmParams = struct()
    clmParams.window_size = np.array(
        [[17, 17], [15, 15], [13, 13]]
    )
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    pdmLoc = '../models/hierarch_pdm/pdm_28_r_eye.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm_right = struct()
    pdm_right.M = pdmLoc_dict["M"].astype(np.double)
    pdm_right.E = pdmLoc_dict["E"].astype(np.double)
    pdm_right.V = pdmLoc_dict["V"].astype(np.double)
    
    pdmLoc = '../models/hierarch_pdm/pdm_28_l_eye.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm_left = struct()
    pdm_left.M = pdmLoc_dict["M"].astype(np.double)
    pdm_left.E = pdmLoc_dict["E"].astype(np.double)
    pdm_left.V = pdmLoc_dict["V"].astype(np.double)
    clmParams.regFactor = 2.0
    clmParams.sigmaMeanShift = 1.5
    clmParams.tikhonov_factor = 0
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 0
    clmParams.tikhonov_factor = 0
    
    return clmParams, pdm_right, pdm_left

def Load_CLM_params_inner():
    clmParams = struct()
    clmParams.window_size = np.array([19, 19])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    pdmLoc = '../models/hierarch_pdm/pdm_51_inner.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm_mouth = struct()
    pdm_mouth.M = pdmLoc_dict["M"].astype(np.double)
    pdm_mouth.E = pdmLoc_dict["E"].astype(np.double)
    pdm_mouth.V = pdmLoc_dict["V"].astype(np.double)
    
    clmParams.regFactor = 2.5
    clmParams.sigmaMeanShift = 1.75
    clmParams.tikhonov_factor = 2.5
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 5
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    
    return clmParams, pdm_mouth


def Load_CLM_params_vid():
    clmParams = struct()
    clmParams.window_size = np.array([[21,21], [19,19], [17,17]])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    pdmLoc = '../models/pdm/pdm_68_multi_pie.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm = struct()
    pdm.M = pdmLoc_dict["M"].astype(np.double)
    pdm.E = pdmLoc_dict["E"].astype(np.double)
    pdm.V = pdmLoc_dict["V"].astype(np.double)
    
    clmParams.regFactor = 25
    clmParams.sigmaMeanShift = 2
    clmParams.tikhonov_factor = 5
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    
    return clmParams, pdm

def Load_CLM_params_wild():
    clmParams = struct()
    clmParams.window_size = np.array([[25,25], [25,25], [25,25]])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    pdmLoc = '../models/pdm/pdm_68_aligned_wild.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm = struct()
    pdm.M = pdmLoc_dict["M"].astype(np.double)
    pdm.E = pdmLoc_dict["E"].astype(np.double)
    pdm.V = pdmLoc_dict["V"].astype(np.double)
    
    clmParams.regFactor = 25
    clmParams.sigmaMeanShift = 2
    clmParams.tikhonov_factor = 5
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    
    return clmParams, pdm

def Load_CLM_wild():
    clmParams = struct()
    clmParams.window_size = np.array([[25,25], [23,23], [21,21]])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    patches = Load_Patch_Experts('../models/general/', 'svr_patches_*_wild.mat', [], [], clmParams)
    pdmLoc = '../models/pdm/pdm_68_aligned_wild.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm = struct()
    pdm.M = pdmLoc_dict["M"].astype(np.double)
    pdm.E = pdmLoc_dict["E"].astype(np.double)
    pdm.V = pdmLoc_dict["V"].astype(np.double)
    
    clmParams.regFactor = np.array([35, 27, 20, 5])
    clmParams.sigmaMeanShift = np.array([1.25, 1.375, 1.5, 1.75])
    clmParams.tikhonov_factor = np.array([2.5, 5, 7.5, 12.5])
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    clmParams.multi_modal_types  = patches[0].multi_modal_types
    return patches, pdm, clmParams


def Load_CLNF_general():
    clmParams = struct()
    clmParams.window_size = np.array([[25,25], [23,23], [21,21]])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    patches = Load_Patch_Experts('../models/general/', 'svr_patches_*_general.mat', [], [], clmParams)
    pdmLoc = '../models/pdm/pdm_68_aligned_wild.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm = struct()
    pdm.M = pdmLoc_dict["M"].astype(np.double)
    pdm.E = pdmLoc_dict["E"].astype(np.double)
    pdm.V = pdmLoc_dict["V"].astype(np.double)
    
    clmParams.regFactor = np.array([35, 27, 20, 5])
    clmParams.sigmaMeanShift = np.array([1.25, 1.375, 1.5, 1.75])
    clmParams.tikhonov_factor = np.array([2.5, 5, 7.5, 12.5])
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    clmParams.multi_modal_types  = patches[0].multi_modal_types
    return patches, pdm, clmParams


def Load_CLNF_inner():
    clmParams = struct()
    clmParams.window_size = np.array([19, 19])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    pdmLoc = '../models/hierarch_pdm/pdm_51_inner.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    pdm = struct()
    pdm.M = pdmLoc_dict["M"].astype(np.double)
    pdm.E = pdmLoc_dict["E"].astype(np.double)
    pdm.V = pdmLoc_dict["V"].astype(np.double)
    
    clmParams.regFactor = 2.5
    clmParams.sigmaMeanShift = 1.75
    clmParams.tikhonov_factor = 2.5
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 5
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    
    patches = Load_Patch_Experts('../models/general/', 'ccnf_patches_*general_no_out.mat', [], [], clmParams)
    clmParams.multi_modal_types  = patches[0].multi_modal_types
    inds_full = list(range(17, 68))
    inds_inner = list(range(0, 51))
    return patches, pdm, clmParams, inds_full, inds_inner

def Load_CLNF_wild():
    clmParams = struct()
    clmParams.window_size = np.array([[25,25], [23,23], [21,21], [21, 21]])
    clmParams.numPatchIters = clmParams.window_size.shape[0]
    patches = Load_Patch_Experts('../models/general/', 'ccnf_patches_*_wild.mat', [], [], clmParams)
    
    pdmLoc = '../models/pdm/pdm_68_aligned_wild.mat'
    pdmLoc_dict = loadmat(pdmLoc)
    
    pdm = struct()
    pdm.M = pdmLoc_dict["M"].astype(np.double)
    pdm.E = pdmLoc_dict["E"].astype(np.double)
    pdm.V = pdmLoc_dict["V"].astype(np.double)
    
    clmParams.regFactor = np.array([35, 27, 20, 20])
    clmParams.sigmaMeanShift = np.array([1.25, 1.375, 1.5, 1.75])
    clmParams.tikhonov_factor = np.array([2.5, 5, 7.5, 7.5])
    
    clmParams.startScale = 1
    clmParams.num_RLMS_iter = 10
    clmParams.fTol = 0.01
    clmParams.useMultiScale = True
    clmParams.use_multi_modal = 1
    clmParams.multi_modal_types  = patches[0].multi_modal_types
    return patches, pdm, clmParams



def Load_Patch_Experts(col_patch_dir, col_patch_file, depth_patch_dir, depth_patch_file, clmParams):
    colourPatchFiles = glob.glob(os.path.join(col_patch_dir, col_patch_file))
    for i in range(len(colourPatchFiles)):
        file_name = colourPatchFiles[i]
        file_dict = loadmat(file_name)
        normalisationOptions = file_dict["normalisationOptions"]
        if ('ccnf_ration' in normalisationOptions.keys()):
            patch = struct()
            patch.centers = file_dict["centers"]
            patch.trainingScale = file_dict["trainingScale"]
            patch.visibilities = file_dict["visiIndex"]
            patch.patch_experts= file_dict["patch_experts"].patch_experts
            patch.correlations = file_dict["patch_experts"].correlations
            patch.rms_errors = file_dict["patch_experts"].rms_errors
            patch.modalities = file_dict["patch_experts"].types
            patch.multi_modal_types = file_dict["patch_experts"].types
            patch.type = "CCNF"
            patch.normalisationOptionsCol = normalisationOptions
            window_sizes = np.unique(clmParams.window_size.reshape(-1))
            for i in range(window_sizes.shape[0]):
                for view in range(patch.patche_experts.shape[0]):
                    for lmk in range(patch.patche_experts.shape[1]):
                        visiIndex = file_dict["visiIndex"]
                        if (visiIndex[view, lmk]):
                            num_modalities = patch.patch_experts[view,lmk].thetas.shape[2]
                            num_hls = patch.patch_experts[view,lmk].thetas.shape[0]
                            patchSize = np.sqrt(patch.patch_experts[view,lmk].thetas.shape[1] -1)
                            patchSize = [patchSize, patchSize]
                            
                            w = [[None] * num_hls] * num_modalities
                            norm_w = [[None] * num_hls] * num_modalities
                            
                            for hl in range(num_hls):
                                for p in range(num_modalities):
                                    w_c = patch.patch_experts[view,lmk].thetas[hl, 1:, p]
                                    norm_w_c = np.norm(w_c)
                                    w_c = w_c/np.norm(w_c)
                                    w_c = w_c.reshape(patchSize)
                                    w[hl, p] = w_c
                                    norm_w[hl, p] = norm_w_c
                            patch.patch_experts[view,lmk].w = w
                            patch.patch_experts[view,lmk].norm_w = norm_w
            
            
        
