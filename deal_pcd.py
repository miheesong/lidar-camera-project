import open3d
import numpy as np
import yaml
import cv2


def draw_pcd(pcd_array):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcd_array)
    open3d.visualization.draw_geometries([pcd])
    
def read_pld(filename):
    pcd = open3d.io.read_point_cloud(filename, format="pcd")
    pcd_array = np.array(pcd.points)
    
    pcd_array = pcd_array[ ~np.isnan(pcd_array).any(axis=1),:]  # Nan 행 제거
    
    return pcd_array

def parse_camera(f_path):
    with open(f_path) as f:
        calib_cam = {}
        calib_info = yaml.load(f, Loader=yaml.FullLoader)["camera"]["front"]
        calib_cam["P"] = calib_info["P"] 
        calib_cam["R"] = calib_info["R"]  
        calib_cam["t"] = calib_info["T"]  
        calib_cam["size"] = calib_info["size"]  
    return calib_cam

def parse_lidar(f_path):
    with open(f_path) as f:
        calib_lidar = {}
        calib_info = yaml.load(f, Loader=yaml.FullLoader)["lidar"]["rs80"]
        calib_lidar["R"] = calib_info["R"]
        calib_lidar["t"] = calib_info["T"]
    return calib_lidar


def in_image(point, size):
    row = np.bitwise_and(0 <= point[0], point[0] < size["width"])
    col = np.bitwise_and(0 <= point[1], point[1] < size["height"])
    return np.bitwise_and(row, col)

def lidar_to_camera(calib_cam, calib_lidar, points): 
    proj = []
    for p in points:
        proj_point = project_point(p,calib_cam, calib_lidar)
        if in_image(proj_point, calib_cam["size"]) and 0 <= proj_point[2]:
            proj.append(proj_point)
    return np.array(proj)


def project_point(point, calib_cam, calib_lidar):
    ## [x,y,z,1]
    lidar = np.append(point, [1], axis=0)
    lidar = np.transpose(lidar)
    # [R|t]X[x,y,z,1]
    matrix_lidar = np.concatenate([calib_lidar["R"], calib_lidar["t"]], axis=1)
    matrix_lidar = np.matmul(matrix_lidar, lidar)
    matrix_lidar = np.append(matrix_lidar, [1], axis=0) 

    P = np.array(calib_cam["P"]) 
    matrix_cam= np.concatenate([calib_cam["R"], calib_cam["t"]], axis=1)
    matrix_cam = np.concatenate([matrix_cam, [[0, 0, 0, 1]]], axis=0)  
    matrix_cam = np.matmul(P, matrix_cam) 
    matrix_cam = np.matmul(matrix_cam, matrix_lidar)  # (3,1) s*[x,y,1]
    
    depth = matrix_cam[-1]
    
    # x,y 정보
    matrix_cam[:-1] = matrix_cam[:-1] / depth 
    matrix_cam[:-1] = np.array(list(map(int, matrix_cam[:-1])))
   
    return matrix_cam

def save_depth_gt(depth_gt, calib_cam):
    img = np.zeros((calib_cam["size"]["height"], calib_cam["size"]["width"]), dtype=np.float32)
    
    for x, y, d in depth_gt:
        if img[int(y)][int(x)] == 0:
            img[int(y)][int(x)] = d
        else:
            if img[int(y)][int(x)] > d:
                img[int(y)][int(x)] = d
                
    cv2.imwrite("./gt_img/000001.png", img)

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3
                
     

if __name__ == "__main__":
    
    calib_file = "./dataset/calibration.yaml"
    pcd_file = "./pcd_dir/1656481494.537210464.pcd"
    
    calib_cam = parse_camera(calib_file)
    calib_lidar = parse_lidar(calib_file)
    
    points = read_pld(pcd_file)
    
    depth_gt = lidar_to_camera(calib_cam, calib_lidar, points)
    
    save_depth_gt(depth_gt, calib_cam)
