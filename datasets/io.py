import h5py
import numpy as np
import open3d
import os

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        elif file_extension in ['.npz']:
            return cls._read_npz(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)
        
    @classmethod
    def _read_npz(cls, file_path):
        """
        读取稀疏体素注意力 npz，返回最小必要字段。
        不返回 meta，避免 collate_fn 乱搞嵌套结构。
        """
        data = np.load(file_path, allow_pickle=True)

        out = {
            "coords_zyx": data["coords_zyx"].astype(np.int32),
            "values":     data["values"].astype(np.float32),
            "origin":     data["origin"].astype(np.float32),
            "voxel_size": float(data["voxel_size"]),
            "grid_shape": data["grid_shape"].astype(np.int32),
        }
        return out

    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    @classmethod
    def _read_pcd(cls, file_path):
        pc = open3d.io.read_point_cloud(file_path)
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]