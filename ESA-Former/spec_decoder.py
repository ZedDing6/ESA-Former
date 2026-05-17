import os

import torch
from PIL import Image
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from osgeo import gdal, osr
from torchvision.datasets import DatasetFolder
from typing import Callable, Optional, Any
from torchvision import transforms

import numpy as np
class CustomTIFDataset(DatasetFolder):

    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = default_loader,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader,
            extensions=IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tif_path = os.path.join(self.root, self.samples[idx][0])
        img = gdal.Open(tif_path)
        geo_converter = GeoConverter(tif_path)
        rgb_image1 = np.zeros((img.RasterYSize, img.RasterXSize, 3), dtype=np.uint8)
        rgb_image2 = np.zeros((img.RasterYSize, img.RasterXSize, 3), dtype=np.uint8)
        band1 = img.GetRasterBand(1).ReadAsArray()
        band2 = img.GetRasterBand(2).ReadAsArray()
        band3 = img.GetRasterBand(3).ReadAsArray()
        band4 = img.GetRasterBand(4).ReadAsArray()
        band5 = img.GetRasterBand(5).ReadAsArray()
        bands = [band1, band2, band3, band4, band5]
        # bands = [band1, band2, band3, band4]
        img = torch.from_numpy(np.stack(bands, axis=0)).float()
        target = self.samples[idx][1]
        latitude, longitude = 0, 0
        label = (target, latitude, longitude)

        return img, label


    # def __getitem__(self, idx):
    #     tif_path = os.path.join(self.root, self.samples[idx][0])
    #     img = gdal.Open(tif_path)
    #
    #     bands = []
    #     for i in range(5):
    #         band = img.GetRasterBand(i + 1).ReadAsArray().astype(np.float32)
    #         nodata = img.GetRasterBand(i + 1).GetNoDataValue()
    #         if nodata is not None:
    #             band[band == nodata] = 0.0
    #
    #         bands.append(band)
    #
    #     img_tensor = torch.from_numpy(
    #         np.stack(bands, axis=0)
    #     )
    #
    #     normalize = transforms.Normalize(
    #         mean=[0.485, 0.456, 0.406, 0.137, 0.3],
    #         std=[0.229, 0.224, 0.225, 0.051, 0.162]
    #     )
    #
    #     # img_tensor: (5, H, W), float32
    #     img_tensor = normalize(img_tensor)
    #     target = self.samples[idx][1]
    #     label = (target, 0, 0)
    #
    #     return img_tensor, label


class GeoConverter:
    def __init__(self, tif_path):
        self.tif_path = tif_path
        self.dataset = gdal.Open(tif_path, gdal.GA_ReadOnly)
        self.geotransform = self.dataset.GetGeoTransform()

    def calculate_center_coordinates(self):
        """计算图像中心像素的投影坐标"""
        center_x_pixel = self.dataset.RasterXSize // 2
        center_y_pixel = self.dataset.RasterYSize // 2

        # 计算对应的投影坐标
        x_proj = (
                self.geotransform[0] +
                center_x_pixel * self.geotransform[1] +
                center_y_pixel * self.geotransform[2]
        )
        y_proj = (
                self.geotransform[3] +
                center_x_pixel * self.geotransform[4] +
                center_y_pixel * self.geotransform[5]
        )

        return x_proj, y_proj

    def transform_to_geographic(self, x_proj, y_proj):
        """将投影坐标转换为地理坐标"""
        # 创建 WGS84 空间参考对象
        wgs84_srs = osr.SpatialReference()
        wgs84_srs.ImportFromEPSG(4326)

        # 创建坐标转换对象
        spatial_ref = osr.SpatialReference(wkt=self.dataset.GetProjection())
        transform = osr.CoordinateTransformation(spatial_ref, wgs84_srs)

        # 转换为地理坐标
        lat, lon, _ = transform.TransformPoint(x_proj, y_proj)

        return lat, lon

    @staticmethod
    def decimal_to_dms(degrees, is_longitude=False):
        """Convert decimal degrees to degrees, minutes, seconds format."""
        hemisphere = 'W' if degrees < 0 else 'E' if is_longitude else 'S' if degrees < 0 else 'N'
        abs_degrees = abs(degrees)
        d = int(abs_degrees)
        m = int((abs_degrees - d) * 60)
        s = (abs_degrees - d - m / 60) * 3600
        return f"{d}°{m}'{s:.2f}\"{hemisphere}"

    def get_geo(self):
        """获取中心坐标的地理位置"""
        x_proj, y_proj = self.calculate_center_coordinates()
        lat, lon = self.transform_to_geographic(x_proj, y_proj)

        lat_dms = self.decimal_to_dms(lat, is_longitude=False)
        lon_dms = self.decimal_to_dms(lon, is_longitude=True)

        return lat_dms, lon_dms

    def close(self):
        """关闭数据集"""
        if self.dataset is not None:
            self.dataset = None
class TIFImageFolder(CustomTIFDataset):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super().__init__(
            root,
            loader=loader,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
        )
        self.imgs = self.samples
