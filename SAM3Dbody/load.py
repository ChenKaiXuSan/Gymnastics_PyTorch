#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/code/SAM3Dbody/load.py
Project: /workspace/code/SAM3Dbody
Created Date: Friday January 23rd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 23rd 2026 4:50:57 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List

def load_data(input_data: Union[str, List[str], Path]) -> List[np.ndarray]:
    """
    根据输入加载所有帧。
    
    参数:
        input_data: 
            - 如果是视频: 视频文件的路径字符串 (str) 或 Path 对象
            - 如果是图片: 包含图片路径的列表 (List[str])
            
    返回:
        List[np.ndarray]: 包含所有读取到的帧的列表。每帧为 RGB 格式的数组。
    """
    frames_list = []

    # --- 情况 1: 输入是图片路径列表 ---
    if isinstance(input_data, list):
        print(f"检测到图片列表，正在读取 {len(input_data)} 张图片...")
        for img_path in input_data:
            path_str = str(img_path)
            # 读取图片
            img = cv2.imread(path_str)
            if img is not None:
                # 转换 BGR 到 RGB
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames_list.append(img_rgb)
            else:
                print(f"警告: 无法读取图片 -> {path_str}")

    # --- 情况 2: 输入是视频路径 (字符串或 Path) ---
    else:
        path_str = str(input_data)
        # 检查是否为支持的视频格式
        video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.MP4', '.MOV'}
        if Path(path_str).suffix.lower() not in video_exts:
            print(f"错误: 不支持的视频格式或输入类型错误 -> {path_str}")
            return []

        print(f"检测到视频文件，正在提取帧: {path_str}")
        cap = cv2.VideoCapture(path_str)
        
        if not cap.isOpened():
            print(f"错误: 无法打开视频 -> {path_str}")
            return []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 转换 BGR 到 RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(frame_rgb)
        
        cap.release()
        print(f"视频读取完成，共提取 {len(frames_list)} 帧")

    return frames_list
