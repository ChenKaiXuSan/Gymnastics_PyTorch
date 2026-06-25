#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/prepare_dataset/split_video_to_image.py
Project: /workspace/code/prepare_dataset
Created Date: Friday January 23rd 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday January 23rd 2026 4:11:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import cv2
import os
from pathlib import Path


def process_videos_with_structure(input_root, output_root, interval=30):
    """
    input_root: 入力フォルダ (/workspace/data/raw)
    output_root: 出力フォルダ (/workspace/data/frames)
    interval: フレームの抽出間隔
    """

    # 対応する動画形式
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

    # Pathオブジェクトに変換して扱いやすくする
    input_path = Path(input_root)
    output_path = Path(output_root)

    print(f"探索開始: {input_path}")

    # フォルダ内を再帰的に探索
    for root, dirs, files in os.walk(input_root):
        for file in files:
            if file.lower().endswith(video_extensions):
                # フルパスの取得
                video_full_path = Path(root) / file

                # 相対パスを取得して、保存先のディレクトリ構造を決定
                # 例: raw/subdir/vid.mp4 -> subdir/vid
                relative_path = video_full_path.relative_to(input_path)
                save_dir = output_path / relative_path.with_suffix("")

                # 保存先フォルダを作成
                save_dir.mkdir(parents=True, exist_ok=True)

                # 動画の処理
                extract_frames(str(video_full_path), str(save_dir), interval)


def extract_frames(video_path, save_dir, interval):
    video_name = Path(video_path).name
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"失敗: 開けませんでした {video_name}")
        return

    count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 指定間隔で保存
        if count % interval == 0:
            # --- 回転処理を追加 ---
            # cv2.ROTATE_90_CLOCKWISE: 時計回りに90度
            # cv2.ROTATE_180: 180度
            # cv2.ROTATE_90_COUNTERCLOCKWISE: 反時計回りに90度
            rotated_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            file_name = f"{saved_count:05d}.jpg"
            save_path = os.path.join(save_dir, file_name)

            # 回転後の画像を保存
            cv2.imwrite(save_path, rotated_frame)
            saved_count += 1

        count += 1

    cap.release()
    print(f"完了: {video_name} -> {saved_count}枚保存先: {save_dir}")


# --- 実行設定 ---
INPUT_DIR = "/workspace/data/raw"
OUTPUT_DIR = "/workspace/data/frames"
FRAME_INTERVAL = 1  # 1フレームごとに1枚保存

if __name__ == "__main__":
    process_videos_with_structure(INPUT_DIR, OUTPUT_DIR, FRAME_INTERVAL)
