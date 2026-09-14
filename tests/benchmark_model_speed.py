
# _*_ coding: utf-8 _*_
# Author: GC Zhu
# Email: zhugc2016@gmail.com

import time
import numpy as np
import cv2

from gazefollower.face_alignment import MediaPipeFaceAlignment
from gazefollower.gaze_estimator import MGazeNetGazeEstimator

def speed_test(image_path: str, iterations: int = 100, warmup: int = 5):
    """
    分别测试人脸检测和视线估计的耗时。

    Args:
        image_path: 测试图片路径
        iterations: 总运行次数（含预热）
        warmup: 预热次数（不计入统计）
    """
    # 1. 加载图片
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # 2. 初始化模型
    fa = MediaPipeFaceAlignment()
    model_path = r"D:\python_projects\mnn_mobile_model\new\model_epoch_4.mnn"
    ga = MGazeNetGazeEstimator(model_path=model_path)

    # 3. 分别记录两个阶段的耗时（单位：秒）
    fa_timings = []
    ga_timings = []

    for i in range(iterations):
        timestamp = time.time()  # 模拟时间戳

        # ----- 阶段1：人脸检测 -----
        t_start = time.perf_counter()
        face_info = fa.detect(timestamp, frame)
        t_fa = time.perf_counter() - t_start

        # ----- 阶段2：视线估计 -----
        t_start = time.perf_counter()
        gaze_info = ga.detect(frame, face_info)
        t_ga = time.perf_counter() - t_start

        # 只记录预热后的数据
        if i >= warmup:
            fa_timings.append(t_fa)
            ga_timings.append(t_ga)

        # 可选进度输出
        if (i + 1) % 10 == 0:
            print(f"第 {i+1}/{iterations} 次完成 | FA: {t_fa*1000:.2f} ms, GA: {t_ga*1000:.2f} ms")

    # 4. 统计并打印结果
    def print_stats(name, timings):
        if not timings:
            print(f"{name}: 无有效数据")
            return
        avg_ms = np.mean(timings) * 1000
        std_ms = np.std(timings) * 1000
        fps = 1.0 / np.mean(timings)
        print(f"{name} 平均耗时: {avg_ms:.2f} ms ± {std_ms:.2f} ms")
        print(f"{name} 等效 FPS: {fps:.1f} fps")

    print("\n========== 速度测试结果（预热 {} 次，有效 {} 次）==========".format(warmup, len(fa_timings)))
    print_stats("MediaPipeFaceAlignment (FA)", fa_timings)
    print_stats("GazeEstimator (GA)", ga_timings)
    # 总耗时
    total_timings = [a + b for a, b in zip(fa_timings, ga_timings)]
    print_stats("整体 Pipeline", total_timings)
    print("======================================================")

    # 释放资源（如果实现了 release）
    # fa.release()   # MediaPipe 通常无需显式释放
    ga.release()

if __name__ == "__main__":
    image_path = "asset/example.jpg"   # 请修改为实际图片路径
    speed_test(image_path, iterations=100, warmup=5)