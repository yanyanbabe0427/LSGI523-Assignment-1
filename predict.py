import sys
import argparse
import os
from ultralytics import YOLO

def main(opt):
    # 加载训练好的模型权重（权重文件已包含模型结构，无需单独加载yaml）
    model = YOLO(opt.weights)
    model.info()

    # 批量预测测试集文件夹
    results = model.predict(
        source=opt.test_dir,    # 测试集图片文件夹路径
        save=True,              # 保存绘制好检测框的结果图片
        save_txt=True,          # 保存每张图的检测结果为txt标签文件
        save_conf=True,         # txt文件中同时保存检测置信度
        imgsz=640,              # 与训练时的图片尺寸保持一致
        conf=0.5,               # 置信度阈值，低于该值的检测框会被过滤
        iou=0.45,               # NMS非极大值抑制的IOU阈值
        project='runs/predict',  # 结果保存的根目录
        name='test_predict'     # 本次预测结果的文件夹名称
    )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, 
                        default='runs/detect/train2/weights/best.pt', 
                        help='训练好的权重文件路径')
    parser.add_argument('--test_dir', type=str, 
                        default='/root/ultralytics-YOLO26/road-damage/test/images', 
                        help='测试集图片文件夹路径')
    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
