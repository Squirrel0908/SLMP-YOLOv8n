import pandas as pd
import numpy as np
import matplotlib.pylab as plt
fig,axes = plt.subplots(1,1,figsize=(4,4),dpi=100,facecolor="w")
def deal_yolov7_result(data_path):
    with open(data_path) as f:
        data = np.array(list(map(lambda x:np.array(x.strip().split()), f.readlines())))
    return data

if __name__ == '__main__':
    #epoch=300
    epoch = 300
    yolov8_result_csv_0005=r'C:\Users\Sen\Desktop\ultralytics-20240707\runs\train\C2f_star_LADH_MLCA2\results.csv'
    #yolov8_SLMP2_result_csv=r'C:\Users\Sen\Desktop\ultralytics-20240707\runs\train\C2f_star_LADH_MLCA2\results.csv'

    yolov8_result_data = pd.read_csv(yolov8_result_csv_0005)
   
    #yolov8_SLMP2_result_data = pd.read_csv(yolov8_SLMP2_result_csv)
    plt.figure(figsize=(10,10))
    plt.plot(np.arange(epoch),yolov8_result_data['       metrics/mAP50(B)'],label='loss',linewidth=2)
    #plt.plot(np.arange(epoch),yolov8_SLMP2_result_data['       metrics/mAP50(B)'],label='yolov8+C2f_star+LH+MLCA',linewidth=2)
    plt.legend()
    #fig.subplots_adjust(left=1,bottom=1)
    #axes.tick_params(axis='x', labelsize=18)
    plt.tight_layout()
    plt.rcParams['font.sans-serif'] = ['SimHei']

    #plt.xlabel('轮数Epoch')
    #plt.ylabel('损失值Loss value')
    plt.savefig('loss0005-curve.png')

