
import os
import xml.dom.minidom
import numpy as np


def readxml(filename):
    '''
    返回中心
    '''
    DOMTree = xml.dom.minidom.parse(filename)
    data = DOMTree.documentElement

    # style = xml中的大类 ; typename = 细分属性; typevalue = 细分属性的值; valuename = xml文件，需要获取的值的tag;
    def get_data_vaule(style, typename, typevalue, valuename):
        nodelist = data.getElementsByTagName(style)  # 根据标签的名字获得节点列表

        for node in nodelist:
            if typevalue == node.getAttribute(typename):
                node_name = node.getElementsByTagName(valuename)
                value = node_name[0].childNodes[0].nodeValue
                return value
        return

    width = get_data_vaule('size', "", "", 'width')
    # print('width:', width)
    height = get_data_vaule('size', "", "", 'height')
    # print('height:', height)
    depth = get_data_vaule('size', "", "", 'depth')
    # print('width:', depth)

    class_name = get_data_vaule('object', "", "", 'name')
    # print('class_name:', class_name)

    xmin = get_data_vaule('bndbox', "", "", 'xmin')
    # print('xmin:', xmin)
    ymin = get_data_vaule('bndbox', "", "", 'ymin')
    # print('ymin:', ymin)
    xmax = get_data_vaule('bndbox', "", "", 'xmax')
    # print('xmax:', xmax)
    ymax = get_data_vaule('bndbox', "", "", 'ymax')
    # print('ymax:', ymax)

    return (int(xmax)+int(xmin))/2,(int(ymax)+int(ymin))/2,int(xmin),int(xmax),int(ymin),int(ymax)

def make_label(filename):
    '''
    制作标签，8*8的特征图，二分类，标签形状为8*8*1 有物体为1，其余都为0
    '''
    x, y,xmin,xmax,ymin,ymax = readxml(filename)

    x1 = int(xmin / 16)
    x2 = int(xmax / 16)
    y1 = int(ymin / 16)
    y2 = int(ymax / 16)
    # print(xmin,xmax,ymin,ymax)
    # print(x1,x2,y1,y2)
    label = np.zeros((8, 8, 1))
    label[:,:,0:1] = 0
    for i in range(8):
        if i>y1 and i <y2:
            for j in range(8):
                if j >x1 and j <x2:
                    # label[i][j][1] = 1
                    label[i][j][0] = 1
    return label

if __name__ == '__main__':
    import cv2

    s = 1661157807160
    label = make_label('./data/train/label/%s.xml' % s)
    print(label.reshape(8, 8))
    imagegt = cv2.imread('./data/train/img/%s.jpg' % s)
    out = label.reshape(8, 8)
    for i in range(8):
        for j in range(8):
            if out[i][j] > 0:
                out[i][j] = 1
                print(i, j)
                cv2.rectangle(imagegt, (j * 16, i * 16), (j * 16 + 16, i * 16 + 16), (0, 0, 255), 2)
            else:
                out[i][j] = 0
    print(out)
    imagegt = cv2.resize(imagegt, (256, 256))
    cv2.imshow('1', imagegt)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
