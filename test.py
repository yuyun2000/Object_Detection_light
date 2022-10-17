import tensorflow as tf
import cv2
import numpy as np
model = tf.keras.models.load_model("./beizil.h5")

# imagegt = cv2.imread('./data/val/img/1661157807514.jpg')
# image = imagegt.astype(np.float32)
# img = image / 255
# img = img.reshape(1,128,128,3)
# out = model(img,training=True)
# out = np.array(tf.reshape(out[0:1,:,:,0:1],(8,8)))
# m = max(out.flatten())
# print(out)
# for i in range(8):
#     for j in range(8):
#         if out[i][j] >=m and out[i][j]>1:
#             out[i][j] =1
#             cv2.rectangle(imagegt,(j*16,i*16),(j*16+16,i*16+16),(0,0,255),2)
#         else:
#             out[i][j] =0
# print(out)
# imagegt = cv2.resize(imagegt,(256,256))
# cv2.imshow('1',imagegt)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


##---------------------------以下是视频
vid = cv2.VideoCapture('./data/val/img/t1.mp4')
fourcc = cv2.VideoWriter_fourcc(*'I420')
outv = cv2.VideoWriter('output.avi',fourcc,20,(256,256))
while True:
    flag,img = vid.read(0)
    if not flag:
        break
    img0 = cv2.resize(img,(128,128))
    img = img0.astype(np.float32)
    img = img / 255
    img = img.reshape(1, 128, 128, 3)
    out = model(img, training=True)
    # print(out)
    out = np.array(tf.reshape(out[0:1, :, :, 0:1], (8, 8)))
    m = max(out.flatten())
    print(m)
    # print(out)
    for i in range(8):
        for j in range(8):
            if out[i][j] >= m and m > 4.5:
                out[i][j] = 1
                cv2.rectangle(img0, (j * 16, i * 16), (j * 16 + 16, i * 16 + 16), (0, 0, 255), 2)
            else:
                out[i][j] = 0
    # print(out)
    img0 = cv2.resize(img0, (256, 256))
    outv.write(img0)
    cv2.imshow('1', img0)
    if ord('q') == cv2.waitKey(1):
        break
vid.release()
#销毁所有的数据
cv2.destroyAllWindows()
outv.release()