from Inspection.core.executor import Executor
from Inspection.utils.path_manager import RESOURCES_PATH
from Inspection.adapters.custom_adapters.U2Net import *
exe = Executor('U2Net', 'simulation')
FILE_RECORD_PATH = exe.now_record_path
sys.argv[0] = '/mnt/autor_name/haoTingDeWenJianJia/U-2-Net/u2net_portrait_demo.py'
import cv2
import torch
from model import U2NET
from torch.autograd import Variable
import numpy as np
from glob import glob
import os


def detect_single_face(face_cascade, img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        print(
            'Warming: no face detection, the portrait u2net will run on the whole image!'
            )
        return None
    wh = 0
    idx = 0
    for i in range(0, len(faces)):
        x, y, w, h = faces[i]
        if wh < w * h:
            idx = i
            wh = w * h
    return faces[idx]


def crop_face(img, face):
    if face is None:
        return img
    x, y, w, h = face
    height, width = img.shape[0:2]
    l, r, t, b = 0, 0, 0, 0
    lpad = int(float(w) * 0.4)
    left = x - lpad
    if left < 0:
        l = lpad - x
        left = 0
    rpad = int(float(w) * 0.4)
    right = x + w + rpad
    if right > width:
        r = right - width
        right = width
    tpad = int(float(h) * 0.6)
    top = y - tpad
    if top < 0:
        t = tpad - y
        top = 0
    bpad = int(float(h) * 0.2)
    bottom = y + h + bpad
    if bottom > height:
        b = bottom - height
        bottom = height
    im_face = img[top:bottom, left:right]
    if len(im_face.shape) == 2:
        im_face = np.repeat(im_face[:, :, (np.newaxis)], (1, 1, 3))
    im_face = np.pad(im_face, ((t, b), (l, r), (0, 0)), mode='constant',
        constant_values=((255, 255), (255, 255), (255, 255)))
    hf, wf = im_face.shape[0:2]
    if hf - 2 > wf:
        wfp = int((hf - wf) / 2)
        im_face = np.pad(im_face, ((0, 0), (wfp, wfp), (0, 0)), mode=
            'constant', constant_values=((255, 255), (255, 255), (255, 255)))
    elif wf - 2 > hf:
        hfp = int((wf - hf) / 2)
        im_face = np.pad(im_face, ((hfp, hfp), (0, 0), (0, 0)), mode=
            'constant', constant_values=((255, 255), (255, 255), (255, 255)))
    im_face = cv2.resize(im_face, (512, 512), interpolation=cv2.INTER_AREA)
    return im_face


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d - mi) / (ma - mi)
    return dn


def inference(net, input):
    tmpImg = np.zeros((input.shape[0], input.shape[1], 3))
    input = input / np.max(input)
    tmpImg[:, :, (0)] = (input[:, :, (2)] - 0.406) / 0.225
    tmpImg[:, :, (1)] = (input[:, :, (1)] - 0.456) / 0.224
    tmpImg[:, :, (2)] = (input[:, :, (0)] - 0.485) / 0.229
    tmpImg = tmpImg.transpose((2, 0, 1))
    tmpImg = tmpImg[(np.newaxis), :, :, :]
    tmpImg = torch.from_numpy(tmpImg)
    tmpImg = tmpImg.type(torch.FloatTensor)
    if torch.cuda.is_available():
        tmpImg = Variable(tmpImg.cuda())
    else:
        tmpImg = Variable(tmpImg)
    d1, d2, d3, d4, d5, d6, d7 = exe.run('forward', x=tmpImg)
    pred = 1.0 - d1[:, (0), :, :]
    pred = normPRED(pred)
    pred = pred.squeeze()
    pred = pred.cpu().data.numpy()
    del d1, d2, d3, d4, d5, d6, d7
    return pred


def main():
    im_list = glob('./test_data/test_portrait_images/your_portrait_im/*')
    print('Number of images: ', len(im_list))
    out_dir = FILE_RECORD_PATH + '/your_portrait_results'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    face_cascade = cv2.CascadeClassifier(
        './saved_models/face_detection_cv2/haarcascade_frontalface_default.xml'
        )
    model_dir = './saved_models/u2net_portrait/u2net_portrait.pth'
    net = exe.create_interface_objects(interface_class_name='U2NET', in_ch=
        3, out_ch=1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    for i in range(0, len(im_list)):
        print('--------------------------')
        print('inferencing ', i, '/', len(im_list), im_list[i])
        img = cv2.imread(im_list[i])
        height, width = img.shape[0:2]
        face = detect_single_face(face_cascade, img)
        im_face = crop_face(img, face)
        im_portrait = inference(net, im_face)
        cv2.imwrite(out_dir + '/' + im_list[i].split('/')[-1][0:-4] +
            '.png', (im_portrait * 255).astype(np.uint8))


main()
