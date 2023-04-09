import numpy as np
import os,cv2,time,torch,random,pytorchvideo,warnings,argparse,math
warnings.filterwarnings("ignore",category=UserWarning)

from pytorchvideo.transforms.functional import (
    uniform_temporal_subsample,
    short_side_scale_with_boxes,
    clip_boxes_to_image,)
from torchvision.transforms._functional_video import normalize
from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
from pytorchvideo.models.hub import slowfast_r50_detection
from pytorchvideo.models.hub import slowfast_16x8_r101_50_50
from pytorchvideo.models.hub import slowfast_r101
from pytorchvideo.data.utils import thwc_to_cthw
from deep_sort.deep_sort import DeepSort
from multiprocessing import Queue, Process,freeze_support,set_start_method,get_context

def tensor_to_numpy(tensor):
    #这里
    img = tensor.cpu().numpy().transpose((1, 2, 0))
    return img

def ava_inference_transform(clip, boxes,
    num_frames = 32, #if using slowfast_r50_detection, change this to 32, 4 for slow  32*2  Txt
    crop_size = 640, 
    data_mean = [0.45, 0.45, 0.45],  #均值归一化
    data_std = [0.225, 0.225, 0.225],
    slow_fast_alpha = 4, #if using slowfast_r50_detection, change this to 4, None for slow
):
    boxes = np.array(boxes)
    roi_boxes = boxes.copy() #检测到的有人的区域boxs信息
    #clip中的信息是num_frames中每帧的信息
    clip = uniform_temporal_subsample(clip, num_frames)
    clip = clip.float()
    clip = clip / 255.0  #归一化
    newclip = clip
    height, width = clip.shape[2], clip.shape[3]
    boxes = clip_boxes_to_image(boxes, height, width)
    #观察这里是否clip变成了boxes大小的信息图片
    clip, boxes = short_side_scale_with_boxes(clip,size=crop_size,boxes=boxes,)
    clip = normalize(clip,
        np.array(data_mean, dtype=np.float32),
        np.array(data_std, dtype=np.float32),) 
    boxes = clip_boxes_to_image(boxes, clip.shape[2],  clip.shape[3])
    if slow_fast_alpha is not None:
        fast_pathway = clip #32帧信息
        slow_pathway = torch.index_select(clip,1,
            torch.linspace(0, clip.shape[1] - 1, clip.shape[1] // slow_fast_alpha).long()) #4帧信息
        clip = [slow_pathway, fast_pathway]
    
    return clip, torch.from_numpy(boxes), roi_boxes

def plot_one_box(x, img, color=[100,100,100], text_info="None",
                 velocity=None,thickness=1,fontsize=0.5,fontthickness=1):
    # Plots one bounding box on image img
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness, lineType=cv2.LINE_AA)
    t_size = cv2.getTextSize(text_info, cv2.FONT_HERSHEY_TRIPLEX, fontsize , fontthickness+2)[0]
    cv2.rectangle(img, c1, (c1[0] + int(t_size[0]), c1[1] + int(t_size[1]*1.45)), color, -1)
    cv2.putText(img, text_info, (c1[0], c1[1]+t_size[1]+2), 
                cv2.FONT_HERSHEY_TRIPLEX, fontsize, [255,255,255], fontthickness)
    return img

def deepsort_update(Tracker,pred,xywh,np_img):
    outputs = Tracker.update(xywh, pred[:,4:5],pred[:,5].tolist(),cv2.cvtColor(np_img,cv2.COLOR_BGR2RGB))
    return outputs

def save_yolopreds_tovideo(yolo_preds,id_to_ava_labels,color_map,output_video):
    for i, (im, pred) in enumerate(zip(yolo_preds.imgs, yolo_preds.pred)):
        im=cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:#11类权重里面person是标签0 所以会跳到trackkid中去
                    ava_label = ''
                elif trackid in id_to_ava_labels.keys():
                    #这里下次debug下  看trackid和ava_labels如何匹配的。
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box,im,color,text)

        output_video.write(im.astype(np.uint8))

def show_yolopreds_tovideo(yolo_preds,id_to_ava_labels,color_map):
    for i, (im, pred) in enumerate(zip(yolo_preds.imgs, yolo_preds.pred)):
        #im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        if pred.shape[0]:
            for j, (*box, cls, trackid, vx, vy) in enumerate(pred):
                if int(cls) != 0:#11类权重里面person是标签0 所以会跳到trackkid中去
                    ava_label = ''
                elif trackid in id_to_ava_labels.keys():
                    #这里下次debug下  看trackid和ava_labels如何匹配的。
                    ava_label = id_to_ava_labels[trackid].split(' ')[0]
                else:
                    ava_label = 'Unknow'
                text = '{} {} {}'.format(int(trackid),yolo_preds.names[int(cls)],ava_label)
                color = color_map[int(cls)]
                im = plot_one_box(box,im,color,text)
        cv2.imshow("real time",im.astype(np.uint8))

def producer(frames):
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    res, img = video.read()
    width, height = img.shape[1], img.shape[0]
    time.sleep(5)
    while (True):
        # start = time.time()  # 计时
        imgs = []
        start = time.time()
        while (time.time() - start < 1):
            ret, img = video.read()
            imgs.append(img)
        imgs = [torch.from_numpy(img) for img in imgs]
        consume_time = time.time() - start
        print("speed %f seconds to %d frames" % (consume_time, len(imgs)))
        frames.put(imgs)

def consumer(frames,config,):
    print("consumer")
    model = torch.hub.load(r'E:\gitcode\yolov5-master', 'custom', path=r'E:\gitcode\yolov5-master\weights\best.pt',
                           source='local')
    model.conf = config.conf
    model.iou = config.iou
    model.max_det = 200
    if config.classes:
        model.classes = config.classes
    device = config.device
    video_model = slowfast_r50_detection(True).eval().to(device)
    deepsort_tracker = DeepSort("deep_sort/deep_sort/deep/checkpoint/ckpt.t7")
    ava_labelnames, _ = AvaLabeledVideoFramePaths.read_label_map("selfutils/temp.pbtxt")
    coco_color_map = [[random.randint(0, 255) for _ in range(3)] for _ in range(80)]
    imsize = config.imsize
    times = 0
    while True:
        if times == 20:
            break
        start = time.time()
        imgs = frames.get()
        video_clips = thwc_to_cthw(torch.stack(imgs)).to(torch.float32)
        # print("speed %f seconds to %d frames" % (end,len(imgs)))
        img_num = video_clips.shape[1]
        # print("can finish clips convert")
        imgs = []
        for j in range(img_num):
            imgs.append(tensor_to_numpy(video_clips[:, j, :, :]))
        with torch.no_grad():
            yolo_preds = model(imgs, size=imsize)

        yolo_preds.files = [f"img_{times * 25 + k}.jpg" for k in range(img_num)]
        print(times, video_clips.shape, img_num)
        deepsort_outputs = []
        for j in range(len(yolo_preds.pred)):
            temp = deepsort_update(deepsort_tracker, yolo_preds.pred[j].cpu(), yolo_preds.xywh[j][:, 0:4].cpu(),
                                   yolo_preds.imgs[j])
            if len(temp) == 0:
                temp = np.ones((0, 8))
            deepsort_outputs.append(temp.astype(np.float32))
        yolo_preds.pred = deepsort_outputs
        id_to_ava_labels = {}
        if yolo_preds.pred[img_num // 2].shape[0]:
            inputs, inp_boxes, _ = ava_inference_transform(video_clips, yolo_preds.pred[img_num // 2][:, 0:4],
                                                           crop_size=imsize)
            inp_boxes = torch.cat([torch.zeros(inp_boxes.shape[0], 1), inp_boxes], dim=1)
            if isinstance(inputs, list):
                inputs = [inp.unsqueeze(0).to(device) for inp in inputs]
            else:
                inputs = inputs.unsqueeze(0).to(device)
            with torch.no_grad():
                slowfaster_preds = video_model(inputs, inp_boxes.to(device))
                # 将当前一个框中的人物的动作信息输入到video_model中进行预测。
                slowfaster_preds = slowfaster_preds.cpu()
            # yolo_preds.pred[img_num//2][:,5]将该帧的目标预测标签[:,5]跟当前slowfast预测的信息合并，形成avalabel
            for tid, avalabel in zip(yolo_preds.pred[img_num // 2][:, 5].tolist(),
                                     np.argmax(slowfaster_preds, axis=1).tolist()):
                id_to_ava_labels[tid] = ava_labelnames[avalabel + 1]
        # end = time.time() - start
        # print("speed %f seconds to %d frames" % (end, len(imgs)))
        show_yolopreds_tovideo(yolo_preds, id_to_ava_labels, coco_color_map)
        end = time.time() - start
        print("speed %f seconds to test %d frames" % (end, len(imgs)))
        print("=============")
        times += 1  # 用来计时当前是多少秒
        if cv2.waitKey(100) & 0xff == ord('q'):
            break

# def main(config,frames):
#     #model = torch.hub.load('ultralytics/yolov5', 'yolov5l6')
#     camera = config.camera
#     if camera:
#
#
#
#         p1.join()
#         c1.join()
    

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=r"E:\gitcode\yolo_slowfast\vid", help='test imgs folder or video or camera')
    parser.add_argument('--output', type=str, default=r"E:\gitcode\yolo_slowfast\out", help='folder to save result imgs, can not use input folder')
    # object detect config
    parser.add_argument('--camera', type =bool,default = False,help='open camera')
    parser.add_argument('--imsize', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cuda', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    #freeze_support()
    config = parser.parse_args()
    print(config)
    frames = Queue()

    c1 = Process(target=consumer, args=(frames,config,))
    c1.start()
    # time.sleep(5)
    p1 = Process(target=producer, args=(frames,))
    p1.start()

#python PC_RTIME_yolo_slowfast.py --camera true
