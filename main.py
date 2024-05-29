import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO(r"D:\code-project\best.pt")
tracker = DeepSort(max_age=30)


cap = cv2.VideoCapture(r"D:\code-project\Videos\test3.mp4")

# khởi tạo VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('demo3.avi', fourcc, 20.0, (1280, 720))

# def compute_color_for_labels(label):
#     if label==0: #car
#         color = (222,82,175)
#     elif label==1: #motorbike
#         color = (0,204,255)
#     elif label==2: #truck
#         color = (85,45,255)
#     elif label==3: #bus
#         color = (0,149,255)
#     else:
#         color = [int ((p * (label ** 2 - label +1)) % 255) for p in palette]
#     return tuple(color)

# Định nghĩa các đỉnh của các tứ giác
vertices1 = np.array([(465, 350), (609, 350), (510, 630), (2, 630)], dtype=np.int32)
vertices2 = np.array([(678, 350), (815, 350), (1203, 630), (743, 630)], dtype=np.int32)

# Định nghĩa phạm vi dọc cho việc cắt và ngưỡng làn đường
xv1, xv2 = 325, 635
lane_threshold = 609

# Định nghĩa ngưỡng để xem xét giao thông
traffic_flow = 2

track_dict={}
vehicles_left_lane = 0
vehicles_right_lane = 0

font = cv2.FONT_HERSHEY_SIMPLEX
org_in = (10, 50)
org_out = (750, 50)
left_line = (10, 100)
right_line = (750, 100)
fontScale = 1

while True:
    vleft,vright = 0,0
    ret, big_frame = cap.read()
    height, width = big_frame.shape[:2]
    #print(height,width)
    #height 720 width 1280
    detection_frame = big_frame.copy()

    # Làm đen các vùng ngoài phạm vi dọc đã chỉ định
    detection_frame[:xv1, :] = 0  # Làm đen từ trên xuống x1
    detection_frame[xv2:, :] = 0  # Làm đen từ x2 xuống dưới cùng của khung hình

    result = model.predict(big_frame,conf = 0.5,verbose = False)
    processed_frame = result[0].plot(line_width=1)

    processed_frame[:xv1, :] = big_frame[:xv1, :].copy()
    processed_frame[xv2:, :] = big_frame[xv2:, :].copy()  

    # Vẽ các tứ giác trên khung hình đã xử lý
    cv2.polylines(big_frame, [vertices1], isClosed=True, color=(0, 255, 0), thickness=2)
    cv2.polylines(big_frame, [vertices2], isClosed=True, color=(255, 0, 0), thickness=2)
    
    if len(result):
        result = result[0]
        names = result.names
        detect = []
        for box in result.boxes:
            
            x1,y1,x2,y2 = list(map(int,box.xyxy[0]))
            xc = x1+(x2-x1)//2
            yc = y1 + (y2-y1)//2
            conf = box.conf.item()
            cls = int(box.cls.item())
            detect.append([[x1,y1,x2-x1,y2-y1],conf,cls])
            
            if xc < lane_threshold and 350< yc <630:
                    vleft += 1
            elif xc > lane_threshold and 350 < yc < 630:
                    vright += 1
           
        tracks = tracker.update_tracks(detect, frame = big_frame)
    
        # Vẽ lên màn hình các khung chữ nhật kèm ID
        for i,track in enumerate(tracks):
            if track.is_confirmed() and track.det_conf :
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = list(map(int, ltrb))

                yc = y1+abs(y2-y1)//2
                xc = x1+abs(x2-x1)//2
                xy = (xc,yc)
                x_y = cv2.circle(big_frame,(xc,yc),(2), (0,255,0),2)

                track_id = track.track_id
                name = names[track.det_class]
                confidence = track.det_conf

                if track_id not in track_dict.keys():
                    track_dict[track_id] = 0 
                        
                label = f"{str(track_id)} {name} {confidence:.2f}"

                cv2.rectangle(big_frame, (x1, y1), (x2, y2),  (0,0,255), 2)
                cv2.putText(big_frame, label, (x1 + 5, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                if 300<= yc <=700 and xc <=640 and track_dict[track_id] == 0:
                    track_dict[track_id] = 1  
                    vehicles_left_lane += 1
                elif 300 < yc < 700 and xc > 640 and track_dict[track_id] == 0:
                    track_dict[track_id] = 1
                    vehicles_right_lane += 1
            
                # if xc < lane_threshold and 350< yc <630:
                #     if track_dict[track_id] == 0:
                #         track_dict[track_id] = 1
                #         vehicles_left_lane += 1
                # elif xc > lane_threshold and 350 < yc < 630:
                #     if track_dict[track_id] == 0:
                #         track_dict[track_id] = 1
                #         vehicles_right_lane += 1

        # Xác định cường độ giao thông cho làn đường bên trái
        
        traffic_intensity_left = "Heavy" if vleft > traffic_flow else "Smooth"
        # Xác định cường độ giao thông cho làn đường bên phải
        traffic_intensity_right = "Heavy" if vright > traffic_flow else "Smooth"

        cv2.putText(big_frame, f'Vehicles Left Lane: {vehicles_left_lane}', org_in, font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'Vehicles Right Lane: {vehicles_right_lane}', org_out, font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'traffic flow Left Lane: {traffic_intensity_left}', left_line, font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'traffic flow Right Lane: {traffic_intensity_right}', right_line, font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'vehicles in box: {vleft}', (10,150), font, 1, (255,255,255), 2, cv2.LINE_AA)
        cv2.putText(big_frame, f'vehicles in box: {vright}', (750,150), font, 1, (255,255,255), 2, cv2.LINE_AA)

    # start_point = (0,700)
    # end_point = (1280,700)
    # start_point_2 = (0, 300)
    # end_point_2 = (1280, 300) 
    # thickness = 2
    # color = (0,0,255)

    # big_frame_1 = cv2.line(big_frame, start_point, end_point, color, thickness)
    # big_frame_2 = cv2.line(big_frame, start_point_2, end_point_2, color, thickness)
    # text_in = "so xe: " + str(vehicles_left_lane)
    # text_out = "so xe: " + str(vehicles_right_lane)
    # big_frame_out = cv2.putText(big_frame, text_out, org_out, font, fontScale, (255,255,255), thickness)
    # big_frame_in = cv2.putText(big_frame, text_in, org_in, font, fontScale, (255,255,255), thickness)

    # start_point_x = (640,0)
    # end_point_x = (640, 720)
    # big_frame_x = cv2.line(big_frame, start_point_x, end_point_x, color, thickness)

    cv2.imshow("frame", big_frame)
    out.write(big_frame)


    if cv2.waitKey(8) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()