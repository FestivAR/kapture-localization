import socket
from _thread import *
import time
import argparse
import sys
import subprocess
import os
import os.path as path
from private import Config

# from kapture_localization.utils.subprocess import run_python_command
sys.path.append(path.dirname(path.abspath(path.dirname(__file__))))

from pipeline.kapture_pipeline_localize_from_image import localize_image_pipeline

def localize(img_path):
    HERE_PATH = path.abspath(path.normpath(path.dirname(__file__)))
    # loc_from_img_path = path.join(HERE_PATH, '../pipeline/kapture_pipeline_localize_from_image.py')
    '''
    kapture_pipeline_localize_from_image.py 
    -lfext ../../../r2d2/ 
    -gfext ../../../deep-image-retrieval/ 
    -i ./mapping/ 
    --query-img ./image/ 
    -kpt ./reconstruction/keypoints/faster2d2_WASF_N16/ 
    -desc ./reconstruction/descriptors/faster2d2_WASF_N16/ 
    -gfeat ./reconstruction/global_features/Resnet101-AP-GeM-LM18/ 
    -matches ./reconstruction/r2d2_500/NN_no_gv/matches/ 
    -matches-gv ./reconstruction/r2d2_500/NN_colmap_gv/matches/ 
    --colmap-map ./colmap-sfm/r2d2_500/AP-GeM-LM18_top5/ 
    -o ./colmap-localization/r2d2_500/AP-GeM-LM18_top5 
    --topk 5 
    --config 2
    '''
    map_path = path.join(HERE_PATH, 'data/mapping')
    reconstruction_path = path.join(HERE_PATH, 'data/reconstruction')

    trajectory = localize_image_pipeline(lfeat_ext_path='../../r2d2/',
                                        gfeat_ext_path='../../deep-image-retrieval',
                                        kapture_map_path=map_path,
                                        query_image_path=img_path,
                                        keypoints_path=path.join(reconstruction_path,'keypoints/faster2d2_WASF_N16/'),
                                        descriptors_path=path.join(reconstruction_path, 'descriptors/faster2d2_WASF_N16/'),
                                        global_features_path=path.join(reconstruction_path, 'global_features/Resnet101-AP-GeM-LM18/'),
                                        matches_path=path.join(reconstruction_path, 'r2d2_500/NN_no_gv/matches/'),
                                        matches_gv_path=path.join(reconstruction_path, 'r2d2_500/NN_colmap_gv/matches/'),
                                        colmap_map_path=path.join(HERE_PATH, 'data/colmap-sfm/r2d2_500/AP-GeM-LM18_top5/'),
                                        localization_output_path=path.join(HERE_PATH, 'data/colmap-localization/r2d2_500/AP-GeM-LM18_top5'),
                                        topk=5, 
                                        config=2,
                                        colmap_binary='colmap',
                                        python_binary=sys.executable,
                                        force_overwrite_existing=True)
    
    print('trajectory:', trajectory)
    result = trajectory[2].strip().split(',')
    result = ''.join(result[2:]).lstrip()
    print('result:', result)
    return result


def threaded(client_socket, addr):

    print('Connected by :', addr[0], ':', addr[1])

    while True:

        try : 
            data = client_socket.recv(1024)
            if not data:
                print('Disconnected by' + addr[0], ':', addr[1])
                break
            print('Received from ' + addr[0], ':', addr[1])

            request_data = data.decode('utf-8').split()
            id = request_data[0]
            file_name = request_data[1]
            file_size = int(request_data[2])
            print(request_data)

            dir_path = path.join(os.getcwd(), 'data', id)
            img_dir_path = path.join(os.getcwd(), 'data', id, 'test')
            os.makedirs(img_dir_path, exist_ok=True)
                
            with open(path.join(img_dir_path, file_name), 'wb') as f:
                data = client_socket.recv(1024)
                # client_socket.sendall('test_send0'.encode())
                pre = data[-3:]
                while data:
                    f.write(data)
                    data = client_socket.recv(1024)
                    if b'EOF' in pre+data[-3:]:
                        f.write(data[:-3])
                        print('EOF\n')
                        break
                    print("length : ", + len(data))
                    pre = data[-3:]
            print('End write')
            client_socket.sendall(localize(dir_path).encode())
            print("trajectory transferred")
            # break
            
        except ConnectionResetError as e:
            print('Disconnected by' + addr[0], ':', addr[1])
            break
            
    client_socket.close()


HOST = Config.serv_addr
PORT = Config.serv_port

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

print('server start')

while True:
    print('wait')
    client_socket, addr = server_socket.accept()
    start_new_thread(threaded, (client_socket, addr))

# server_socket.close()
