import socket
from _thread import *
import time
import os

#localization
def localize_img(id):
    kapture_path = './room_half_kapture'
    img_path = './' + id
    command = ''
    command += 'kapture_pipeline_localize_from_image.py '
    command += '-v debug '
    command += '-i {}/mapping '.format(kapture_path)
    command += '--query-img {} '.format(img_path)
    command += '-kpt {}/local_features/r2d2_WASF_N16/keypoints '.format(kapture_path)
    command += '-desc {}/local_features/r2d2_WASF_N16/descriptors '.format(kapture_path)
    command += '-gfeat {}/global_features/Resnet101-AP-GeM-LM18/global_features '.format(kapture_path)
    command += '-matches {}/local_features/r2d2_WASF_N16/NN_no_gv/matches '.format(kapture_path)
    command += '-matches-gv {}/local_features/r2d2_WASF_N16/NN_colmap_gv/matches '.format(kapture_path)
    command += '--colmap-map {}/colmap-sfm/r2d2_WASF_N16/AP-GeM-LM18_top5 '.format(kapture_path)
    command += '-o {}/colmap-localization '.format(kapture_path)
    command += '--topk 5 '
    command += '--config 2'
    os.system(command)
    
#get trajectory from txt
def get_trajectory():
    kapture_path = './room_half_kapture'
    path = kapture_path + '/colmap-localization/r2d2_500/AP-GeM-LM18_top5/AP-GeM-LM18_top5/kapture_localized_recover/sensors/trajectories.txt'
    parameter = []
    with open(path, mode = 'rt') as f:
        f.readline()
        f.readline()
        parameter = f.readline().split(', ')
        parameter[-1] = parameter[-1].strip()
    trajectory = ' '.join(parameter[-7:])
    return trajectory


def threaded(client_socket, addr):

    print('Connected by :', addr[0], ':', addr[1])

    while True:

        try : 
            data = client_socket.recv(1024)
            if not data:
                print('Disconnected by' + addr[0], ':', addr[1])
                break

            print('Received from ' + addr[0], ':', addr[1], data.decode())

            request_data = data.decode().split()
            id = request_data[0]
            file_name = request_data[1]
            file_size = int(request_data[2])
            print(request_data)

            path = './' + id
            if not os.path.isdir(path):
                os.mkdir(path)
                
            with open(path + '/' + file_name, 'wb') as f:
                data = client_socket.recv(1024)
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
            print('l')
            #localize_img(id)
            client_socket.send(get_trajectory().encode())
            print("trajectory transferred")
            break   
            
        except ConnectionResetError as e:
            print('Disconnected by' + addr[0], ':', addr[1])
            break
            
    client_socket.close()


HOST = '125.130.0.42'
PORT = 17000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen()

print('server start')

while True:
    print('wait')
    client_socket, addr = server_socket.accept()
    start_new_thread(threaded, (client_socket, addr))

server_socket.close()