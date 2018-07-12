## this is for GPU demo with only one FPGA and one computer for computer for display
## xxu8@nd.edu


import socket  
import threading  
import struct  
import time  
import cv2  
import numpy  
import xml.etree.ElementTree as ET

NofClients = 1  
  
class Senders_Carame_Object:  
    def __init__(self,addr_ports=[("192.168.1.80",3000)]): 
        print 'Senders_Carame_Object init'
        print addr_ports
        self.resolution=(640,360)        
        self.img_fps=10
        self.addr_ports=addr_ports
        self.connections=[]
        for i in range (0,NofClients):                   
            print "setup connection "+str(i)
            self.Set_Socket(self.addr_ports[i])
     
    def Set_Socket(self,addr_port):  
        print 'Senders_Carame_Object Set_Socket'
        self.connections.append(socket.socket(socket.AF_INET,socket.SOCK_STREAM))
        self.connections[-1].setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)   
        self.connections[-1].bind(addr_port)  
        self.connections[-1].listen(5)  
        print("the process work in the port:%d" % addr_port[1])  

  
def check_option(object,clients):  
    for i in range (0, NofClients): 
		info=struct.unpack('hh',clients[i].recv(4))  ## 8 or 12
		if info[0]!=object.resolution[0] or info[1]!=object.resolution[1]:
		    print "error: check option fails, received resolution is: "+str(info[0])+","+str(info[1])
		    return 1
		else:
			return 0
  
def RT_Image(object,clients):  
    print 'RT_Image '
    if(check_option(object,clients)==1):  
        return  
    # camera=cv2.VideoCapture(0)                                
    camera=cv2.VideoCapture("demo_video.avi")                                
    img_param=[int(cv2.IMWRITE_JPEG_QUALITY),object.img_fps]  
    indexN = 0
    #used for test
    # xmlDir = "./Training_data_xml_dac_demo/xmls_demo"
    #end
    while(1):  
        ##time.sleep(0.1) ## about 10 fps              
        _,object.img=camera.read()   
        indexN = indexN + 1
        object.img=cv2.resize(object.img,object.resolution)       
        #used for test
        # xmlfile = xmlDir + '/' + str(indexN) + ".xml"
        # xmltree = ET.parse(xmlfile)
        # obj = xmltree.find("object")
        # bndbox = obj.find('bndbox')
        # xmin = int(bndbox.find('xmin').text.strip())
        # xmax = int(bndbox.find('xmax').text.strip())
        # ymin = int(bndbox.find('ymin').text.strip())
        # ymax = int(bndbox.find('ymax').text.strip())
        # ymin = int(ymin*360/480)
        # ymax = int(ymax*360/480)
        # object.img = cv2.rectangle(object.img, (xmin, ymin), (xmax, ymax), (0,0,255),4)
        # print("{} bndbox {}".format(indexN, (xmin,ymin, xmax, ymax)))
        #end
        _,img_encode=cv2.imencode('.jpg',object.img,img_param)    
        img_code=numpy.array(img_encode)                         
        object.img_data=img_code.tostring()                     
        try:  
            for i in range (0, NofClients): 
		        clients[i].send(struct.pack("ll",len(object.img_data),indexN)+object.img_data)
            print str(indexN)+', size of the send img:', len(object.img_data)
            ## wait until the images are processed on FPGAs
            feedback=struct.unpack('h',clients[0].recv(2))
            if feedback[0]!=168:
                print "feedback from FPGA error, "+str(feedback)
                return    
        except:  
            camera.release()         
            return  
  
if __name__ == '__main__':  
	senders=Senders_Carame_Object([("127.0.0.1",3010)])   
	clients = []
	for i in range (0, NofClients):                   
		print "connection accept with "+str(i)       
		clients.append(senders.connections[i].accept()[0])
	clientThread=threading.Thread(None,target=RT_Image,args=(senders,clients,))  
	clientThread.start()  





