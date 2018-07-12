import socket  
import cv2  
import threading  
import struct  
import numpy  
import sys
sys.path.append("./build/lib.linux-aarch64-2.7")
import mypack
mypack.netInit()

###### change your team name here
teamName = "teamName"
windowName = "DAC HDC contest team:"+teamName

class process_display_Object:  
    def __init__(self,addr_port_client_Img=("",1000)):  
        print 'displayImg_Connect_Object init'
        self.resolution=[640,360]  
        self.client_port_Img=addr_port_client_Img                                
  
    def Socket_Connect_Client(self): 
        print 'displayImg_Connect_Object   Socket_Connect' 
        self.clientIMG=socket.socket(socket.AF_INET,socket.SOCK_STREAM)  
        self.clientIMG.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)   
        print("As a client, displayImg receives imgs from %s:%d" % (self.client_port_Img[0],self.client_port_Img[1]))  
        self.clientIMG.connect(self.client_port_Img) 

    def ProcessImg(self):  
        print 'displayImg_Object shows imgs' 
        self.name="displayImg"  
        self.clientIMG.send(struct.pack("hh", self.resolution[0], self.resolution[1])) 
        while(1):  
            info=struct.unpack("ll",self.clientIMG.recv(16))  
            buf_size=info[0]          
            if buf_size:  
                try:  
                    self.buf=b""                  
                    temp_buf=self.buf  
                    while(buf_size):              
                        temp_buf=self.clientIMG.recv(buf_size)  
                        buf_size-=len(temp_buf)  
                        self.buf+=temp_buf      
                        data = numpy.fromstring(self.buf, dtype='uint8')     
                        self.image = cv2.imdecode(data, 1) 
                        print("img shape {}".format(self.image.shape))
                        ###### replace this line with your detection code                         
                        # detecRec = numpy.random.random(4) 
                        detecRec = mypack.infer(self.image.astype(numpy.float32, copy=False))[0]
                        print str(info[1])+", perform detection processing successfully, and the result is "+str(detecRec) 
                        self.clientIMG.send(struct.pack("h",168)) ## indicate receive data succesfully
                        cv2.rectangle(self.image,(abs(int(detecRec[0])),abs(int(detecRec[2]))),(abs(int(detecRec[1])),abs(int(detecRec[3]))),(0,255,0),4)   
                        ###### uncomment the following 2 lines to enable fullscreen when you can successfully run the code
                        ##cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN) 
                        ##cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)             
                        cv2.imshow(windowName, self.image)    
                        cv2.waitKey(1)               
                except:  
                    pass;  
                finally:  
                    if(cv2.waitKey(10)==27):          
                        self.client.close()  
                        cv2.destroyAllWindows()  
                        break  

  
    def ProcessInThread(self): 
        print 'displayImg_Connect_Object   Get_Data'
        showThread=threading.Thread(target=self.ProcessImg)  
        showThread.start()  
  
if __name__ == '__main__':  
    ###### Change the ip and port according to your setting in the line below
    displayImg=process_display_Object(("127.0.0.1",3010))  
    displayImg.Socket_Connect_Client()  
    displayImg.ProcessInThread()  



