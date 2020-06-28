import webbrowser
from PyQt5 import QtWidgets
from ..config import *

def new_room_clicked():
    print("New Room")
    url = NEW_ROOM_URL
    webbrowser.get().open(url)

    
def join_room_clicked():
    room_info, done = QtWidgets.QInputDialog.getText( 
             None, 'Input', 'Enter room ID or URL:')
    if done:
        print("Joining {}".format(room_info))
        
        if room_info.lower()[:4] == "http": # url
            url = room_info
        else:
            url = "{}{}".format(JOIN_ROOM_URL, room_info)
        
        webbrowser.get().open(url)