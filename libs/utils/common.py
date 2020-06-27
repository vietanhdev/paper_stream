import os

# a little helper function for getting all dettected marker ids
# from the reference image markers
def which(x, values):
    indices = []
    for ii in list(values):
        if ii in x:
            indices.append(list(x).index(ii))
    return indices


def get_camera_path(camera_name):

    stream = os.popen('v4l2-ctl --list-devices')
    output = stream.read()
    lines = output.split("\n")
    for i, line in enumerate(lines):
        if camera_name in line:
            return lines[i+1].strip()
    
    return ""