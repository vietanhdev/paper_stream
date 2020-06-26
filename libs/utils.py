

# a little helper function for getting all dettected marker ids
# from the reference image markers
def which(x, values):
    indices = []
    for ii in list(values):
        if ii in x:
            indices.append(list(x).index(ii))
    return indices
