import matplotlib.pyplot as plt
import numpy as np
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")

    return np.asfarray(data, float)

if __name__=="__main__":
    seg4_path = "D:/STDC-Seg-master/loss/saba118/loss_seg4.txt"
    seg4_loss = data_read(seg4_path)
    x = range(len(seg4_loss))

    plt.figure()

    plt.xlabel('iters')
    plt.ylabel('loss')

    plt.plot(x, seg4_loss, linewidth=1, linestyle='solid', label="seg4 loss")
    plt.legend()
    plt.title('segmentation loss')
    plt.show()