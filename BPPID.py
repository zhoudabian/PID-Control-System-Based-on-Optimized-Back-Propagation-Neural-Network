import PID
import matplotlib.pyplot as plt

plt.figure(1)  # 创建图表1

def TestPID():
    Process = PID.IncrementalPID()
    ProcessXaxis = [0]
    ProcessYaxis = [0]
    X = [0]
    Y = [0]

    for i in range(1, 500):
        Process.SetStepSignal(10)
        # Process.SetInertiaTime(100, 50)
        ProcessXaxis.append(i)
        ProcessYaxis.append(Process.y)
        X.append(i)
        Y.append(Process.e)


    plt.figure(1)
    plt.plot(ProcessXaxis, ProcessYaxis, 'r')

    plt.xlim(0, 1000)
    plt.ylim(0, 50)
    plt.title("IncrementalPID")
    plt.show()

if __name__ =='__main__':
    TestPID()