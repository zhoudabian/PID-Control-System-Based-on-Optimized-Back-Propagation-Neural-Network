import numpy as np



class IncrementalPID:
    def __init__(self):

        self.xite_1 = 0.2
        self.alfa = 0.95
        self.IN = 4
        self.H = 5
        self.Out = 3
        self.wi = np.mat([[-0.6394, -0.2696, -0.3756, -0.7023],
                         [-0.8603, -0.2013, -0.5024, -0.2596],
                         [-1.0000, 0.5543, -1.0000, -0.5437],
                         [-0.3625, -0.0724, 0.6463, -0.2859],
                         [0.1425, 0.0279, -0.5406, -0.7660]]
                         )


        self.wi_1 = self.wi
        self.wi_2 = self.wi
        self.wi_3 = self.wi
        self.wo = np.mat([[0.7576, 0.2616, 0.5820, -0.1416, -0.1325],
                          [-0.1146, 0.2949, 0.8352, 0.2205, 0.4508],
                          [0.7201, 0.4566, 0.7672, 0.4962, 0.3632]]
                         )

        self.wo_1 = self.wo
        self.wo_2 = self.wo
        self.wo_3 = self.wo

        self.Kp = 0.0
        self.Ki = 0.0
        self.Kd = 0.0

        # self.ts = 40  # 采样周期取值
        self.x = [self.Kp, self.Ki, self.Kd]  # 比例、积分、微分初值

        self.y = 0.0  # 系统输出值
        self.y_1 = 0.0  # 上次系统输出值
        self.y_2 = 0.0  # 上次系统输出值

        self.e = 0.0  # 输出值与输入值的偏差
        self.e_1 = 0.0
        self.e_2 = 0.0
        self.de_1 = 0.0

        self.u = 0.0
        self.u_1 = 0.0
        self.u_2 = 0.0
        self.u_3 = 0.0
        self.u_4 = 0.0
        self.u_5 = 0.0


        self.Oh = np.mat(np.zeros((self.H, 1)))  # %隐含层的输出
        self.I = self.Oh  # 隐含层的输入


        self.Oh_sub = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.K_sub = [0.0, 0.0, 0.0]
        self.dK_sub = [0.0, 0.0, 0.0]
        self.delta3_sub = [0.0, 0.0, 0.0]
        self.dO_sub = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.delta2_sub = [0.0, 0.0, 0.0, 0.0, 0.0]

        self.den = -0.8251
        self.num = 0.2099

    # 设置PID控制器参数
    def SetStepSignal(self, SetSignal):

        self.y = -self.den * self.y_1 + self.num * self.u_5
        self.e = SetSignal - self.y
        self.xi = np.mat([SetSignal, self.y, self.e, 5])

        self.x[0] = self.e - self.e_1  # 比例输出
        self.x[1] = self.e  # 积分输出
        self.x[2] = self.e - 2 * self.e_1 + self.e_2  # 微分输出
        self.epid = np.mat([[self.x[0]], [self.x[1]], [self.x[2]]])#列

        self.I = np.dot(self.xi, (self.wi.T))

        for i1 in range(5):
            self.Oh_sub[i1] = (np.e ** (self.I.tolist()[0][i1]) - np.e ** (-self.I.tolist()[0][i1])) / (np.e ** (self.I.tolist()[0][i1]) + np.e ** (-self.I.tolist()[0][i1]))  # %在激活函数作用下隐含层的输出
        self.Oh = np.mat([[self.Oh_sub[0]], [self.Oh_sub[1]], [self.Oh_sub[2]], [self.Oh_sub[3]], [self.Oh_sub[4]]])

        self.K = np.dot(self.wo, self.Oh)  # 输出层的输入，即隐含层的输出*权值

        for i2 in range(3):
            self.K_sub[i2] = (np.e ** (self.K.tolist()[i2][0])) / (np.e ** (self.K.tolist()[i2][0]) + np.e ** (-self.K.tolist()[i2][0]))  # 输出层的输出，即PID三个参数
        self.K = np.mat([[self.K_sub[0]], [self.K_sub[1]], [self.K_sub[2]]])

        self.Kp = self.K_sub[0]
        self.Ki = self.K_sub[1]
        self.Kd = self.K_sub[2]
        self.Kpid = np.mat([self.Kp, self.Ki, self.Kd]) #行

        self.du = np.dot(self.Kpid, self.epid).tolist()[0][0]
        self.u = self.u_1 + self.du
        # if self.u >= 10:
        #     self.u = 10
        #
        # if self.u <= -10:
        #     self.u = -10

        self.de = self.e - self.e_1
        if self.de > (self.de_1 * 1.04):
            self.xite = 0.7 * self.xite_1
        elif self.de < self.de_1:
            self.xite = 1.05 * self.xite_1
        else:
            self.xite = self.xite_1

        #  权值在线调整
        self.dyu = np.sin((self.y - self.y_1) / (self.u - self.u_1 + 0.0000001))

        for i3 in range(3):
            self.dK_sub[i3] = 2 / ((np.e ** (self.K_sub[i3]) + np.e ** (-self.K_sub[i3])) * (np.e ** (self.K_sub[i3]) + np.e ** (-self.K_sub[i3])))
        self.dK = np.mat([self.dK_sub[0], self.dK_sub[1], self.dK_sub[2]])


        for i4 in range(3):
            self.delta3_sub[i4] = self.e * self.dyu * self.epid.tolist()[i4][0] * self.dK_sub[i4]
        self.delta3 = np.mat([self.delta3_sub[0], self.delta3_sub[1], self.delta3_sub[2]])

        for l in range(3):
            for i5 in range(5):
                self.d_wo = (1 - self.alfa) * self.xite * self.delta3_sub[l] * self.Oh.tolist()[i5][0] + self.alfa * (self.wo_1 - self.wo_2)

        # self.wo = self.wo_1 + self.d_wo + self.alfa * (self.wo_1 - self.wo_2)
        self.wo = self.wo_1 + self.d_wo

        for i6 in range(5):
            self.dO_sub[i6] = 4 / ((np.e ** (self.I.tolist()[0][i6]) + np.e ** (-self.I.tolist()[0][i6])) * (np.e ** (self.I.tolist()[0][i6]) + np.e ** (-self.I.tolist()[0][i6])))
        self.dO = np.mat([self.dO_sub[0], self.dO_sub[1], self.dO_sub[2], self.dO_sub[3], self.dO_sub[4]])

        self.segma = np.dot(self.delta3, self.wo)

        for i7 in range(5):
             self.delta2_sub[i7] = self.dO_sub[i7] * self.segma.tolist()[0][i7]
        self.delta2 = np.mat([self.delta2_sub[0], self.delta2_sub[1], self.delta2_sub[2], self.delta2_sub[3], self.delta2_sub[4]])

        self.d_wi = (1 - self.alfa) * self.xite * self.delta2.T * self.xi + self.alfa * (self.wi_1 - self.wi_2)
        self.wi = self.wi_1 + self.d_wi

        # 参数更新
        self.u_5 = self.u_4
        self.u_4 = self.u_3
        self.u_3 = self.u_2
        self.u_2 = self.u_1
        self.u_1 = self.u

        self.y_2 = self.y_1
        self.y_1 = self.y

        self.wo_3 = self.wo_2
        self.wo_2 = self.wo_1
        self.wo_1 = self.wo

        self.wi_3 = self.wi_2
        self.wi_2 = self.wi_1
        self.wi_1 = self.wi

        self.e_2 = self.e_1
        self.e_1 = self.e
        self.xite_1 = self.xite


        # IncrementValue = self.Kp * (self.Error - self.LastError) + self.Ki * self.Error + self.Kd * (
        #             self.Error - 2 * self.LastError + self.LastLastError)
        # self.PIDOutput += IncrementValue
        # self.LastLastError = self.LastError
        # self.LastError = self.Error

    # 设置一阶惯性环节系统  其中InertiaTime为惯性时间常数
    def SetInertiaTime(self, InertiaTime, SampleTime):
        self.y = (InertiaTime * self.y_1 + SampleTime * self.u_5) / (
                    SampleTime + InertiaTime)
        # self.LastSystemOutput = self.SystemOutput
