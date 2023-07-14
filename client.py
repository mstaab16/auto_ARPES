import zmq
from messages import *
import numpy as np
import matplotlib.pyplot as plt
from fake_crystals import FakeGrapheneCrystal, FakeVoronoiCrystal

context = zmq.Context()


class FakeEndstation:
    def __init__(self, addr='tcp://localhost:5555'):
        self.socket = context.socket(zmq.REQ)
        self.socket.connect(addr)
        self.x = 0
        self.y = 0
        self.crystal = FakeVoronoiCrystal()
        # self.crystal = FakeGrapheneCrystal()
        self.measured_positions = []

    def send_message(self, message):
        self.socket.send(message.model_dump_json().encode('utf-8'))
        raw_recv = self.socket.recv()
        return decode_message(raw_recv)

    def begin_experiment(self, search_axes, data_formats):
        message = InitMessage(search_axes=search_axes, data_formats=data_formats)
        self.send_message(message)

    def send_data(self, data):
        message = DataMessage(format_name="ARPES", data=data.tolist())
        self.send_message(message)
    
    def query_position(self):
        message = QueryMessage()
        return self.send_message(message)

    def experiment_loop(self):
        for i in range(200):
            print("_"*10 + f"Measurement {i}" + "_"*10)
            print("Querying position")
            new_x, new_y = self.query_position().position
            if new_x != self.x or new_y != self.y:
                self.x = new_x
                self.y = new_y
            
            print(f"Measuring at {self.x}, {self.y}")
            measured_x, measured_y, arpes = self.crystal.measure(self.x, self.y)
            self.measured_positions.append([measured_x, measured_y])
            print("Sending data")
            self.send_data(arpes)

if __name__ == "__main__":
    # endstation = FakeEndstation("tcp://192.168.0.23:80")
    endstation = FakeEndstation()
    min_x, max_x, min_y, max_y = endstation.crystal.get_boundaries()
    search_axes=[
        SearchAxis(name="x", min=min_x, max=max_x, step=(max_x-min_x)/91),
        SearchAxis(name="y", min=min_y, max=max_y, step=(max_y-min_y)/91)
        ]
    data_formats=[
        #DataFormat(name='ARPES', dtype='float32', shape=[259,232], offset=[0,0], delta=[1,1]),
        DataFormat(name='ARPES', dtype='float32', shape=[128,128], offset=[0,0], delta=[1,1]),
        ]

    endstation.begin_experiment(search_axes, data_formats)
    endstation.experiment_loop()
    print("Sending shutdown command")
    endstation.send_message(ShutdownMessage())
    plt.scatter(*np.array(endstation.measured_positions).T)
    plt.show()

