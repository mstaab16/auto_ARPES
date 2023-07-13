import zmq
import time
from messages import *
from handlers import Experiment

def bind_zmq_socket(port: int=5555):
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{port}")
    return context, socket

def receive_message(socket):
    message = socket.recv()
    return decode_message(message)

def send_message(socket, message):
    socket.send(message.model_dump_json().encode())

def listen_loop(socket):
    print("Listening for messages...")
    while True:
        message = receive_message(socket)
        if isinstance(message, InitMessage):
            experiment = Experiment(message)
            send_message(socket, OkayResponse())
            print(experiment)
            continue
        elif isinstance(message, ShutdownMessage):
            response_message = experiment.handle_message(message)
            send_message(socket, response_message)
            print(experiment)
            break
        else:
            response_message = experiment.handle_message(message)
            send_message(socket, response_message)
            time.sleep(0.1)

def main():
    context, socket = bind_zmq_socket()
    listen_loop(socket)
    socket.close()
    context.term()

if __name__ == "__main__":
    main()

