from typing import Union
from pydantic import BaseModel, validator
import numpy as np
from datetime import datetime


class SearchAxis(BaseModel):
    name: str
    min: float
    max: float
    step: float


class DataFormat(BaseModel):
    name: str
    shape: list[int]
    offset: list[float]
    delta: list[float]
    dtype: str


class Message(BaseModel):
    type: str
    msg: str = ''

#     @validator("type")
#     def type_must_be_valid(cls, v):
#         if v not in VALID_MESSAGE_TYPES:
#             raise ValueError(f"Invalid message type {v}")
#         return v


class InitMessage(Message):
    type: str = 'InitMessage'
    search_axes: list[SearchAxis]
    data_formats: list[DataFormat]


class DataMessage(Message):
    type: str = 'DataMessage'
    format_name: str
    data: list


class MoveResponse(Message):
    type: str = 'MoveResponse'
    axes_to_move: list[str]
    position: list[float]


class OkayResponse(Message):
    type: str = 'OkayResponse'
    msg: str = 'ok'


class ErrorResponse(Message):
    type: str = 'ErrorResponse'
    msg: str = 'error'


class ShutdownMessage(Message):
    type: str = 'ShutdownMessage'
    msg: str = 'shutdown'


class QueryMessage(Message):
    type: str = 'QueryMessage'
    msg: str = 'query'


type_to_parser_dict = {
        b'InitMessage': InitMessage.parse_raw,
        b'DataMessage': DataMessage.parse_raw,
        b'QueryMessage': QueryMessage.parse_raw,
        b'ShutdownMessage': ShutdownMessage.parse_raw,
        b'OkayResponse': OkayResponse.parse_raw,
        b'ErrorResponse': ErrorResponse.parse_raw,
        b'MoveResponse': MoveResponse.parse_raw,
        }


def decode_message(message: bytes):
    for key, parser in type_to_parser_dict.items():
        if key in message:
            return parser(message)


if __name__ == "__main__":
    messages = [
            InitMessage(
                search_axes=[
                    SearchAxis(name='x', min=-1, max=1, step=0.01),
                    SearchAxis(name='y', min=-1, max=1, step=0.025),
                    ],
                data_formats=[
                    DataFormat(name='ARPES', dtype='float32', shape=[2,2], offset=[74,-15], delta=[0.05,0.1]),
                    ]
                ),
            DataMessage(
                format_name="ARPES",
                data=np.random.random((2,2)).tolist()
                ),
            QueryMessage(),
            MoveResponse(
                axes_to_move=['x', 'y'],
                position=[0.5, 0.5]
                ),
            OkayResponse(),
            ErrorResponse(),
            ShutdownMessage(),
            ]

    with open("messages.json", "w") as f:
        for message in messages:
            f.write(message.json())
            f.write("\n")
    
