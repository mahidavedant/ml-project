"""
src/exception.py

Custom exceptions module for handling specific errors in the project.
"""
import sys
import logging


def error_message_detail(error, error_detail: sys):
    """
    Generate an error message with details.
    """
    _, _, exc_tb = error_detail.exc_info()
    filename = exc_tb.tb_frame.f_code.co_filename
    linenum = exc_tb.tb_lineno
    msg = str(error)

    error_msg = "Error occured in python script name[{0}] line[{1}] error message[{2}]".format(
        filename,
        linenum,
        error
    )

    return error_msg


class CustomException(Exception):
    def __init__(self, error_msg, error_detail: sys):
        super().__init__(error_msg)
        self.error_msg = error_message_detail(error_msg, error_detail)

    def __str__(self):
        return self.error_msg
