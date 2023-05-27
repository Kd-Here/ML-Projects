import sys
from src.logger import logging


# Function to generate a detailed error message
def error_message_detail(error, error_detail: sys):
    # Extracts the exception traceback information
    _, _, exc_tb = error_detail.exc_info()
    
    # Retrieves the file name where the error occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Constructs the error message with file name, line number, and error message
    error_message = "Error occurred in python script name [{0}] at line number [{1}] error message [{2}]".format(file_name, exc_tb.tb_lineno, str(error))
    
    return error_message


# Custom exception class
class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        # Inheriting from the parent class
        super().__init__(error_message)
        

        #Custom class error_message is vairable whose value is outcome of error_message_detail() function and it's parameters are given while function call
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        # Returns the error message when the exception is converted to a string
        return self.error_message

#For trying exception handling with logging them in exception
# if __name__ == "__main__":
#         try:
#             a = 1/0
#         except Exception as e:
#             print("HI",e)
#             logging.info("It's divide by zero!")
#             raise CustomException(e,sys)
