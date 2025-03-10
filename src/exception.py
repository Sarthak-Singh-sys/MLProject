import sys
import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb=error_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message = (
        f"Error occurred in Python script: [{file_name}] "
        f"at line number [{exc_tb.tb_lineno}] "
        f"with error message [{str(error)}]"
    )

    return error_message
    
class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_detail)
    
    def __str__(self):
        return self.error_message
    
# Example usage:
# if __name__ == "__main__":
#     try:
#         a = 1 / 0  # This will raise a ZeroDivisionError
#     except Exception as e:
#         logging.error(CustomException(e, sys))  # Logs the detailed custom exception

    


    


