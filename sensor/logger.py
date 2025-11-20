import logging
import os
from datetime import datetime
import os

#log file name
Log_file_name=f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.log"
# log directory
Log_file_dir=os.path.join(os.getcwd(),"logs")

#create folder if not avaiable
os.makedirs(Log_file_dir,exist_ok=True)

#log file path
Log_file_path=os.path.join(Log_file_dir,Log_file_name)

logging.basicConfig(
    filename=Log_file_path,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)