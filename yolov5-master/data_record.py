import pandas as pd
from datetime import datetime
import time
def record(x,y,coord_record):
    current_time = datetime.now()
    coord_record.append([x,y,current_time])
    df = pd.DataFrame(coord_record, columns=['x coords', 'y coords', 'current time'])
    df.to_excel('./data_record/CoordRecordTest.xlsx')