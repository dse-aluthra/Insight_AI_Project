import matplotlib
import pandas as pd
import numpy as np
import streamlit as st
import re
import argparse
# parse command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--filename", required=True, help="Data file name is needed")
args = vars(ap.parse_args())

# get data file name
fName = args["filename"]

f = open(fName)
s = f.readlines()

m = re.findall('\[(\d)\] Tesla K80 +\| +\d+\'C, +(\d+) % \| +(\d+)', '\n'.join(s))
df = pd.DataFrame(m, columns=['gpu', 'gpu%_usage', 'memory_usage'])

df = df.astype({'gpu': 'int32', 'gpu%_usage': 'int32', 'memory_usage': 'int32'})
df['time'] = np.repeat(range(int(len(m)/8)), 8)
df = df[df.gpu < 4]  # Dropping rows for gpu > 4
# st.write(df.head(16))
st.vega_lite_chart(df, mark='line', x_field='time', y_field='gpu%_usage', color_field='gpu', color_type='nominal')
st.vega_lite_chart(df, mark='bar', x_field='gpu', x_type='ordinal', y_field='memory_usage', y_scale_domain=[0, 11441], color_field='gpu', color_type='nominal')
