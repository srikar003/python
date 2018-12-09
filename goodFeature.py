import numpy as np
import matplotlib.pyplot as plt

greyhounds=500
labradors=500

#let 28 be average height of greyhounds and make it random upto 4 values
grey_height=28+4*np.random.randn(greyhounds) 

#let 24 be average height of labradors and make it random upto 4 values
lab_height=24+4*np.random.randn(labradors)

plt.hist([grey_height,lab_height],color=['black','blue'])
plt.show()