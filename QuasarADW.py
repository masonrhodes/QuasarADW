#Import the necessary modules 
import numpy as np
import matplotlib.pyplot as plt
import collections
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Ask the user to select the number of streamlines and sightlines they would like to test
stream = 100
sight = 100

# Make sure the parameters are acceptable.
# In order to loop through all intersection points we must set one parameter to always be larger
assert(stream >= sight),"The number of streamlines must be greater than or equal to the number of sightlines."

# Define the range that the streamlines can cover (in meters)
xstream_range = np.logspace(14, 16, 1000)

# Define the different positions that the streamlines begin based on user inputs (in meters)
stream_origin = np.linspace(1E14, 3.5E14, stream)

# Define the opening angles of each streamline based on user inputs (in meters)
open_angle = np.linspace(4E13, 7.2E13, stream)

# Define the range that the sightlines can cover (in meters)
xsight_range= np.logspace(14, 16, 1000)

# Define the positions of each sightline based on user inputs (in meters)
sight_num = np.linspace(-0.2E14, 0.2E14, sight)

# Define the terminal velocity of the streamlines (in meters)
v_term = 2.5E7

# Define a constant for the velocity equation from Murray and Chiang model
beta = 2

# Define the angle of the sightlines
theta = 5 * np.pi/180

# Define the streamlines with hyperbolic geometry
streamline = []
for i in range(stream):
    streamline.append(np.sqrt(open_angle[i]**2 * (xstream_range**2 / stream_origin[i]**2 - 1)))
streamline = np.array(streamline)

# Define the sightlines
sightline = []
for i in range(sight):
    sightline.append(np.tan(theta) * (xsight_range + sight_num[i]))
sightline = np.array(sightline)

# Define streamlines in terms of radial distance from source
rad_dist = []
for i in streamline:
    rad_dist.append(np.sqrt(xstream_range**2 + i**2))

# Define velocity magnitude along streamline
stream_vel = []
for i in range(stream):
    stream_vel.append(v_term * (1-(stream_origin[i] / rad_dist[i]))**beta)
    
# Find the indices of the 'intersection' points
intersect_index = np.zeros((sight, stream), dtype=int)
for i in range(sight):
    for j in range(stream):
        if np.nanmin(np.diff(np.sign(sightline[i]-streamline[j]))) != 0:
            intersect_index[i][j] = (np.nanargmin(np.diff(np.sign(sightline[i]-streamline[j]))))

# Find the x and y coordinates where the 'intersection' occurs
x = np.zeros((sight, stream))
y = np.zeros((sight, stream))
x = xstream_range[intersect_index]
for i in range(sight):
    for j in range(stream):
        y[i,j] = (streamline[j,intersect_index[i,j]])
        if y[i,j] == 0:
            y[i,j] = np.nan

# Find the radial distance from the source where the 'intersection' occurs
r = []
for i in range(len(x)):
        r.append(np.sqrt(x[i]**2 + y[i]**2))
r = np.array(r)

# Find the velocity magnitude at each intersection point
vel = []
for i in range(sight):
    for j in range(stream):
        vel.append(v_term*(1-(stream_origin[j]/r[i,j]))**beta)
vel = np.array(vel).reshape(sight, stream)

# Define function to find the derivative of streamline equation
def derivative(xstream_range, open_angle, stream_origin):
    return (open_angle**2 * xstream_range) / (stream_origin**2 * np.sqrt(open_angle**2 * (xstream_range**2 / stream_origin**2 - 1)))

# Find the tangent vector at each intersection point
tangent = []
for i in range(sight):
    for j in range(stream):
        tangent.append(derivative(x[i,j], open_angle[j], stream_origin[j]))
tangent = np.array(tangent).reshape(sight, stream)

# Finds the angle between sightline and streamline at intersection point
# From the equation relating difference of slopes to the angle between the line
phi = []
for i in range(sight):
    for j in range(stream):
        phi.append(np.arctan((tangent[i,j] - np.tan(theta)) / (1 + np.tan(theta) * tangent[i,j])))
phi = np.array(phi).reshape(sight, stream)

# Finds radial component of velocity at each intersection point
# These are the velocities we observe in our sight column
v_r = []
for i in range(sight):
    for j in range(stream):
        v_r.append(vel[i,j] * np.cos(phi[i,j]))
v_r = np.array(v_r).reshape(sight, stream)

# Create velocity bins and find indices of bins that radial velocites are in
bin_num = 100
bins = np.array(np.arange(0, v_term + v_term/bin_num, v_term/bin_num))
ind = []
for i in range(sight):
    ind.append(np.digitize(v_r[i,:], bins))
ind = np.array(ind)
vel_bin = []

# Calculate the actual radial velocties of bins that have entries
for i in range(sight):
    for j in range(stream):
        if ind[i,j] == len(bins):
            vel_bin.append(np.nan)
        else:
            vel_bin.append(bins[ind[i,j]])
vel_bin = np.array(vel_bin).reshape(sight, stream)

# Reduce the velocities in the same bins so that 
# along a streamline only one velocity is absorbed per bin
final_vel = []
for i in vel_bin:
    unique = np.unique(i)
    for j in unique:
        final_vel.append(j)
final_vels = list(collections.Counter(final_vel).keys()) #Use counter to retrieve bin velocities in order
count = collections.Counter(final_vel).values() #Use counter to retrieve the number of velocties sorted into bins

# Take the count of velocities in a bin divided by total number of
# streamlines to find absorption percentage
absorption = []
for i in count:
    absorption.append(i / sight)
    
# Plotting Streamlines and Sightlines
plt.figure(figsize = (20,10))
ax1 = plt.subplot()
for i in streamline:
    ax1.plot(xstream_range, i, color='k')
    plt.xlim(xmin=5E13, xmax=1E15)
    plt.ylim(ymin=5E11, ymax=5E14)
    for j in sightline:
        ax1.plot(xsight_range, j, color='b', alpha=0.1)
plt.show()

# Plot the synthetic absorption profile
plt.figure(figsize = (10,5))
ax2 = plt.subplot()
plt.xlim(xmin=0, xmax=v_term)
plt.ylim(ymin=0, ymax=1)
plt.grid(True, which='both', axis='both')
ax2.xaxis.set_major_locator(MultipleLocator(0.5e7))
ax2.xaxis.set_minor_locator(MultipleLocator(0.1e7))
ax2.yaxis.set_major_locator(MultipleLocator(0.2))
ax2.yaxis.set_minor_locator(MultipleLocator(0.04))
plt.xlabel('velocity m/s')
plt.ylabel('absorption')
ax2.scatter(np.array(final_vels), 1-np.array(absorption))
plt.show()

# Extracting f_deep from the synthetic plot
f_deep = np.amin(1-np.array(absorption))
v_deep = final_vels[np.nanargmin(1-np.array(absorption))]
#print(f_deep, v_deep)

#write to external file
absorption = 1-np.array(absorption)
tup = list(zip(final_vels, absorption))
absorption, final_vels = np.array(list(zip(*np.sort(tup))))
#with open('pi10/pi10_9b225.txt', 'w') as f:
#    for i in range(len(final_vels)):
#        f.write(str(final_vels[i]) + '\t' + str(absorption[i]) + '\n')
#f.close()

#f = open('pi45_b150.txt', 'r')
#lines = f.readlines()
#xplot, yplot = [], []
#for line in lines:
#    xplot.append(float(line.split()[0]))
#    yplot.append(float(line.split()[1]))
#f.close()

#read and plot from external file
#plt.figure(figsize = (10,5))
#ax3 = plt.subplot()
#plt.xlim(xmin=0, xmax=v_term)
#plt.ylim(ymin=0, ymax=1)
#ax3.scatter(xplot, yplot)
#plt.show()
#print(xplot, yplot)
