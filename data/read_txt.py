filename= "D:/weather_data_file.txt"
filenameoutput="D:/a.txt"
filenametime="D:/time.txt"
fp=open(filename,"r")
i=1
fpw=open(filenameoutput,"w")
fptime=open(filenametime,"w")

# Delete the time information, use genfromtxt read txt as matrix, get 11*9 every time
# Every time have 11*9 matrix
for line in fp.readlines():
    if i%12!=1:
        print(line)
        fpw.write(line)
        i+=1
    else:
        i+=1
        fptime.write(line)
##        print(line)
print(9600/12)
fp.close()
fpw.close()
fptime.close()
# read as matrix
from numpy import genfromtxt
my_data = genfromtxt('D:/a.txt', delimiter=' ')
print (my_data)