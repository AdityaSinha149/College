src = open( "file1.txt", "r" )
data = src.read()
src.close()

data = data[::-1]

dst = open( "output1.txt", "w+" )
dst.write(data)
dst.close()