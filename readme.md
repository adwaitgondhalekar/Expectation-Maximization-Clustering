# Project Title

Expectation Maximization Clustering

## Steps to run the code in Ubuntu

1.Look for a folder named "eigen" which is already provided.

2.Place this folder in your home directory.

3.Open a code editor such a VS Code and open a new terminal to start execution of the code.

4.Type
```
g++ -I "path_to_eigen_folder_in_home_directory" em_clustering.cpp
```
for eg.
```
g++ -I /home/adwait/eigen em_clustering.cpp
```

5.The code will get compiled, now type 
```
./a.out
```

6.Code execution will start.

7.In the code em_clustering.cpp there is a variable named "dimensions" of type Integer th value of this variable can be either 2 or 4.

Value = 

**2** indicates that the first 2 attributes in the dataset i.e. Sepal Length and Sepal Width are being considered for all the observations.
	
**4** idicates that all 4 attributes in the dataset (Sepal Length, Sepal Width, Petal Length, Petal Width) are being considered for all the observations.
