Steps to run the code for Expectation Maximization Clustering (in Ubuntu)

1.Look for a folder named "eigen" which is already provided.

2.Place this folder in your home directory.

3.Open a code editor such a VS Code and open a new terminal to start execution of the code.

4.Type g++ -I <path to eigen folder in home directory> em_clustering.cpp

	for eg. g++ -I /home/adwait/eigen em_clustering.cpp

5.The code will get compiled now type ./a.out

6.Code execution will start.

7.In the code em_clustering.cpp there is a variable named "dimensions" of type Integer th value of this variable can be either 2 or 4.

	Value 2 indicates that the first 2 attributes in the dataset i.e. Sepal Length and Sepal Length are being considered for all the observations.
	
	Value 4 idicates that all 4 attributes in the dataset (SepalLength, Sepal Width, PetalLength, Petal Width) are being considered for all the observations.